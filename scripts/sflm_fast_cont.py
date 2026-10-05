"""Option A fast stage (src/sflm/fast_cont.py) on a gauss latent map (scripts/sflm_latent_gauss.py), same protocol and
results.csv format as scripts/sflm_fast.py so scripts/sflm_eb_table.py / sflm_status.py can read the cells.

    python scripts/sflm_fast_cont.py --holdout 1 --out outputs/sflm_ab/h1 --tag A_cont --num_ic 30 100 --reps 0
n = 0 in --num_ic: the population prior of the hold-out subject (src.sflm.fast_cont.prior_predict at --sigma_mm), rep 0 only.
Coordinate ablations (docs/paper/experiment_plan_2026-09-26.md E3): --sigma_mm 1e6 (uniform kernel = no coordinates), --coord_mode
shuffle, --coord_shift <mm>.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from sflm_fast import write_rows  # noqa: E402
from src.sflm.fast import support_offset  # noqa: E402
from src.sflm.fast_cont import EBcont, build_cache_cont, eb_fit_mackay_cont, population_cont, prior_predict  # noqa: E402
from src.sflm.latent import DATA, DEV, STREAMS  # noqa: E402
from src.utils.metrics import explained_variance, pearson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--out", default="outputs/sflm_ab/h1")
    ap.add_argument("--latent", default=None, help="default <out>/latent.pt")
    ap.add_argument("--tag", default="A_cont")
    ap.add_argument("--num_ic", type=int, nargs="+", default=[30, 100, 300, 1000])
    ap.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sigma_mm", type=float, default=10.0, help="kernel width; with --sigma_grid, the starting value")
    ap.add_argument("--sigma_grid", type=float, nargs="*", default=None,
                    help="type-II ML for sigma_mm on the support set: fit at --sigma_mm, profile the evidence on this grid, refit at the "
                         "argmax (repeated --sigma_rounds times); the profile is logged either way")
    ap.add_argument("--sigma_rounds", type=int, default=2)
    ap.add_argument("--lam_pop", type=float, default=1.0)
    ap.add_argument("--jchunk", type=int, default=128)
    ap.add_argument("--sig2_init_scale", type=float, default=1.0)
    ap.add_argument("--em_iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--conv_consec", type=int, default=5)
    ap.add_argument("--ll_tol", type=float, default=0.01, help="#11 alt: an iteration also counts as converged when |d loglik| < ll_tol (nat)")
    ap.add_argument("--tau2_init", type=float, default=1.0)
    ap.add_argument("--guard", choices=["stop", "fallback"], default="stop", help="#12: on a log-likelihood decrease > 1 nat, stop (pilot) or fall back to the plain fixed-point step")
    ap.add_argument("--save_pred", default=None, help="directory: save the (V, 1000) test prediction as float16 h<H>_n<n>_r<rep>.npy")
    ap.add_argument("--save_var", default=None, help="directory: save the (V, 1000) posterior predictive variance (EBcont.predict_var), n > 0")
    ap.add_argument("--coord_mode", choices=["real", "shuffle"], default="real",
                    help="E3 ablation: shuffle permutes the MNI coordinates across the hold-out voxels (default_rng(--coord_seed))")
    ap.add_argument("--coord_seed", type=int, default=0)
    ap.add_argument("--coord_shift", type=float, default=0.0, help="E3 ablation: mm added to the x coordinate of every hold-out voxel")
    a = ap.parse_args()
    H = a.holdout
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"{a.tag}.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"[args] {vars(a)}")
    cache_path = out / "cache_latent.pt"
    cache = torch.load(cache_path) if cache_path.exists() else build_cache_cont(H, cache_path, a.latent or str(out / "latent.pt"))
    pop_path = out / f"pop_cont_l{a.lam_pop:g}.pt"
    if pop_path.exists():
        pop = {k: v.to(DEV) for k, v in torch.load(pop_path).items()}
    else:
        t0 = time.time()
        pop = population_cont(H, cache, a.lam_pop)
        torch.save({k: v.cpu() for k, v in pop.items()}, pop_path)
        log(f"[pop] {pop['A'].shape[0]} candidates in {time.time() - t0:.0f}s")
    vscale = pop["C"].var(0).mean().item()
    log(f"[pop] J {pop['A'].shape[0]} vscale {vscale:.5f} sigma range {pop['A'].max(1).values.min():.3f}-{pop['A'].max(1).values.max():.3f} (max mask weight)")

    meta = np.load(DATA / f"s{H}_voxel_meta.npz")
    good = meta["ncsnr"] > 0.2
    xyz_np = np.load(DATA / f"s{H}_coords_mni.npy").astype(np.float32)
    if a.coord_mode == "shuffle":
        xyz_np = xyz_np[np.random.default_rng(a.coord_seed).permutation(len(xyz_np))]
    xyz_np[:, 0] += a.coord_shift
    xyz_v = torch.from_numpy(xyz_np).to(DEV)
    B_u = torch.from_numpy(np.load(DATA / f"s{H}_betas_unique.npy")).to(DEV)
    true = torch.from_numpy(np.load(DATA / f"s{H}_betas_common.npy")).to(DEV).T
    Z_u, Z_q = cache["Z_u"].to(DEV), cache["Z_q"].to(DEV)
    order = np.random.default_rng(100).permutation(B_u.shape[0])
    info = dict(mode="eb_cont", args=vars(a), vscale=vscale)
    res_path = out / "results.csv"
    for n in a.num_ic:
        for rep in ([0] if n == 0 else a.reps):
            start = support_offset(n, rep)
            idx = torch.from_numpy(order[start:start + n]).to(DEV)
            t0 = time.time()
            sigma_sel, profiles, pvar = a.sigma_mm, [], None
            if n == 0:                                   # K = 0: population prior of the new subject, no support data
                with torch.inference_mode():
                    pred = prior_predict(pop, xyz_v, a.sigma_mm, Z_q)
                diag = dict(tau2=np.nan, sig2_med=np.nan, iters=0, n_estep=0, converged=True, r_max_med=np.nan,
                            counters=dict(n_floor=0, n_clamp=0, n_clip=0, n_guard=0, n_tau2_zero=0))
            else:
                Bs = B_u[idx]
                Bm = Bs.mean(0, keepdim=True)
                with torch.inference_mode():
                    eb = EBcont(Z_u[idx], Bs - Bm, Z_q, pop, vscale, xyz_v, a.sigma_mm, jchunk=a.jchunk)
                    pred, diag = eb_fit_mackay_cont(eb, a, log)
                    if a.sigma_grid:
                        for rnd in range(a.sigma_rounds):
                            prof = eb.sigma_profile(a.sigma_grid)
                            best = max(prof, key=prof.get)
                            profiles.append(dict(sigma=sigma_sel, tau2=diag["tau2"], profile=prof, argmax=best))
                            log(f"[sigma] round {rnd} at sigma {sigma_sel:g} tau2 {diag['tau2']:.5f}: " + " ".join(f"{k:g}:{v:.1f}" for k, v in prof.items()) + f" -> {best:g}")
                            if best == sigma_sel:
                                break
                            sigma_sel = best
                            eb.set_sigma(sigma_sel)
                            pred, diag = eb_fit_mackay_cont(eb, a, log)
                        diag["sigma_profiles"] = profiles
                        diag["sigma_edge"] = bool(sigma_sel in (min(a.sigma_grid), max(a.sigma_grid)))
                    pred += Bm.T
                    sig2_v = diag.pop("sig2")
                    if a.save_var:
                        pvar = eb.predict_var(diag["tau2"], sig2_v)
                    del eb
            torch.cuda.synchronize()
            sec = time.time() - t0
            ev, r = explained_variance(pred, true).cpu().numpy(), pearson(pred, true).cpu().numpy()
            np.save(out / f"{a.tag}_n={n}_rep={rep}_ev.npy", ev.astype(np.float32))
            if a.save_pred:
                Path(a.save_pred).mkdir(parents=True, exist_ok=True)
                np.save(Path(a.save_pred) / f"h{H}_n{n}_r{rep}.npy", pred.half().cpu().numpy())
            if pvar is not None:
                Path(a.save_var).mkdir(parents=True, exist_ok=True)
                np.save(Path(a.save_var) / f"h{H}_n{n}_r{rep}.npy", pvar.half().cpu().numpy())
                ratio = (pvar.mean(1) / (true - pred).var(1)).cpu().numpy()          # predicted / realised residual variance
                log(f"[var] n={n} rep={rep} median over ncsnr>0.2 voxels of mean predicted var / test residual var {np.median(ratio[good]):.3f}")
            row = dict(tag=a.tag, mode="eb_cont", n=n, rep=rep, ev=float(ev[good].mean()), r=float(r[good].mean()), ev_all=float(ev.mean()),
                       sec=sec, tau2=diag["tau2"], alpha=np.nan, sig2_med=diag["sig2_med"], iters=diag["iters"], evals=np.nan,
                       n_estep=diag["n_estep"], converged=diag["converged"], lam_pop=a.lam_pop, sigma_mm=sigma_sel,
                       sigma_start=a.sigma_mm, sigma_edge=diag.get("sigma_edge", False), sigma_rounds=len(profiles),
                       r_max_med=diag["r_max_med"])
            row.update({k: diag["counters"][k] for k in ("n_floor", "n_clamp", "n_clip", "n_guard", "n_tau2_zero")})
            row.update({f"ev_{k}": float(ev[good & np.isin(meta["streams"], v)].mean()) for k, v in STREAMS.items()})
            info[f"diag_n={n}_rep={rep}"] = diag
            log(f"[{a.tag}] n={n} rep={rep} ev={row['ev']:.4f} r={row['r']:.4f} early={row['ev_early']:.4f} tau2={row['tau2']:.5f} sigma={sigma_sel:g} "
                f"iters={row['iters']} r_max_med={row['r_max_med']:.3f} floor/clamp/clip/guard={row['n_floor']}/{row['n_clamp']}/{row['n_clip']}/{row['n_guard']} {sec:.0f}s")
            write_rows(res_path, [row])
    with open(out / f"{a.tag}_info.json", "w") as f:
        json.dump(info, f, indent=1, default=float)


if __name__ == "__main__":
    main()
