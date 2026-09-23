"""sigma_mm by leave-one-training-subject-out for the option A fast stage (docs/SFLM/sigma_mm_2026-09-24.md, experiment 2).

For fold --holdout H, each training subject s is held out in turn: the population is the other two training subjects
(population_cont on cache Z_tr), the support set is n unique images of s (same permutation rule as the hold-out protocol),
the query set is 1,000 other unique images of s; the fast stage runs at each sigma of --sigma_grid and is scored by EV on the
ncsnr > 0.2 voxels of s (and by the converged log evidence).  Nothing about the hold-out subject enters.

    python scripts/sflm_sigma_loso.py --holdout 1 --out outputs/sflm_ab/h1 --sigma_grid 5 7 10 14 20 --num_ic 30 100
Writes <out>/sigma_loso.csv (one row per (held-out training subject, n, rep, sigma)).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.sflm.fast import TRAIN, support_offset  # noqa: E402
from src.sflm.fast_cont import EBcont, build_cache_cont, eb_fit_mackay_cont, population_cont  # noqa: E402
from src.sflm.latent import DATA, DEV  # noqa: E402
from src.utils.metrics import explained_variance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--out", default="outputs/sflm_ab/h1")
    ap.add_argument("--sigma_grid", type=float, nargs="+", default=[5, 7, 10, 14, 20])
    ap.add_argument("--num_ic", type=int, nargs="+", default=[30, 100])
    ap.add_argument("--reps", type=int, nargs="+", default=[0])
    ap.add_argument("--n_query", type=int, default=1000)
    ap.add_argument("--lam_pop", type=float, default=1.0)
    ap.add_argument("--jchunk", type=int, default=128)
    ap.add_argument("--sig2_init_scale", type=float, default=1.0)
    ap.add_argument("--em_iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--conv_consec", type=int, default=5)
    ap.add_argument("--ll_tol", type=float, default=0.01)
    ap.add_argument("--tau2_init", type=float, default=1.0)
    ap.add_argument("--guard", choices=["stop", "fallback"], default="stop")
    a = ap.parse_args()
    H = a.holdout
    out = Path(a.out)
    logf = open(out / "sigma_loso.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cache_path = out / "cache_latent.pt"
    cache = torch.load(cache_path) if cache_path.exists() else build_cache_cont(H, cache_path, str(out / "latent.pt"))
    rows = []
    res_path = out / "sigma_loso.csv"
    for s in TRAIN[H]:
        others = [t for t in TRAIN[H] if t != s]
        pop = population_cont(H, cache, a.lam_pop, subjects=others)
        vscale = pop["C"].var(0).mean().item()
        good = np.load(DATA / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2
        xyz_v = torch.from_numpy(np.load(DATA / f"s{s}_coords_mni.npy").astype(np.float32)).to(DEV)
        B = torch.from_numpy(np.load(DATA / f"s{s}_betas_unique.npy")).to(DEV)
        Z = cache["Z_tr"][s].to(DEV)
        order = np.random.default_rng(100).permutation(B.shape[0])
        qidx = torch.from_numpy(order[-a.n_query:]).to(DEV)                       # query: the last n_query of the permutation
        true = B[qidx].T
        log(f"[loso] held-out training subject s{s}: population {others} J {pop['A'].shape[0]} V {B.shape[1]} good {int(good.sum())}")
        for n in a.num_ic:
            for rep in a.reps:
                start = support_offset(n, rep)
                idx = torch.from_numpy(order[start:start + n]).to(DEV)
                Bs = B[idx]
                Bm = Bs.mean(0, keepdim=True)
                for sg in a.sigma_grid:
                    t0 = time.time()
                    with torch.inference_mode():
                        eb = EBcont(Z[idx], Bs - Bm, Z[qidx], pop, vscale, xyz_v, sg, jchunk=a.jchunk)
                        pred, diag = eb_fit_mackay_cont(eb, a, log)
                        pred += Bm.T
                        prof = eb.sigma_profile(a.sigma_grid)
                        del eb
                    ev = explained_variance(pred, true).cpu().numpy()
                    row = dict(subject=s, n=n, rep=rep, sigma=sg, ev=float(ev[good].mean()), tau2=diag["tau2"], loglik=diag["hist"][-1]["loglik"],
                               iters=diag["iters"], n_guard=diag["counters"]["n_guard"], sec=time.time() - t0,
                               **{f"ev_at_{k:g}": v for k, v in prof.items()})
                    rows.append(row)
                    log(f"[loso] s{s} n={n} rep={rep} sigma={sg:g} ev={row['ev']:.4f} tau2={row['tau2']:.4f} loglik={row['loglik']:.1f} iters={row['iters']} {row['sec']:.0f}s")
                    pd.DataFrame(rows).to_csv(res_path, index=False)
        del pop, B, Z
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    t = df.groupby(["n", "sigma"]).ev.mean().unstack("sigma")
    log("[loso] mean EV over held-out training subjects:\n" + t.round(4).to_string())
    log("[loso] selected sigma per n: " + str({int(n): float(t.loc[n].idxmax()) for n in t.index}))


if __name__ == "__main__":
    main()
