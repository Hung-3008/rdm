"""Test H1 of docs/_archive/plan.md on the A_cont branch: prediction at coordinates of the held-out subject whose betas are masked.

Per (fold, n, rep) the A_cont fast stage (src/sflm/fast_cont.py) is fit once on every voxel. Given tau2, the fast stage is
independent per voxel (r_vj, sig2_v and the posterior of voxel v use only the betas of v), so the fit of the unmasked voxels
equals a fit on the unmasked voxels alone with tau2 fixed; tau2 is the only quantity that also saw masked betas
(--strict_check refits on the unmasked voxels only and reports the difference). For every mask, a masked voxel t
(ncsnr > 0.2) is predicted from
    M0      population prior only: y0_t = sum_j w_tj u_j(x)' C_j, w = normalised MNI kernel (sigma_mm), no data of the subject
    NN      the fitted prediction of the nearest unmasked voxel
    KA_h    Gaussian kernel average (h mm) of the fitted predictions of the unmasked voxels
    KAD_h   M0 + Gaussian kernel average (h mm) of the deviations D_v = y_v - y0_v of the unmasked voxels
    KRIG    M0 + simple kriging of D from the k nearest unmasked voxels; the covariance c(d) = a rho(d; l) + b (+ nugget)
            is fit on the unmasked voxels' deviations only (Gaussian or exponential rho, chosen by weighted least squares)
    own     A_cont fit with the voxel's own support betas (upper bound, E_own)
KA_h and KAD_h are reported over a grid of h: the best h is chosen on the hold-out (oracle, an upper bound for that family).
Predictions and EV are on the 1,000 common images; EV is offset-invariant, so the unknown support mean of t does not matter.

    python scripts/sflm_inpaint_cont.py --holdout 1 --num_ic 100 --reps 0 1 2
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.sflm.fast import support_offset  # noqa: E402
from src.sflm.fast_cont import EBcont, eb_fit_mackay_cont  # noqa: E402
from src.sflm.latent import DATA, DEV, STREAMS  # noqa: E402
from src.utils.metrics import explained_variance  # noqa: E402

H_GRID = (2.0, 3.0, 5.0, 7.0, 10.0, 14.0)
DIST_BINS = (0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 1e9)


@torch.inference_mode()
def prior_prediction(pop, xyz_v, sigma, Z_q, jchunk=256):
    """(V, Q) M0 prediction: posterior at n = 0, responsibilities = prior w_vj, coefficients = C_j."""
    xyz_p, pn, xv2 = pop["xyz"], (pop["xyz"] ** 2).sum(1), (xyz_v ** 2).sum(1, keepdim=True)
    J, P, K = pop["A"].shape[0], pop["A"].shape[1], pop["C"].shape[1]
    logZ = torch.full((xyz_v.shape[0],), -float("inf"), device=DEV)
    for a in range(0, J, jchunk):
        lw = -0.5 * (xv2 - 2 * xyz_v @ xyz_p[a:a + jchunk].T + pn[a:a + jchunk][None]) / sigma ** 2
        logZ = torch.logaddexp(logZ, torch.logsumexp(lw, 1))
    W = torch.zeros(xyz_v.shape[0], P * K, device=DEV)
    off = torch.zeros(xyz_v.shape[0], device=DEV)
    for a in range(0, J, jchunk):
        sl = slice(a, a + jchunk)
        w = torch.exp(-0.5 * (xv2 - 2 * xyz_v @ xyz_p[sl].T + pn[sl][None]) / sigma ** 2 - logZ[:, None])  # (V, Jc)
        cs = pop["C"][sl] / pop["std"][sl]                                                                  # (Jc, K)
        W += w @ torch.einsum("jp,jk->jpk", pop["A"][sl], cs).reshape(-1, P * K)
        off += w @ (pop["mean"][sl] * cs).sum(1)
    return (Z_q.reshape(Z_q.shape[0], -1) @ W.T).T - off[:, None]


def fit_covariance(D, xyz, rng, n_anchor=3000, dmax=40.0, width=1.0):
    """Isotropic covariance of the deviation field from voxel pairs (anchors x all): binned <D_u, D_v> / Q, then weighted
    least squares c(d) = a rho(d; l) + b over d in (0, dmax], rho Gaussian or exponential, a, b >= 0; nugget = c(0) - a - b."""
    Q = D.shape[1]
    anc = torch.from_numpy(rng.choice(D.shape[0], min(n_anchor, D.shape[0]), replace=False)).to(DEV)
    cov = (D[anc] @ D.T) / Q                                                         # (A, N)
    dist = torch.cdist(xyz[anc], xyz)
    keep = (dist > 0) & (dist <= dmax)
    b_idx = torch.clamp((dist[keep] / width).long(), max=int(dmax / width) - 1)
    nb = int(dmax / width)
    cnt = torch.zeros(nb, device=DEV).index_add_(0, b_idx, torch.ones_like(b_idx, dtype=torch.float32))
    sm = torch.zeros(nb, device=DEV).index_add_(0, b_idx, cov[keep])
    dsum = torch.zeros(nb, device=DEV).index_add_(0, b_idx, dist[keep])
    ok = cnt > 0
    d, c, wt = (dsum[ok] / cnt[ok]).cpu().numpy(), (sm[ok] / cnt[ok]).cpu().numpy(), cnt[ok].cpu().numpy()
    c0 = float((D ** 2).sum(1).mean() / Q)
    best = None
    for fam in ("gauss", "exp"):
        for ell in np.geomspace(0.5, 60, 120):
            rho = np.exp(-0.5 * (d / ell) ** 2) if fam == "gauss" else np.exp(-d / ell)
            X = np.stack([rho, np.ones_like(rho)], 1) * np.sqrt(wt)[:, None]
            y = c * np.sqrt(wt)
            for cols in ([0, 1], [0]):                                                   # nnls over 2 params by enumeration
                coef, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
                if (coef < 0).any():
                    continue
                full = np.zeros(2)
                full[cols] = coef
                sse = float(((X @ full - y) ** 2).sum())
                if best is None or sse < best["sse"]:
                    best = dict(fam=fam, ell=float(ell), a=float(full[0]), b=float(full[1]), sse=sse)
    best["c0"] = c0
    best["nugget"] = max(c0 - best["a"] - best["b"], 1e-6 * c0)
    best["bins"] = dict(d=d.tolist(), c=c.tolist(), n=wt.tolist())
    return best


def rho_fn(cv, d):
    return torch.exp(-0.5 * (d / cv["ell"]) ** 2) if cv["fam"] == "gauss" else torch.exp(-d / cv["ell"])


@torch.inference_mode()
def predict_masked(P_own, P0, xyz, U, T, cv, k=64, tchunk=1024):
    """Predictions (|T|, Q) at target voxels T from unmasked voxels U for every method; plus distance to the nearest U."""
    xu, D_u, Pu = xyz[U], (P_own - P0)[U], P_own[U]
    out = {m: [] for m in ["NN", "KRIG"] + [f"KA_{h:g}" for h in H_GRID] + [f"KAD_{h:g}" for h in H_GRID]}
    dnn = []
    for a in range(0, len(T), tchunk):
        t = T[a:a + tchunk]
        dist = torch.cdist(xyz[t], xu)                                                   # (Tc, Nu)
        dmin, imin = dist.min(1)
        dnn.append(dmin)
        out["NN"].append(Pu[imin])
        for h in H_GRID:
            w = torch.softmax(-0.5 * (dist / h) ** 2, 1)
            out[f"KA_{h:g}"].append(w @ Pu)
            out[f"KAD_{h:g}"].append(P0[t] + w @ D_u)
        dk, ik = dist.topk(k, 1, largest=False)                                          # (Tc, k)
        xk = xu[ik]                                                                       # (Tc, k, 3)
        Ckk = cv["a"] * rho_fn(cv, torch.cdist(xk, xk)) + cv["b"] + cv["nugget"] * torch.eye(k, device=DEV)
        ck = cv["a"] * rho_fn(cv, dk) + cv["b"]
        lam = torch.linalg.solve(Ckk, ck[:, :, None])[:, :, 0]                           # (Tc, k)
        out["KRIG"].append(P0[t] + torch.einsum("tk,tkq->tq", lam, D_u[ik]))
    return {m: torch.cat(v) for m, v in out.items()}, torch.cat(dnn)


def masks(xyz, good, streams):
    """{scheme: [(name, mask)]}; within a scheme the masks partition the ncsnr > 0.2 voxels (A: two complementary halves)."""
    z = xyz[:, 2]
    half = np.random.default_rng(0).random(len(good)) < 0.5
    out = {"A_random50": [("A_half0", half), ("A_half1", ~half)]}
    for thick in (10, 20):
        z0 = np.floor(z[good].min())
        out[f"B_slab{thick}"] = [(f"B_slab{thick}_z{int(lo)}", (z >= lo) & (z < lo + thick))
                                 for lo in np.arange(z0, z[good].max() + 1e-6, thick)]
    out["C_stream"] = [(f"C_{nm}", np.isin(streams, v)) for nm, v in STREAMS.items()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--out", default=None, help="default outputs/sflm_ab/h<holdout>")
    ap.add_argument("--num_ic", type=int, nargs="+", default=[100])
    ap.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sigma_mm", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=64, help="neighbours used by KRIG")
    ap.add_argument("--strict_check", nargs="*", default=[], help="mask names refit on the unmasked voxels only")
    a = ap.parse_args()
    H = a.holdout
    out = Path(a.out or f"outputs/sflm_ab/h{H}")
    res_dir = out / "inpaint"
    res_dir.mkdir(exist_ok=True)
    solver = SimpleNamespace(sig2_init_scale=1.0, em_iters=200, tol=1e-3, conv_consec=5, ll_tol=0.01, tau2_init=1.0, guard="stop")

    def log(msg):
        print(msg, flush=True)

    cache = torch.load(out / "cache_latent.pt")
    pop = {k: v.to(DEV) for k, v in torch.load(out / "pop_cont_l1.pt").items()}
    vscale = pop["C"].var(0).mean().item()
    meta = np.load(DATA / f"s{H}_voxel_meta.npz")
    good = meta["ncsnr"] > 0.2
    xyz_np = np.load(DATA / f"s{H}_coords_mni.npy").astype(np.float32)
    xyz = torch.from_numpy(xyz_np).to(DEV)
    B_u = torch.from_numpy(np.load(DATA / f"s{H}_betas_unique.npy")).to(DEV)
    true = torch.from_numpy(np.load(DATA / f"s{H}_betas_common.npy")).to(DEV).T
    Z_u, Z_q = cache["Z_u"].to(DEV), cache["Z_q"].to(DEV)
    order = np.random.default_rng(100).permutation(B_u.shape[0])
    t0 = time.time()
    P0 = prior_prediction(pop, xyz, a.sigma_mm, Z_q)
    P0 = P0 - P0.mean(1, keepdim=True)
    ev0 = explained_variance(P0, true).cpu().numpy()
    log(f"[M0] fold {H}: E0 (all good voxels) {ev0[good].mean():.4f} in {time.time() - t0:.0f}s")
    np.save(res_dir / "M0_ev.npy", ev0.astype(np.float32))
    schemes = masks(xyz_np, good, meta["streams"])
    rows, cov_info = [], {}
    for n in a.num_ic:
        for rep in a.reps:
            t0 = time.time()
            idx = torch.from_numpy(order[support_offset(n, rep):support_offset(n, rep) + n]).to(DEV)
            Bs = B_u[idx]
            Bc = Bs - Bs.mean(0, keepdim=True)
            with torch.inference_mode():
                eb = EBcont(Z_u[idx], Bc, Z_q, pop, vscale, xyz, a.sigma_mm)
                P_own, diag = eb_fit_mackay_cont(eb, solver, lambda m: None)
                del eb
            P_own = P_own - P_own.mean(1, keepdim=True)
            ev_own = explained_variance(P_own, true).cpu().numpy()
            ref = out / f"A_cont_n={n}_rep={rep}_ev.npy"
            chk = float(np.abs(ev_own - np.load(ref)).max()) if ref.exists() else float("nan")
            log(f"[own] n={n} rep={rep} E_own {ev_own[good].mean():.4f} tau2 {diag['tau2']:.5f} max |ev - saved A_cont| {chk:.1e} "
                f"({time.time() - t0:.0f}s)")
            rng = np.random.default_rng(rep)
            for scheme, mlist in schemes.items():
                ev_s = {}                                        # method -> per-voxel EV over the scheme's targets
                dn_s = np.full(len(good), np.nan, np.float32)
                for name, m in mlist:
                    T_np = np.where(m & good)[0]
                    U_np = np.where(~m)[0]
                    if len(T_np) == 0:
                        continue
                    T, U = torch.from_numpy(T_np).to(DEV), torch.from_numpy(U_np).to(DEV)
                    cv = fit_covariance((P_own - P0)[U], xyz[U], rng)
                    cov_info[f"n={n}_rep={rep}_{name}"] = {k: v for k, v in cv.items() if k != "bins"} | (
                        {"bins": cv["bins"]} if name in ("B_slab20_z-25", "A_half0") else {})
                    preds, dnn = predict_masked(P_own, P0, xyz, U, T, cv, k=a.k)
                    preds["M0"], preds["own"] = P0[T], P_own[T]
                    evm = {mt: explained_variance(p, true[T]).cpu().numpy() for mt, p in preds.items()}
                    for mt, e in evm.items():
                        ev_s.setdefault(mt, np.full(len(good), np.nan, np.float32))[T_np] = e
                    dn_s[T_np] = dnn.cpu().numpy()
                    rows.append(dict(fold=H, n=n, rep=rep, scheme=scheme, mask=name, n_vox=len(T_np), n_unmasked=len(U_np),
                                     d_nn_med=float(dnn.median()), cov_fam=cv["fam"], cov_ell=cv["ell"],
                                     cov_a=cv["a"], cov_b=cv["b"], cov_nug=cv["nugget"], **{f"E_{mt}": float(e.mean()) for mt, e in evm.items()}))
                    if name in a.strict_check:
                        with torch.inference_mode():
                            ebU = EBcont(Z_u[idx], Bc[:, U], Z_q, pop, vscale, xyz[U], a.sigma_mm)
                            PU, dU = eb_fit_mackay_cont(ebU, solver, lambda m: None)
                            del ebU
                        P_strict = P_own.clone()
                        P_strict[U] = PU - PU.mean(1, keepdim=True)
                        cvs = fit_covariance((P_strict - P0)[U], xyz[U], np.random.default_rng(rep))
                        ps, _ = predict_masked(P_strict, P0, xyz, U, T, cvs, k=a.k)
                        e_s = explained_variance(ps["KRIG"], true[T]).cpu().numpy().mean()
                        log(f"[strict] n={n} rep={rep} {name}: tau2 full {diag['tau2']:.5f} unmasked-only {dU['tau2']:.5f}; "
                            f"E_KRIG {evm['KRIG'].mean():.4f} strict {e_s:.4f} (diff {e_s - evm['KRIG'].mean():+.5f})")
                np.savez_compressed(res_dir / f"ev_{scheme}_n={n}_rep={rep}.npz", d_nn=dn_s, **ev_s)
                sel = good & ~np.isnan(ev_s["M0"])
                log(f"[{scheme}] n={n} rep={rep} " + " ".join(f"{mt} {e[sel].mean():.4f}" for mt, e in ev_s.items()
                                                                if not mt.startswith(("KA_", "KAD_")))
                    + " | best KA " + f"{max(ev_s[f'KA_{h:g}'][sel].mean() for h in H_GRID):.4f}"
                    + " best KAD " + f"{max(ev_s[f'KAD_{h:g}'][sel].mean() for h in H_GRID):.4f}")
            log(f"[cell] n={n} rep={rep} done in {time.time() - t0:.0f}s")
            pd.DataFrame(rows).to_csv(res_dir / "masks.csv", index=False)
            with open(res_dir / "cov.json", "w") as f:
                json.dump(cov_info, f, indent=1)


if __name__ == "__main__":
    main()
