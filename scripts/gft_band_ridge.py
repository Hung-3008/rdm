"""Ridge in the spatial-frequency domain of the beta map (hold-out S1, prior from S2/S5/S7; 2026-09-18).
Transform: graph Fourier basis U (V x V orthonormal) = eigenvectors of L = I - D^-1/2 W D^-1/2 of the 26-neighbour
voxel graph on the native ijk grid (edge weight exp(-d^2/2), d in voxels); A = B U, inverse B = A U'. Cached in outputs/gft/.
Features: unit-norm CLIP-512, or with --basis the standardised K-dim features of a FunctionalBasis.
  M1 spectrum of test betas and of full-data predictability per band
  M2 invariance check + few-shot ridge with shared / per-voxel / per-band / per-mode lambda, truncation, oracle
  M3 cross-subject (MNI) correspondence of weight maps per band
  M4 few-shot ridge shrunk toward the MNI population prior with per-band (lambda, gamma)

    python scripts/gft_band_ridge.py [--basis outputs/repro/clip_b16/popeb_s1/K32/basis.pt]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from src.model.lowrank_basis import FunctionalBasis  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--basis", default=None)
args = ap.parse_args()
D = Path("data/braincorl_nsdgeneral"); F = D / "clip_b16"
H, TRAIN, SIG = 1, [2, 5, 7], 2.0
torch.set_grad_enabled(False)
dev = torch.device("cuda")
# last lambda = predict the support mean; grids for unit-norm CLIP features / standardised basis features
LAMS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 1e9] if args.basis else [1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100, 1e6]
GAMS = [0.0, 0.5, 1.0]
NS, REPS, NBAND = [30, 100, 300, 1000], 5, 12
NL = len(LAMS)
t0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = False

unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
meta = np.load(D / f"s{H}_voxel_meta.npz"); ijk = meta["ijk"]; V = len(ijk)
cache = Path(f"outputs/gft/s{H}_V{V}.npz")
if not cache.exists():
    pairs = cKDTree(ijk).query_pairs(1.8, output_type="ndarray")
    L = torch.zeros(V, V, dtype=torch.float64, device=dev)
    i, j = T(pairs.T)
    L[i, j] = L[j, i] = T(np.exp(-0.5 * ((ijk[pairs[:, 0]] - ijk[pairs[:, 1]]).astype(np.float64) ** 2).sum(1)))
    dm = L.sum(1).clamp_min(1e-12).rsqrt()
    L.mul_(-dm[:, None]).mul_(dm[None, :]).diagonal().add_(1.0)
    ev_, U_ = torch.linalg.eigh(L)
    cache.parent.mkdir(parents=True, exist_ok=True); np.savez(cache, ev=ev_.cpu().numpy(), U=U_.float().cpu().numpy())
    del L, U_; torch.cuda.empty_cache()
z = np.load(cache); evals = z["ev"]; U = T(z["U"])
edges = np.r_[0, np.unique(np.round(np.logspace(np.log10(8), np.log10(V), NBAND)).astype(int))]
band_np = np.searchsorted(edges, np.arange(V), side="right") - 1; NB = band_np.max() + 1
band = T(band_np)
good = T(meta["ncsnr"] > 0.2)
print(f"V={V} good={int(good.sum())} bands={NB} {time.time()-t0:.0f}s", flush=True)

raw = lambda name: unit(np.load(F / name).astype(np.float32))
if args.basis:
    basis = FunctionalBasis.from_file(args.basis, dev)
    _f = basis(T(raw(f"s{TRAIN[0]}_unique_img_feats_normed.npy"))); MU, SD = _f.mean(0), _f.std(0).clamp_min(1e-6)
    emb = lambda name: ((basis(T(raw(name))) - MU) / SD).cpu().numpy()
else:
    emb = raw
load_X = lambda s: emb(f"s{s}_unique_img_feats_normed.npy")
load_B = lambda s: np.load(F / f"s{s}_unique_nrn_feats_msk.npy").astype(np.float32)


def full_W(X, B, lam=1.0):
    Xc = X - X.mean(0)
    return torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(X.shape[1], device=dev), Xc.T @ (B - B.mean(0)))   # (512, V)


def ev_good(pred, true):
    e = 1 - ((true - pred) ** 2).mean(0) / true.var(0, unbiased=False).clamp_min(1e-12)
    return float(e[good].mean())


def paths(X, B, Xq, g=0.0, C0=None):
    """Ridge (toward g*C0) for every lambda: query predictions (NL, nq, V), LOO residuals (NL, n, V), support mean."""
    xm, bm = X.mean(0), B.mean(0); Xc, Xqc = X - xm, Xq - xm
    R = B - bm; base = bm.expand(len(Xq), -1)
    if g:
        R = R - g * (Xc @ C0); base = base + g * (Xqc @ C0)
    s2, Q = torch.linalg.eigh((Xc @ Xc.T).double()); s2 = s2.clamp_min(0).float(); Q = Q.float()
    QtR = Q.T @ R; Kq = Xqc @ Xc.T @ Q; n = len(X); P, E = [], []
    for lam in LAMS:
        k = 1 / (s2 + lam)
        P.append(base + (Kq * k) @ QtR)
        h = ((Q ** 2) * (s2 * k)).sum(1) + 1 / n
        E.append((R - (Q * (s2 * k)) @ QtR) / (1 - h).clamp_min(0.02)[:, None])
    return torch.stack(P), torch.stack(E), bm


def take(PA, c):
    return torch.gather(PA, 0, c[None, None, :].expand(1, PA.shape[1], -1))[0]


def per_band(score):
    """score (ncfg, V) -> (V,) index of the best config of each band."""
    c = torch.zeros(V, dtype=torch.long, device=dev)
    for b in range(NB):
        m = band == b
        c[m] = score[:, m].sum(1).argmin()
    return c


# ---------------------------------------------------------------- MNI population prior (CPU sparse, then GPU)
ch = np.load(F / f"s{H}_coords.npy"); C0 = None
for s in TRAIN:
    Ws = full_W(T(load_X(s)), T(load_B(s))).cpu().numpy(); cs = np.load(F / f"s{s}_coords.npy")
    M = cKDTree(ch).sparse_distance_matrix(cKDTree(cs), 3 * SIG, output_type="coo_matrix")
    G = sp.csr_matrix((np.exp(-0.5 * (M.data / SIG) ** 2), (M.row, M.col)), shape=M.shape)
    rs = np.asarray(G.sum(1)).ravel(); empty = rs == 0
    Cs = np.asarray((sp.diags(1 / np.maximum(rs, 1e-12)) @ G @ Ws.T).T, dtype=np.float32)
    if empty.any():
        Cs[:, empty] = Ws[:, cKDTree(cs).query(ch[empty])[1]]
    C0 = T(Cs) / len(TRAIN) if C0 is None else C0 + T(Cs) / len(TRAIN); torch.cuda.empty_cache()
    print(f"prior from S{s}: {int(empty.sum())} hold-out voxels with no neighbour within {3*SIG:g} mm  {time.time()-t0:.0f}s", flush=True)

Xu, Bu = T(load_X(H)), T(load_B(H))
Xq = T(emb("common_img_feats_normed.npy"))
Bq = T(np.load(F / f"s{H}_common_nrn_feats_msk.npy").astype(np.float32))

# ---------------------------------------------------------------- M1
best = (-1, None)
for lam in [1, 3, 10, 30, 100, 1000]:
    Wf = full_W(Xu, Bu, lam); Pf = Bu.mean(0) + (Xq - Xu.mean(0)) @ Wf
    best = max(best, (ev_good(Pf, Bq), lam))
Wf = full_W(Xu, Bu, best[1]); Pf = Bu.mean(0) + (Xq - Xu.mean(0)) @ Wf
print(f"\n[M1] full-data ridge (9000 img) lam={best[1]}: EV ncsnr>0.2 = {best[0]:.4f}")
Aq, Af = (Bq - Bq.mean(0)) @ U, (Pf - Bq.mean(0)) @ U; tot = (Aq ** 2).sum(); cum = 0
print("band  modes  mean eigval  %energy(test betas)   cum%   EV of full-data ridge within band")
for b in range(NB):
    m = band == b; e = float((Aq[:, m] ** 2).sum() / tot); cum += e
    evb = float(1 - ((Aq[:, m] - Af[:, m]) ** 2).sum() / (Aq[:, m] ** 2).sum())
    print(f"{b:3d}  {int(m.sum()):6d}  {evals[band_np == b].mean():9.4f}  {100*e:12.2f}  {100*cum:12.2f}  {evb:12.4f}")

# ---------------------------------------------------------------- M3
half = len(Xu) // 2
Wa, Wb, Wp = full_W(Xu[:half], Bu[:half]) @ U, full_W(Xu[half:], Bu[half:]) @ U, C0 @ U
cos = lambda A, B_: float((A * B_).sum() / ((A ** 2).sum() * (B_ ** 2).sum()).sqrt())
print("\n[M3] cosine between weight maps (features x modes) per band; halves = disjoint 4500-image fits of S1")
print("band  modes  r_self(half a, half b)  r_cross(half a, MNI prior S2/S5/S7)  r_cross/r_self")
for b in range(NB):
    m = band == b; rs_, rc = cos(Wa[:, m], Wb[:, m]), cos(Wa[:, m], Wp[:, m])
    print(f"{b:3d}  {int(m.sum()):6d}  {rs_:12.3f}  {rc:24.3f}  {rc/max(rs_,1e-6):18.2f}")
del Wa, Wb, Wp, Wf, Pf, Aq, Af; torch.cuda.empty_cache()

# ---------------------------------------------------------------- M2 + M4
order = torch.from_numpy(np.random.default_rng(0).permutation(len(Xu))).to(dev)
names = ["vox_shared", "vox_pervoxel", "spec_shared", "spec_perband", "spec_permode", "spec_trunc", "spec_perband_ORACLE",
         "prior_global", "prior_perband", "prior_perband_ORACLE"]
for n in NS:
    res = {k: [] for k in names}; inv_err = []; lam_log = []; gam_log = []
    for r in range(REPS):
        idx = order[r * n:(r + 1) * n]; PAs, LMs = [], []
        for g in GAMS:
            P, E, bm = paths(Xu[idx], Bu[idx], Xq, g, C0)
            if g == 0.0:
                loo_v = (E ** 2).mean(1); k0 = int(loo_v.mean(1).argmin())
                res["vox_shared"].append(ev_good(P[k0], Bq))
                res["vox_pervoxel"].append(ev_good(take(P, loo_v.argmin(0)), Bq)); Pk0 = P[k0].clone()
            PAs.append((P - bm) @ U); LMs.append(((E @ U) ** 2).mean(1)); del P, E
        back = lambda A: bm + A @ U.T
        AqT = (Bq - bm) @ U
        PA, LM = PAs[0], LMs[0]                                             # ---- no prior
        ps = back(PA[int(LM.mean(1).argmin())]); inv_err.append(float((ps - Pk0).abs().max()))
        res["spec_shared"].append(ev_good(ps, Bq))
        c = per_band(LM); lam_log.append([LAMS[int(c[band == b][0])] for b in range(NB)])
        res["spec_perband"].append(ev_good(back(take(PA, c)), Bq))
        res["spec_permode"].append(ev_good(back(take(PA, LM.argmin(0))), Bq))
        cut = min((float(torch.where(band <= b, LM[k0], LM[-1]).sum()), b) for b in range(NB))[1]
        res["spec_trunc"].append(ev_good(back(take(PA, torch.where(band <= cut, k0, NL - 1))), Bq))
        res["spec_perband_ORACLE"].append(ev_good(back(take(PA, per_band(((PA - AqT) ** 2).sum(1)))), Bq))
        PA, LM = torch.cat(PAs), torch.cat(LMs)                             # ---- with prior: configs = gamma x lambda
        res["prior_global"].append(ev_good(back(PA[int(LM.mean(1).argmin())]), Bq))
        c = per_band(LM); gam_log.append([GAMS[int(c[band == b][0]) // NL] for b in range(NB)])
        res["prior_perband"].append(ev_good(back(take(PA, c)), Bq))
        sc = torch.stack([((PA[i] - AqT) ** 2).sum(0) for i in range(len(PA))])
        res["prior_perband_ORACLE"].append(ev_good(back(take(PA, per_band(sc))), Bq))
        del PAs, LMs, PA, LM; torch.cuda.empty_cache()
    a0, p0 = np.array(res["vox_shared"]), np.array(res["prior_global"])
    print(f"\n[M2/M4] n={n}, {REPS} draws; max|inverse(spec_shared) - vox_shared| = {max(inv_err):.1e}")
    for k in names:
        a = np.array(res[k]); d = a - (p0 if k.startswith("prior") else a0)
        print(f"  {k:22s} EV {a.mean():.4f}  d vs {'prior_global' if k.startswith('prior') else 'vox_shared':12s} {d.mean():+.4f} [min {d.min():+.4f}, max {d.max():+.4f}]")
    print("  lambda per band (median), band 0 = lowest freq:", [float(x) for x in np.median(np.array(lam_log), 0)])
    print("  gamma  per band (mean),   band 0 = lowest freq:", [round(float(x), 2) for x in np.mean(np.array(gam_log), 0)], flush=True)
print(f"done {time.time()-t0:.0f}s")
