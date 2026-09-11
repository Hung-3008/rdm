"""Anatomical prior-mean ridge on top of the low-rank basis (propose01 variant A).
Tests (a) shrink factor gamma toward C0 chosen by LOO (global / per-voxel / per-ROI),
(c) Gaussian distance weighting instead of ball mean, (d) nested hold-outs S2/S5/S7.
Optional (b): --surf_map <npz> with per-subject arrays 'vox2fsavg_s{i}' giving, for each fsaverage
vertex, the func voxel index of subject i (or -1); the prior then averages over vertex correspondences.

    python prior_test.py --holdout 1 --train 2 5 7 --basis outputs/propose01/sweep/K16_a1/basis.pt
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
from src.model.lowrank_basis import FunctionalBasis, LowRankConfig  # noqa
from src.dataset import SOURCES  # noqa
from src.eval import roi_masks  # noqa
from src.utils import explained_variance  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", type=int, required=True)
ap.add_argument("--train", type=int, nargs="+", required=True)
ap.add_argument("--basis", required=True)
ap.add_argument("--num_ic", type=int, default=100)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--surf_map", default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()

D = Path("data/braincorl/clip_b16")
dev = torch.device("cuda")
basis = FunctionalBasis.from_file(args.basis, dev)
LAMS = LowRankConfig().lambdas
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
t0 = time.time()


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@torch.no_grad()
def feats(img):
    return basis(torch.from_numpy(img).to(dev))


def full_C(s):
    """Ridge on all unique images of subject s in the shared basis -> (K, V) on CPU."""
    img = unit(np.load(D / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
    B = torch.from_numpy(np.load(D / f"s{s}_unique_nrn_feats_msk.npy").astype(np.float32)).to(dev)
    F = feats(img)
    Fn = (F - F.mean(0, keepdim=True)) / F.std(0, keepdim=True).clamp_min(1e-6)
    C = torch.linalg.solve(Fn.T @ Fn + torch.eye(Fn.shape[1], device=dev), Fn.T @ (B - B.mean(0, keepdim=True)))
    del B, F, Fn; torch.cuda.empty_cache()
    return C.cpu().numpy()


h = args.holdout
coords = {s: np.load(D / f"s{s}_coords.npy").astype(np.float32) for s in [h] + args.train}
Ct = {s: full_C(s) for s in args.train}
V = coords[h].shape[0]
print(f"holdout S{h}, train {args.train}, C_full done {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ priors
def prior_ball(r):
    C0 = np.zeros((Ct[args.train[0]].shape[0], V), np.float32)
    for s in args.train:
        tree = cKDTree(coords[s]); nb = tree.query_ball_point(coords[h], r); _, nn = tree.query(coords[h])
        for v, lst in enumerate(nb):
            C0[:, v] += Ct[s][:, lst if lst else [nn[v]]].mean(1)
    return C0 / len(args.train)


def prior_gauss(sigma, rmax=None):
    rmax = rmax or 3 * sigma
    C0 = np.zeros((Ct[args.train[0]].shape[0], V), np.float32)
    for s in args.train:
        tree = cKDTree(coords[s]); nb = tree.query_ball_point(coords[h], rmax); _, nn = tree.query(coords[h])
        for v, lst in enumerate(nb):
            if not lst:
                C0[:, v] += Ct[s][:, nn[v]]; continue
            d = np.linalg.norm(coords[s][lst] - coords[h][v], axis=1)
            w = np.exp(-0.5 * (d / sigma) ** 2); w /= w.sum()
            C0[:, v] += Ct[s][:, lst] @ w
    return C0 / len(args.train)


def prior_surface(path, fallback):
    """Average C over training-subject voxels that share an fsaverage vertex with the hold-out voxel."""
    m = np.load(path)
    vh = m[f"vox2fsavg_s{h}"]                       # (n_vertices,) voxel index of hold-out or -1
    C0 = np.zeros_like(fallback); cnt = np.zeros(V)
    for s in args.train:
        vs = m[f"vox2fsavg_s{s}"]
        ok = (vh >= 0) & (vs >= 0)
        np.add.at(C0.T, vh[ok], Ct[s][:, vs[ok]].T); np.add.at(cnt, vh[ok], 1)
    covered = cnt > 0
    C0[:, covered] /= cnt[covered]; C0[:, ~covered] = fallback[:, ~covered]
    print(f"surface prior: {covered.mean()*100:.1f}% of hold-out voxels covered by >=1 vertex correspondence")
    return C0


priors = {"ball3.2": prior_ball(3.2), "gauss2": prior_gauss(2.0), "gauss3": prior_gauss(3.0), "gauss5": prior_gauss(5.0),
          "gauss8": prior_gauss(8.0, rmax=12.0)}
if args.surf_map:
    priors["surface"] = prior_surface(args.surf_map, priors["gauss5"])
    # surface prior smoothed over volume neighbours (sigma 3 mm) to reduce vertex-sampling noise
    tree = cKDTree(coords[h]); nb = tree.query_ball_point(coords[h], 6.0); Cs = np.zeros_like(priors["surface"])
    for v, lst in enumerate(nb):
        d = np.linalg.norm(coords[h][lst] - coords[h][v], axis=1); w = np.exp(-0.5 * (d / 3.0) ** 2); w /= w.sum()
        Cs[:, v] = priors["surface"][:, lst] @ w
    priors["surface_sm3"] = Cs
priors = {k: torch.from_numpy(v).to(dev) for k, v in priors.items()}
print(f"priors done {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ hold-out data
bundle = SOURCES.build(dict(name="npy_multisubj", data_dir=str(D), train_subjects=args.train, val_subjects=args.train[:1])).load_eval(h)
F_all = feats(bundle.ic_img).cpu(); Fq = feats(bundle.query_img)
true = torch.from_numpy(bundle.query_beta.T).to(dev)
rois = roi_masks(f"data/braincorl/s{h}_voxel_meta.npz"); good = rois["ncsnr>0.2"]
roi_ids = np.zeros(V, np.int64)                                  # 0 = other, 1..4 = faces/places/bodies/words
for j, c in enumerate(["faces", "places", "bodies", "words"], 1):
    roi_ids[rois[c]] = j
roi_ids_t = torch.from_numpy(roi_ids).to(dev)


def loo_grid(Fn, Bc, C0):
    """For every (lambda, gamma): C (K,V) and per-voxel LOO MSE. Returns dict keyed by (lam, gam)."""
    n, K = Fn.shape
    evals, Q = torch.linalg.eigh(Fn.T @ Fn); FQ = Fn @ Q
    out = {}
    for lam in LAMS:
        inv = 1.0 / (evals + lam); hd = ((FQ ** 2) * inv).sum(1)
        if (1.0 - hd).min() < 0.05:
            continue
        for g in GAMMAS:
            R = Bc - g * (Fn @ C0)
            Cd = Q @ (inv[:, None] * (FQ.T @ R))
            loo = (((R - Fn @ Cd) / (1.0 - hd)[:, None]) ** 2).mean(0)       # (V,)
            out[(lam, g)] = (g * C0 + Cd, loo)
    return out


def select(grid, mode):
    """mode: 'global' one (lam,gam) for all voxels; 'voxel' per voxel; 'roi' per ROI id (floc labels)."""
    keys = list(grid); L = torch.stack([grid[k][1] for k in keys])            # (n_cfg, V)
    if mode == "global":
        best = int(L.mean(1).argmin()); return grid[keys[best]][0], {keys[best]: 1.0}
    if mode == "voxel":
        idx = L.argmin(0)
    else:
        idx = torch.zeros(V, dtype=torch.long, device=dev)
        for r in range(5):
            m = roi_ids_t == r
            if m.any():
                idx[m] = int(L[:, m].mean(1).argmin())
    C = torch.zeros_like(grid[keys[0]][0]); usage = {}
    for j, k in enumerate(keys):
        m = idx == j
        if m.any():
            C[:, m] = grid[k][0][:, m]; usage[k] = float(m[torch.from_numpy(good).to(dev)].float().mean())
    return C, usage


variants = ["A"] + [f"{p}:g1" for p in priors] + [f"{p}:{m}" for p in priors for m in ["global", "voxel", "roi"]]
res = {k: [] for k in variants}; usage_log = {k: [] for k in variants}
start = 0
for rep in range(args.repeats):
    idx = bundle.icl_order[start:start + args.num_ic]; start += args.num_ic
    Fs = F_all[idx].to(dev); B = torch.from_numpy(np.asarray(bundle.ic_beta[idx], dtype=np.float32)).to(dev)
    Bm = B.mean(0, keepdim=True); Fm, Fsd = Fs.mean(0, keepdim=True), Fs.std(0, keepdim=True).clamp_min(1e-6)
    Fn = (Fs - Fm) / Fsd; Fqn = (Fq - Fm) / Fsd; Bc = B - Bm

    def ev_of(C):
        return explained_variance((Bm + Fqn @ C).T, true).cpu().numpy()

    gridA = loo_grid(Fn, Bc, torch.zeros_like(priors["ball3.2"]))
    CA, uA = select({k: v for k, v in gridA.items() if k[1] == 0.0}, "global")
    res["A"].append(ev_of(CA)); usage_log["A"].append(uA)
    for p, C0 in priors.items():
        grid = loo_grid(Fn, Bc, C0)
        C, u = select({k: v for k, v in grid.items() if k[1] == 1.0}, "global")
        res[f"{p}:g1"].append(ev_of(C)); usage_log[f"{p}:g1"].append(u)
        for m in ["global", "voxel", "roi"]:
            C, u = select(grid, m)
            res[f"{p}:{m}"].append(ev_of(C)); usage_log[f"{p}:{m}"].append(u)
    print(f"rep {rep} done {time.time()-t0:.0f}s  A={res['A'][-1][good].mean():.4f}  ball3.2:g1={res['ball3.2:g1'][-1][good].mean():.4f}  ball3.2:global={res['ball3.2:global'][-1][good].mean():.4f}")

# ------------------------------------------------------------------ report
print(f"\n=== S{h} hold-out, {args.num_ic} support x {args.repeats} draws, mean EV ===")
print(f"{'variant':18s} " + " ".join(f"{k:>10s}" for k in rois) + "   d(ncsnr>.2) vs A [95% CI]")
evA = np.stack(res["A"])
rng = np.random.default_rng(0)
rows = []
for k in variants:
    evs = np.stack(res[k]); d = (evs - evA)[:, good].mean(0)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)])
    ci = "" if k == "A" else f"{d.mean():+.4f} [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]"
    print(f"{k:18s} " + " ".join(f"{evs[:, m].mean():10.4f}" for m in rois.values()) + "   " + ci)
    rows.append((k, *[evs[:, m].mean() for m in rois.values()], d.mean()))
print("\nselected (lambda, gamma) usage on ncsnr>0.2 voxels, averaged over draws (global / roi modes):")
for k in variants:
    if k.endswith(":global") or k.endswith(":roi") or k == "A":
        agg = {}
        for u in usage_log[k]:
            for kk, f in u.items():
                agg[kk] = agg.get(kk, 0) + f / args.repeats
        print(f"  {k:18s} " + ", ".join(f"lam={kk[0]:g},g={kk[1]:g}:{f:.2f}" for kk, f in sorted(agg.items(), key=lambda x: -x[1])[:4]))
if args.out:
    np.savez(args.out, **{k: np.stack(v) for k, v in res.items()}, good=good)
print(f"done {time.time()-t0:.0f}s")
