"""E21 (docs/results.md): is there a shared representation across subjects?  Tests of the "shared latent
space" hypothesis (docs/propose/propose01.md, PRH mapping) on the 1,000 common NSD images of S1/S2/S5/S7
(repeat-averaged betas; no per-trial data is staged, so ceilings are voxel-split / ncsnr based).

A. RSA: Spearman correlation of image x image RDMs (correlation distance over z-scored voxels) between
   subjects, per ROI; ceiling = within-subject RDM correlation between two random halves of the voxels,
   Spearman-Brown corrected; comparison = RDM of CLIP / DINOv2 / SigLIP features.
B. Mutual k-NN alignment (Huh et al. 2024, k = 10) on the same spaces.
C. Voxel-level cross-subject prediction, 2-fold over the common images (500 fit / 500 test): for each
   target voxel of subject i (ncsnr > 0.2) predict its response from (a) MNI-nearest voxel of subject j,
   (b) best train-correlated voxel of subject j, (c) ridge on all ncsnr > 0.2 voxels of subject j,
   (d) ridge on CLIP-512, (e) ridge on subject j voxels + CLIP.  Per-voxel lambda by LOO on the fit half.
   EV on the test half, mean over ncsnr > 0.2 and per ROI, and normalised by NC = ncsnr^2 / (ncsnr^2 + 1/3).
    python scripts/propose01_shared_latent.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from src.eval import roi_masks  # noqa: E402

dev = torch.device("cuda"); D = Path("data/braincorl"); OUT = Path("outputs/propose01/shared_latent"); OUT.mkdir(parents=True, exist_ok=True)
SUBJ = [1, 2, 5, 7]; ROI = ["faces", "places", "bodies", "words", "ncsnr>0.2"]; K_NN = 10; SEEDS = 5
BACKBONE = {"clip": "clip_b16", "dinov2": "dinov2_base", "siglip": "siglip_so400m"}
torch.manual_seed(0); rng = np.random.default_rng(0)


def zscore(x):
    return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)


def rdm(x):
    """Correlation-distance RDM (n x n) of rows of x, returned as the upper triangle."""
    z = zscore(x.T).T                          # z-score each row across features
    c = z @ z.T / z.shape[1]
    iu = torch.triu_indices(c.shape[0], c.shape[0], 1, device=c.device)
    return (1 - c[iu[0], iu[1]]).cpu().numpy()


def knn_sets(x, k):
    z = zscore(x.T).T; c = z @ z.T; c.fill_diagonal_(-1e9)
    return c.topk(k, 1).indices


def mutual_knn(x, y, k=K_NN):
    a, b = knn_sets(x, k), knn_sets(y, k)
    return (a[:, :, None] == b[:, None, :]).any(2).float().sum(1).mean().item() / k


def ridge_loo_predict(Xtr, Ytr, Xte, lams, c0=100.0):
    """Dual-form ridge (n < p); per-target lambda chosen by closed-form LOO on the fit half.
    Xtr (n, p), Ytr (n, V) *uncentred*: the intercept is a constant feature of squared scale c0 (lightly
    penalised), so the LOO identity holds exactly.  Centring with the full fit set instead leaks the held-out
    point into the LOO fit and, for p >> n and small lambda, made LOO optimistic by up to 0.18 MSE (verified
    2026-09-13 by brute force).  Kernel scaled to unit mean diagonal so lambda is comparable across predictor
    sets; float64 for the LOO identity.  Returns predictions on Xte (m, V)."""
    Xtr, Xte, Ytr = Xtr.double(), Xte.double(), Ytr.double()
    K = Xtr @ Xtr.T; sc = K.diagonal().mean(); K = K / sc + c0; Kte = Xte @ Xtr.T / sc + c0
    e, Q = torch.linalg.eigh(K); QY = Q.T @ Ytr
    best = torch.full((Ytr.shape[1],), float("inf"), device=dev, dtype=torch.float64); pred = torch.zeros(Xte.shape[0], Ytr.shape[1], device=dev, dtype=torch.float64)
    for lam in lams:
        inv = 1.0 / (e + lam); dinv = ((Q ** 2) * inv).sum(1)       # diag of (K + lam I)^-1
        alpha = Q @ (inv[:, None] * QY)                             # (K + lam I)^-1 Y; residual y - K alpha = lam * alpha
        loo = ((alpha / dinv[:, None]) ** 2).mean(0)                # e_i / (1 - h_ii) with 1 - h_ii = lam * dinv_i
        upd = loo < best; best = torch.where(upd, loo, best); pred[:, upd] = (Kte @ alpha)[:, upd]
    return pred.float()


def ev(pred, true):
    return (1 - (true - pred).var(0) / true.var(0)).cpu().numpy()


# --------------------------------------------------------------------------- load
S = {}
for s in SUBJ:
    b = torch.from_numpy(np.load(D / "clip_b16" / f"s{s}_common_nrn_feats_msk.npy").astype(np.float32)).to(dev)
    meta = np.load(D / f"s{s}_voxel_meta.npz"); r = roi_masks(D / f"s{s}_voxel_meta.npz")
    S[s] = dict(B=zscore(b), raw=b, roi=r, ncsnr=meta["ncsnr"], xyz=torch.from_numpy(np.load(D / "clip_b16" / f"s{s}_coords.npy").astype(np.float32)).to(dev))
feat = {k: zscore(torch.from_numpy(np.load(D / v / "common_img_feats_normed.npy").astype(np.float32)).to(dev)) for k, v in BACKBONE.items()}
N = S[1]["B"].shape[0]

# --------------------------------------------------------------------------- A + B: RSA and mutual kNN
rows = []
for roi in ROI:
    spaces = {f"S{s}": S[s]["B"][:, S[s]["roi"][roi]] for s in SUBJ}; spaces.update(feat)
    R = {k: rdm(v) for k, v in spaces.items()}
    for a, b in itertools.combinations(spaces, 2):
        rows.append(dict(roi=roi, a=a, b=b, kind="pair", rsa=spearmanr(R[a], R[b])[0], mknn=mutual_knn(spaces[a], spaces[b])))
    for s in SUBJ:                                                  # voxel-split ceiling
        X = spaces[f"S{s}"]; rs, ms = [], []
        for seed in range(SEEDS):
            p = torch.from_numpy(np.random.default_rng(seed).permutation(X.shape[1])).to(dev); h = X.shape[1] // 2
            r_half = spearmanr(rdm(X[:, p[:h]]), rdm(X[:, p[h:]]))[0]; rs.append(2 * r_half / (1 + r_half))
            ms.append(mutual_knn(X[:, p[:h]], X[:, p[h:]]))
        rows.append(dict(roi=roi, a=f"S{s}", b=f"S{s}", kind="ceiling", rsa=np.mean(rs), mknn=np.mean(ms)))
    perm = torch.from_numpy(rng.permutation(N)).to(dev)             # null: image labels shuffled in one subject
    rows.append(dict(roi=roi, a="S1", b="S2perm", kind="null", rsa=spearmanr(R["S1"], rdm(spaces["S2"][perm]))[0],
                     mknn=mutual_knn(spaces["S1"], spaces["S2"][perm])))
    print(f"[A/B] {roi}: n_vox " + ", ".join(f"S{s}={int(S[s]['roi'][roi].sum())}" for s in SUBJ), flush=True)
ab = pd.DataFrame(rows); ab.to_csv(OUT / "rsa_mknn.csv", index=False)

# --------------------------------------------------------------------------- C: voxel-level cross-subject prediction
# All standardisation uses fit-fold statistics only; EV is computed per test fold and averaged over the 2 folds.
lams = [float(x) for x in np.logspace(-2, 3, 11)]; folds = [torch.from_numpy(np.arange(N) % 2 == f).to(dev) for f in (0, 1)]


def std_by(tr, x):
    m, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    return (x - m) / sd


def nn_predict(xtr, ytr, xte):
    """Per-target 1-d OLS with intercept on the fit fold."""
    xm, ym = xtr.mean(0), ytr.mean(0); xc = xtr - xm
    slope = (xc * (ytr - ym)).sum(0) / ((xc ** 2).sum(0) + 1e-6)
    return ym + slope * (xte - xm)


rows = []; pervox = {}
for i in SUBJ:
    good = S[i]["roi"]["ncsnr>0.2"]; Yraw = S[i]["raw"][:, good]; nc = S[i]["ncsnr"][good]; NC = nc ** 2 / (nc ** 2 + 1 / 3)
    rois = {r: S[i]["roi"][r][good] for r in ROI}
    Fraw = {k: S[k]["raw"][:, S[k]["roi"]["ncsnr>0.2"]] for k in SUBJ if k != i}; Fraw.update(feat)
    for j in SUBJ:
        if j == i:
            continue
        gj = S[j]["roi"]["ncsnr>0.2"]
        d = torch.cdist(S[i]["xyz"][good], S[j]["xyz"][gj]); anat = d.argmin(1); dmin = d.min(1).values.cpu().numpy()
        evs = {k: [] for k in ["anat_nn", "func_nn", "ridge_subj", "ridge_clip", "ridge_subj+clip"]}
        for te in folds:
            tr = ~te; Y = std_by(tr, Yraw); X = std_by(tr, Fraw[j]); C = std_by(tr, feat["clip"])
            Ytr, Yte, Xtr, Xte, Ctr, Cte = Y[tr], Y[te], X[tr], X[te], C[tr], C[te]
            evs["anat_nn"].append(ev(nn_predict(Xtr[:, anat], Ytr, Xte[:, anat]), Yte))
            fn = (Ytr.T @ Xtr).argmax(1)                                # max train correlation (both standardised)
            evs["func_nn"].append(ev(nn_predict(Xtr[:, fn], Ytr, Xte[:, fn]), Yte))
            evs["ridge_subj"].append(ev(ridge_loo_predict(Xtr, Ytr, Xte, lams), Yte))
            evs["ridge_clip"].append(ev(ridge_loo_predict(Ctr, Ytr, Cte, lams), Yte))
            sx = torch.sqrt((Xtr ** 2).sum(1).mean()); sc = torch.sqrt((Ctr ** 2).sum(1).mean())
            evs["ridge_subj+clip"].append(ev(ridge_loo_predict(torch.cat([Xtr / sx, Ctr / sc], 1), Ytr, torch.cat([Xte / sx, Cte / sc], 1), lams), Yte))
        for k, e2 in evs.items():
            e = np.mean(e2, 0); pervox[f"S{i}_from_S{j}_{k}"] = e
            for r, m in rois.items():
                rows.append(dict(target=f"S{i}", source=f"S{j}", method=k, roi=r, n_vox=int(m.sum()), ev=float(e[m].mean()),
                                 ev_over_nc=float((e[m] / NC[m]).mean()), mni_dist_mm=float(dmin[m].mean())))
        print(f"[C] S{i} <- S{j}: " + ", ".join(f"{k}={float(np.mean(e2, 0).mean()):.4f}" for k, e2 in evs.items()), flush=True)
        torch.cuda.empty_cache()
    # backbone-only rows once per target (identical for every source; source = "-")
    for name in BACKBONE:
        if name == "clip":
            continue
        e2 = []
        for te in folds:
            tr = ~te; Y = std_by(tr, Yraw); F = std_by(tr, feat[name])
            e2.append(ev(ridge_loo_predict(F[tr], Y[tr], F[te], lams), Y[te]))
        e = np.mean(e2, 0); pervox[f"S{i}_ridge_{name}"] = e
        for r, m in rois.items():
            rows.append(dict(target=f"S{i}", source="-", method=f"ridge_{name}", roi=r, n_vox=int(m.sum()), ev=float(e[m].mean()),
                             ev_over_nc=float((e[m] / NC[m]).mean()), mni_dist_mm=np.nan))
c = pd.DataFrame(rows); c.to_csv(OUT / "voxel_prediction.csv", index=False); np.savez(OUT / "voxel_prediction_pervoxel.npz", **pervox)
print(c[c.roi == "ncsnr>0.2"].groupby("method")[["ev", "ev_over_nc"]].mean().round(4))
print("DONE")
