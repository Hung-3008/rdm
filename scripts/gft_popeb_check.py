"""PopEB-v2 + graph Fourier bands on S1 (2026-09-18, negative result): can a per-band choice improve PopEB-v2?
(a) residual: per band, PopEB coefficients m unchanged or m + ridge_lambda(B - F m); LOO with the PopEB hat diagonal
    (Cbar and sig2 taken as fixed), the first version (in-sample error for the unchanged candidate), and the oracle
    (choice on the test set = upper bound).
(b) kappa: per band, PopEB posterior with prior variance kappa * tau2; LOO and oracle.
Needs outputs/gft/s1_V15724.npz from scripts/gft_band_ridge.py.

    python scripts/gft_popeb_check.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np, torch
from pathlib import Path
from src.config import load_config
from src.dataset import SOURCES
from src.model.lowrank_basis import *
from src.utils import explained_variance, unit_norm
torch.set_grad_enabled(False); dev = torch.device("cuda"); NG = Path("data/braincorl_nsdgeneral"); D = NG / "clip_b16"
cfg = load_config("src/config/popeb_clip.yaml", [f"data.source.data_dir={D}", "data.source.train_subjects=[2,5,7]", "data.source.val_subjects=[2,5,7]"])
fc = LowRankConfig(**cfg["fast"]); basis = FunctionalBasis.from_file("outputs/repro/clip_b16/popeb_s1/K32/basis.pt", dev)
bundle = SOURCES.build(cfg["data"]["source"]).load_eval(1)
pop, pxyz, imgs = [], [], []
for s in [2, 5, 7]:
    img = unit_norm(np.load(D / f"s{s}_unique_img_feats_normed.npy").astype(np.float32)); imgs.append(img)
    g = np.load(NG / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2
    pop.append(full_coefficients(basis, img, np.load(D / f"s{s}_unique_nrn_feats_msk.npy"), dev)[:, g]); pxyz.append(np.load(D / f"s{s}_coords.npy").astype(np.float32)[g])
Cpop = torch.from_numpy(np.concatenate(pop, 1)).to(dev); tau2 = float(fc.tau2_ratio * Cpop.var(1).mean())
xyz = torch.from_numpy(np.load(D / "s1_coords.npy").astype(np.float32)).to(dev)
spatial = (xyz, torch.from_numpy(np.concatenate(pxyz)).to(dev), fc.spatial_sigma); Fm, Fs = (t.clone() for t in feature_stats(basis, imgs, dev))
U = torch.from_numpy(np.load("outputs/gft/s1_V15724.npz")["U"]).to(dev); V = U.shape[0]
edges = np.r_[0, np.unique(np.round(np.logspace(np.log10(8), np.log10(V), 12)).astype(int))]
band = torch.from_numpy(np.searchsorted(edges, np.arange(V), side="right") - 1).to(dev); NB = int(band.max()) + 1
good = torch.from_numpy(np.load(NG / "s1_voxel_meta.npz")["ncsnr"] > 0.2).to(dev)
true = torch.from_numpy(bundle.query_beta).to(dev); Fq = (basis(torch.from_numpy(bundle.query_img).to(dev)) - Fm) / Fs
LAMS = list(fc.lambdas); ev = lambda P: float(explained_variance(P.T, true.T)[good].mean())
def pick(score, P):
    out = torch.empty_like(P[0])
    for b in range(NB):
        m = band == b; out[:, m] = P[int(score[:, m].sum(1).argmin())][:, m]
    return out
print("(a) residual ridge per band")
start = 0
for n, reps in [(100, 5), (30, 3), (300, 3), (1000, 3)]:
  acc = {k: [] for k in ["popeb", "gft_insample(v1)", "gft_loo_hat", "gft_ORACLE", "global_loo_hat"]}; keep = []
  for r in range(reps):
      idx = bundle.icl_order[start:start + n]; start += n
      F = (basis(torch.from_numpy(bundle.ic_img[idx]).to(dev)) - Fm) / Fs; B = torch.from_numpy(bundle.ic_beta[idx]).to(dev); bm = B.mean(0, keepdim=True); Bc = B - bm
      m = popeb(F, Bc, Cpop, tau2, fc.lam0, spatial=spatial)
      e, Q = torch.linalg.eigh(F.T @ F); FQ, FqQ = F @ Q, Fq @ Q
      Chat = Q @ ((1 / (e + fc.lam0))[:, None] * (FQ.T @ Bc)); h0 = ((FQ ** 2) / (e + fc.lam0)).sum(1)
      s2 = (((Bc - F @ Chat) / (1 - h0)[:, None]) ** 2).mean(0)                                   # (V,)
      g = (e[:, None] / s2[None]) / (e[:, None] / s2[None] + 1 / tau2) / (e[:, None] + fc.lam0)   # (K, V) PopEB filter
      R = Bc - F @ m; base = Fq @ m; QtR = FQ.T @ R
      P = [base @ U]; Lin = [((R @ U) ** 2).mean(0)]; Lhat = [(((R / (1 - (FQ ** 2) @ g).clamp_min(0.05)) @ U) ** 2).mean(0)]
      for lam in LAMS:
          inv = 1 / (e + lam); h = ((FQ ** 2) * inv).sum(1)
          if (1 - h).min() < 0.05: continue
          Cd = inv[:, None] * QtR; res = R - FQ @ Cd
          P.append((base + FqQ @ Cd) @ U); Lin.append((((res / (1 - h)[:, None]) @ U) ** 2).mean(0))
          ht = (FQ ** 2) @ (g + inv[:, None] - (e * inv)[:, None] * g)
          Lhat.append((((res / (1 - ht).clamp_min(0.05)) @ U) ** 2).mean(0))
      Lin, Lhat = torch.stack(Lin), torch.stack(Lhat); AqT = (true - bm) @ U
      Lor = torch.stack([((p - AqT) ** 2).mean(0) for p in P])
      acc["popeb"].append(ev(bm + base)); acc["gft_insample(v1)"].append(ev(bm + pick(Lin, P) @ U.T))
      acc["gft_loo_hat"].append(ev(bm + pick(Lhat, P) @ U.T)); acc["gft_ORACLE"].append(ev(bm + pick(Lor, P) @ U.T))
      acc["global_loo_hat"].append(ev(bm + P[int(Lhat.sum(1).argmin())] @ U.T))
      keep.append([int(Lhat[:, band == b].sum(1).argmin()) for b in range(NB)])
  p0 = np.array(acc["popeb"])
  print(f"n={n}: " + "  ".join(f"{k} {np.mean(v):.4f} ({np.mean(np.array(v)-p0):+.4f})" for k, v in acc.items()))
  print("   candidate index per band, draw 0 (0 = PopEB unchanged; k>0 = + residual ridge with k-th usable lambda):", keep[0], flush=True)

print("(b) prior variance kappa * tau2 per band")
KAP = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
start = 0
for n, reps in [(100, 5), (30, 3), (300, 3), (1000, 3)]:
    acc = {k: [] for k in ["popeb", "kappa_global_loo", "kappa_band_loo", "kappa_band_ORACLE", "kappa_global_ORACLE"]}; keep = []
    for r in range(reps):
        idx = bundle.icl_order[start:start + n]; start += n
        F = (basis(torch.from_numpy(bundle.ic_img[idx]).to(dev)) - Fm) / Fs; B = torch.from_numpy(bundle.ic_beta[idx]).to(dev); bm = B.mean(0, keepdim=True); Bc = B - bm
        m = popeb(F, Bc, Cpop, tau2, fc.lam0, spatial=spatial)
        e, Q = torch.linalg.eigh(F.T @ F); FQ, FqQ = F @ Q, Fq @ Q
        ch = (FQ.T @ Bc) / (e + fc.lam0)[:, None]; h0 = ((FQ ** 2) / (e + fc.lam0)).sum(1)      # C_hat in Q coordinates
        s2 = (((Bc - FQ @ ch) / (1 - h0)[:, None]) ** 2).mean(0); es = e[:, None] / s2[None]
        cb = tau2 * ((es + 1 / tau2) * (Q.T @ m) - es * ch)                                 # Cbar recovered from m, Q coordinates
        P, L = [], []
        for k in KAP:
            a = es / (es + 1 / (k * tau2)); mk = a * ch + (1 - a) * cb                          # posterior mean, prior variance k*tau2
            P.append((FqQ @ mk) @ U)
            L.append((((Bc - FQ @ mk) / (1 - (FQ ** 2) @ (a / (e + fc.lam0)[:, None])).clamp_min(0.05)) @ U).pow(2).mean(0))
        L = torch.stack(L); AqT = (true - bm) @ U; Lor = torch.stack([((p - AqT) ** 2).mean(0) for p in P])
        acc["popeb"].append(ev(bm + Fq @ m)); acc["kappa_global_loo"].append(ev(bm + P[int(L.sum(1).argmin())] @ U.T))
        acc["kappa_band_loo"].append(ev(bm + pick(L, P) @ U.T)); acc["kappa_band_ORACLE"].append(ev(bm + pick(Lor, P) @ U.T))
        acc["kappa_global_ORACLE"].append(ev(bm + P[int(Lor.sum(1).argmin())] @ U.T))
        keep.append([KAP[int(L[:, band == b].sum(1).argmin())] for b in range(NB)])
    p0 = np.array(acc["popeb"])
    print(f"n={n}: " + "  ".join(f"{k} {np.mean(v):.4f} ({np.mean(np.array(v)-p0):+.4f})" for k, v in acc.items()))
    print("   kappa per band by LOO, draw 0 (band 0 = lowest freq):", keep[0], flush=True)
