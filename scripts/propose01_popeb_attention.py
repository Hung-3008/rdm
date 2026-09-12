"""E11 (docs/results.md): where does PopEB-v2 put its population weight?  For every hold-out voxel v the
softmax weights w_vj over the population (33,506 voxels of the training subjects) are computed exactly as in
`popeb` (K32 config, n = 100 support, draw 0), and summarised by floc ROI: the share of weight that lands on
population voxels of the same ROI, the effective number of population voxels (exp of the entropy), and the
mean MNI distance to the attended voxels. Also the same with spatial weighting switched off (sigma = 0) to
separate the functional (coefficient-space) and the anatomical part of the attention.
    python scripts/propose01_popeb_attention.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.config import load_config  # noqa: E402
from src.dataset import SOURCES  # noqa: E402
from src.eval import roi_masks  # noqa: E402
from src.model.lowrank_basis import FunctionalBasis, LowRankConfig, feature_stats, full_coefficients  # noqa: E402
from src.utils import unit_norm  # noqa: E402

dev = torch.device("cuda"); D = Path("data/braincorl/clip_b16"); OUT = Path("outputs/propose01/attention"); OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config("src/config/popeb_clip.yaml", []); f = LowRankConfig(**cfg["fast"]); TR = cfg["data"]["source"]["train_subjects"]
basis = FunctionalBasis.from_file("outputs/propose01/sweep/K32_a1/basis.pt", dev)
ROI = ["faces", "places", "bodies", "words"]


def roi_id(meta_path):
    r = roi_masks(meta_path); ids = np.zeros(len(r["ncsnr>0.2"]), np.int64)
    for j, c in enumerate(ROI, 1):
        ids[r[c]] = j
    return ids, r

pop, xyz, pid, imgs = [], [], [], []
for s in TR:
    img = unit_norm(np.load(D / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
    Cs = full_coefficients(basis, img, np.load(D / f"s{s}_unique_nrn_feats_msk.npy"), dev); torch.cuda.empty_cache()
    ids, r = roi_id(f"data/braincorl/s{s}_voxel_meta.npz"); g = r["ncsnr>0.2"]
    pop.append(Cs[:, g]); xyz.append(np.load(D / f"s{s}_coords.npy").astype(np.float32)[g]); pid.append(ids[g]); imgs.append(img)
Cpop = torch.from_numpy(np.concatenate(pop, 1)).to(dev); xyz_p = torch.from_numpy(np.concatenate(xyz, 0)).to(dev); pid = np.concatenate(pid)
f.tau2 = float(f.tau2_ratio * Cpop.var(1).mean().item()); Fm, Fsd = feature_stats(basis, imgs, dev)

bundle = SOURCES.build(cfg["data"]["source"]).load_eval(1)
vid, r1 = roi_id("data/braincorl/s1_voxel_meta.npz"); good = r1["ncsnr>0.2"]
xyz_v = torch.from_numpy(np.load(D / "s1_coords.npy").astype(np.float32)).to(dev)
idx = bundle.icl_order[:100]
with torch.no_grad():
    B = torch.from_numpy(bundle.ic_beta[idx]).to(dev); Fs = basis(torch.from_numpy(bundle.ic_img[idx]).to(dev))
    Fn = (Fs - Fm) / Fsd; Bc = B - B.mean(0, keepdim=True)
    n, K = Fn.shape; G = Fn.T @ Fn; e, Q = torch.linalg.eigh(G); FQ = Fn @ Q
    inv = 1.0 / (e + f.lam0); h = ((FQ ** 2) * inv).sum(1)
    Chat = Q @ (inv[:, None] * (FQ.T @ Bc)); sig2 = (((Bc - Fn @ Chat) / (1 - h)[:, None]) ** 2).mean(0)
    I = torch.eye(K, device=dev); CC = (Cpop[:, None, :] * Cpop[None, :, :]).reshape(K * K, -1); pn = (xyz_p ** 2).sum(1)
    pid_t = torch.from_numpy(pid).to(dev)
    rows = []
    for sigma in [f.spatial_sigma, 0.0]:
        same = np.zeros(Chat.shape[1]); neff = np.zeros(Chat.shape[1]); dist = np.zeros(Chat.shape[1]); top1 = np.zeros(Chat.shape[1])
        share = np.zeros((Chat.shape[1], 5))
        for a in range(0, Chat.shape[1], 512):
            s2 = sig2[a:a + 512]; c = Chat[:, a:a + 512].T
            M = torch.linalg.solve(s2[:, None, None] * I[None] + (f.tau2 * G)[None], G[None].expand(len(s2), K, K))
            cM = torch.einsum("vi,vij->vj", c, M)
            logits = -0.5 * ((cM * c).sum(1, keepdim=True) - 2 * cM @ Cpop + M.reshape(len(s2), -1) @ CC)
            d2 = (xyz_v[a:a + 512] ** 2).sum(1, keepdim=True) - 2 * xyz_v[a:a + 512] @ xyz_p.T + pn[None, :]
            if sigma > 0:
                logits = logits - 0.5 * d2 / sigma ** 2
            w = torch.softmax(logits, 1)
            for j in range(5):
                share[a:a + 512, j] = w[:, pid_t == j].sum(1).cpu().numpy()
            neff[a:a + 512] = torch.exp(-(w * torch.log(w.clamp_min(1e-30))).sum(1)).cpu().numpy()
            dist[a:a + 512] = (w * d2.clamp_min(0).sqrt()).sum(1).cpu().numpy()
            top1[a:a + 512] = w.max(1).values.cpu().numpy()
        for j, c in enumerate(ROI, 1):
            m = good & (vid == j)
            rows.append(dict(sigma=sigma, roi=c, n_vox=int(m.sum()), share_same_roi=share[m, j].mean(),
                             share_other_floc=(share[m, 1:].sum(1) - share[m, j]).mean(), share_non_floc=share[m, 0].mean(),
                             pop_frac_same_roi=float((pid == j).mean()), n_eff=np.median(neff[m]), top1=np.median(top1[m]), dist_mm=dist[m].mean()))
        m = good & (vid == 0)
        rows.append(dict(sigma=sigma, roi="non-floc", n_vox=int(m.sum()), share_same_roi=share[m, 0].mean(), share_other_floc=share[m, 1:].sum(1).mean(),
                         share_non_floc=share[m, 0].mean(), pop_frac_same_roi=float((pid == 0).mean()), n_eff=np.median(neff[m]), top1=np.median(top1[m]), dist_mm=dist[m].mean()))
df = pd.DataFrame(rows); pd.set_option("display.width", 250)
print("## E11 population attention of S1 voxels (n = 100, draw 0). share_same_roi = weight on population voxels of the same floc ROI;")
print("## pop_frac_same_roi = chance level (fraction of the population in that ROI); n_eff = exp(entropy) (median); dist_mm = weighted mean MNI distance")
print(df.round(3).to_string(index=False))
df.to_csv(OUT / "attention_by_roi.csv", index=False)
