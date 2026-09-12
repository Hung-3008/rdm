"""E10 / E12 / E15 (docs/results.md): on hold-out S1 with the K32 PopEB-v2 config and the K16 A config.
  E10  very small support (n = 10, 20; 5 draws each, drawn from the tail of the support ordering so they
       do not overlap the 100 x 5 draws): Ridge, A, PopEB-v2 and BraInCoRL (context below its training
       range of 30).
  E12  adaptation cost: wall-clock of the PopEB-v2 fast stage (all 22,612 voxels) and of one BraInCoRL
       forward over all voxels, same support (n = 100), same GPU; plus peak GPU memory.
  E15  noise-ceiling-normalised EV: EV / NC with NC = ncsnr^2 / (ncsnr^2 + 1/3) (3 repeats of the common
       images), by ncsnr bin, for PopEB-v2 and BraInCoRL at n = 100 (E07 / E00 outputs).
    python scripts/propose01_smalln_cost.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.config import load_config  # noqa: E402
from src.dataset import SOURCES  # noqa: E402
from src.eval import predict_icl, predict_ridge, roi_masks  # noqa: E402
from src.model import build_model  # noqa: E402
from src.model.checkpoint import load_original_checkpoint  # noqa: E402
from src.model.lowrank_basis import FunctionalBasis, LowRankConfig, feature_stats, full_coefficients, predict_lowrank  # noqa: E402
from src.utils import explained_variance, unit_norm  # noqa: E402

dev = torch.device("cuda")
OUT = Path("outputs/propose01/smalln_cost"); OUT.mkdir(parents=True, exist_ok=True)
D = Path("data/braincorl/clip_b16")
cfg2 = load_config("src/config/popeb_clip.yaml", []); cfg16 = load_config("src/config/lowrank_clip.yaml", [])
bundle = SOURCES.build(cfg2["data"]["source"]).load_eval(1)
rois = roi_masks("data/braincorl/s1_voxel_meta.npz"); good = rois["ncsnr>0.2"]; ncsnr = np.load("data/braincorl/s1_voxel_meta.npz")["ncsnr"]
true = torch.from_numpy(bundle.query_beta.T).to(dev); q_img = torch.from_numpy(bundle.query_img).to(dev)
TR = cfg2["data"]["source"]["train_subjects"]


def popeb_setup(ckpt, cfg):
    basis = FunctionalBasis.from_file(Path(ckpt) / "basis.pt", dev)
    f = LowRankConfig(**cfg["fast"])
    pop, xyz, imgs = [], [], []
    for s in TR:
        img = unit_norm(np.load(D / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
        Cs = full_coefficients(basis, img, np.load(D / f"s{s}_unique_nrn_feats_msk.npy"), dev); torch.cuda.empty_cache()
        g = np.load(f"data/braincorl/s{s}_voxel_meta.npz")["ncsnr"] > 0.2
        pop.append(Cs[:, g]); xyz.append(np.load(D / f"s{s}_coords.npy").astype(np.float32)[g]); imgs.append(img)
    Cpop = torch.from_numpy(np.concatenate(pop, 1)).to(dev)
    if f.tau2_ratio > 0:
        f.tau2 = float(f.tau2_ratio * Cpop.var(1).mean().item())
    spatial = (torch.from_numpy(np.load(D / "s1_coords.npy").astype(np.float32)).to(dev),
               torch.from_numpy(np.concatenate(xyz, 0)).to(dev), f.spatial_sigma) if f.spatial_sigma else None
    F_stats = feature_stats(basis, imgs, dev) if f.std_global else None
    return basis, f, Cpop, spatial, F_stats


basis32, f32, Cpop32, spatial32, Fst32 = popeb_setup("outputs/propose01/sweep/K32_a1", cfg2)
basis16, f16, _, _, _ = popeb_setup("outputs/propose01/sweep/K16_a1", cfg16)
model = build_model(load_config("src/config/braincorl_clip_3090.yaml", [])["module"]["model"])
load_original_checkpoint(model, "references/BraInCoRL/checkpoints/CLIP_trained_on_s257.pth"); model = model.to(dev).eval()

# ------------------------------------------------------------------ E10
res = {}
order = bundle.icl_order
start = 9000 - 5 * 30            # tail of the ordering: disjoint from the 100 x 5 and scaling draws (front 3,490 indices)
for n in [10, 20]:
    for rep in range(5):
        idx = order[start:start + n]; start += n
        ic_img = torch.from_numpy(bundle.ic_img[idx]).to(dev); ic_beta = torch.from_numpy(bundle.ic_beta[idx]).to(dev)
        preds = {
            "ridge": predict_ridge(bundle.ic_img[idx], bundle.ic_beta[idx], bundle.query_img, dev)[0],
            "A_k16": predict_lowrank(ic_img, ic_beta, q_img, basis16, f16)[0],
            "popeb_v2": predict_lowrank(ic_img, ic_beta, q_img, basis32, f32, Cpop=Cpop32, spatial=spatial32, F_stats=Fst32)[0],
            "braincorl": predict_icl(model, bundle.ic_img[idx], bundle.ic_beta[idx], bundle.query_img, bundle.query_beta, bundle.gt, dev, 128)[0],
        }
        for k, p in preds.items():
            ev = explained_variance(p, true).cpu().numpy(); np.save(OUT / f"{k}_num_ic={n}_rep={rep}_expl_var.npy", ev)
            res.setdefault((n, k), []).append(ev)
        print(f"E10 n={n} rep={rep} " + " ".join(f"{k}={explained_variance(p, true)[torch.from_numpy(good).to(dev)].mean().item():.4f}" for k, p in preds.items()))
rng = np.random.default_rng(0)
def boot(d):
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]); return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
print("\n## E10 EV ncsnr>0.2 (mean of 5 draws) and PopEB-v2 - BraInCoRL [CI]")
e10 = {}
for n in [10, 20]:
    M = {k: np.mean(res[(n, k)], 0) for k in ["ridge", "A_k16", "popeb_v2", "braincorl"]}
    d = boot((M["popeb_v2"] - M["braincorl"])[good])
    e10[n] = {k: float(v[good].mean()) for k, v in M.items()} | {"v2_minus_bc": [float(x) for x in d]}
    print(f"n={n}: " + " ".join(f"{k}={v[good].mean():.4f}" for k, v in M.items()) + f"   v2-BraInCoRL {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")
    print("   ROI (v2 / BraInCoRL): " + " ".join(f"{c} {M['popeb_v2'][rois[c]].mean():.3f}/{M['braincorl'][rois[c]].mean():.3f}" for c in ["faces", "places", "bodies", "words"]))

# ------------------------------------------------------------------ E12
idx = order[:100]
ic_img = torch.from_numpy(bundle.ic_img[idx]).to(dev); ic_beta = torch.from_numpy(bundle.ic_beta[idx]).to(dev)
def timed(fn, reps=3):
    fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(); return (time.time() - t) / reps, torch.cuda.max_memory_allocated() / 2 ** 30
t_v2, m_v2 = timed(lambda: predict_lowrank(ic_img, ic_beta, q_img, basis32, f32, Cpop=Cpop32, spatial=spatial32, F_stats=Fst32))
t_a, m_a = timed(lambda: predict_lowrank(ic_img, ic_beta, q_img, basis16, f16))
t_bc, m_bc = timed(lambda: predict_icl(model, bundle.ic_img[idx], bundle.ic_beta[idx], bundle.query_img, bundle.query_beta, bundle.gt, dev, 128), reps=1)
n_params_bc = sum(p.numel() for p in model.parameters()); n_params_b32 = sum(p.numel() for p in basis32.parameters())
print(f"\n## E12 adaptation cost on S1 (22,612 voxels, 100 support, 1,000 queries; mean of 3 runs)")
print(f"PopEB-v2: {t_v2:.2f} s, peak {m_v2:.2f} GiB (basis {n_params_b32:,} params + population {Cpop32.shape[1]} x 32)")
print(f"A (K16):  {t_a:.2f} s, peak {m_a:.2f} GiB")
print(f"BraInCoRL forward (batch 128 voxels, bf16): {t_bc:.2f} s, peak {m_bc:.2f} GiB ({n_params_bc:,} params)")

# ------------------------------------------------------------------ E15
nc = ncsnr ** 2 / (ncsnr ** 2 + 1.0 / 3.0)
ev_v2 = np.mean([np.load(f"outputs/propose01/popeb_v2_s1_ic100/lowrank_popeb_num_ic=100_rep={r}_expl_var.npy") for r in range(5)], 0)
ev_bc = np.mean([np.load(f"outputs/propose01/braincorl_s1_ic100/icl_num_ic=100_rep={r}_expl_var.npy") for r in range(5)], 0)
ev_a = np.mean([np.load(f"outputs/propose01/sweep/K16_a1/eval_lowrank_s1/lowrank_num_ic=100_rep={r}_expl_var.npy") for r in range(5)], 0)
print("\n## E15 noise-ceiling-normalised EV at n = 100 (EV / NC, mean over voxels in bin), NC = ncsnr^2/(ncsnr^2 + 1/3)")
bins = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 10]
e15 = []
for a, b in zip(bins[:-1], bins[1:]):
    m = good & (ncsnr >= a) & (ncsnr < b)
    row = dict(bin=f"[{a},{b})", n=int(m.sum()), NC=float(nc[m].mean()), A=float((ev_a[m] / nc[m]).mean()),
               popeb_v2=float((ev_v2[m] / nc[m]).mean()), braincorl=float((ev_bc[m] / nc[m]).mean()))
    e15.append(row); print(row)
m = good
print(dict(bin="all>0.2", n=int(m.sum()), NC=float(nc[m].mean()), A=float((ev_a[m] / nc[m]).mean()), popeb_v2=float((ev_v2[m] / nc[m]).mean()), braincorl=float((ev_bc[m] / nc[m]).mean())))
json.dump(dict(E10=e10, E12=dict(popeb_v2_s=t_v2, popeb_v2_gib=m_v2, A_s=t_a, A_gib=m_a, braincorl_s=t_bc, braincorl_gib=m_bc,
                                  params_braincorl=n_params_bc, params_basis32=n_params_b32), E15=e15),
          open(OUT / "summary.json", "w"), indent=1)
