"""E21 part D: does the population prior help most where the response is shared across subjects?
Per S1 voxel (ncsnr > 0.2): gain = EV(PopEB-v2) - EV(A) at n = 100 (mean of 5 draws, E07 / E01 files) against
cross-subject predictability = EV of ridge from S2/S5/S7 voxels on the common images (E21 part C, mean over
sources).  Spearman over voxels, overall and within ncsnr bins (to separate "shared" from "high SNR").
    python scripts/propose01_shared_latent_h2.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from src.eval import roi_masks  # noqa: E402

OUT = "outputs/propose01/shared_latent/"
r = roi_masks("data/braincorl/s1_voxel_meta.npz"); good = r["ncsnr>0.2"]; ncsnr = np.load("data/braincorl/s1_voxel_meta.npz")["ncsnr"][good]
def load5(pat):
    return np.mean([np.load(pat.format(k))[good] for k in range(5)], 0)
A = load5("outputs/propose01/final_s1_ic100_pervoxel/A_num_ic=100_rep={}_expl_var.npy")
P = load5("outputs/propose01/popeb_v2_s1_ic100/lowrank_popeb_num_ic=100_rep={}_expl_var.npy")
B = load5("outputs/propose01/braincorl_s1_ic100/icl_num_ic=100_rep={}_expl_var.npy")
pv = np.load(OUT + "voxel_prediction_pervoxel.npz")
xs = np.mean([pv[f"S1_from_S{j}_ridge_subj"] for j in [2, 5, 7]], 0)      # cross-subject predictability
xc = np.mean([pv[f"S1_from_S{j}_ridge_clip"] for j in [2, 5, 7]], 0)      # CLIP predictability (same fit size, 500)
xf = np.mean([pv[f"S1_from_S{j}_func_nn"] for j in [2, 5, 7]], 0)
gain = P - A; gain_b = P - B
print(f"S1 ncsnr>0.2 voxels {good.sum()}: mean EV A {A.mean():.4f} PopEB-v2 {P.mean():.4f} BraInCoRL {B.mean():.4f}; gain {gain.mean():.4f}")
print("Spearman over voxels:")
for name, x in [("xsubj ridge", xs), ("clip ridge", xc), ("xsubj func_nn", xf), ("ncsnr", ncsnr), ("xsubj - clip", xs - xc)]:
    print(f"  gain(PopEB-A) vs {name:14s} rho {spearmanr(gain, x)[0]:+.3f} | gain(PopEB-BraInCoRL) rho {spearmanr(gain_b, x)[0]:+.3f}")
# within ncsnr bins
rows = []
for lo, hi in [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 9)]:
    m = (ncsnr >= lo) & (ncsnr < hi)
    rows.append(dict(bin=f"[{lo},{hi})", n=int(m.sum()), rho_xsubj=spearmanr(gain[m], xs[m])[0], rho_clip=spearmanr(gain[m], xc[m])[0],
                     rho_xsubj_minus_clip=spearmanr(gain[m], (xs - xc)[m])[0], gain=gain[m].mean(), xsubj=xs[m].mean(), clip=xc[m].mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# quartiles of (xsubj - clip) within each ncsnr bin: gain per quartile
print("gain by quartile of (xsubj - clip) residualised on ncsnr bin:")
q = np.zeros_like(xs, int)
for lo, hi in [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 9)]:
    m = (ncsnr >= lo) & (ncsnr < hi); d = (xs - xc)[m]; q[m] = np.searchsorted(np.quantile(d, [0.25, 0.5, 0.75]), d)
print(pd.DataFrame(dict(gain_vs_A=[gain[q == k].mean() for k in range(4)], gain_vs_B=[gain_b[q == k].mean() for k in range(4)],
                        xsubj_minus_clip=[(xs - xc)[q == k].mean() for k in range(4)]), index=["Q1", "Q2", "Q3", "Q4"]).round(4))
