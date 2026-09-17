"""Hold-out S1 table for docs/results.md E34: MedARC trunk (CLIP / 3 backbones) + ridge or PopEB vs BraInCoRL-scratch
and PopEB-v2 of E20 on the same support draws. Mean over draws of the per-voxel metric, all voxels and ncsnr > 0.2;
paired bootstrap over voxels of the draw-averaged difference.
    PYTHONPATH=. python scripts/medarc_holdout_aggregate.py
"""
import numpy as np
import pandas as pd

from src.eval import roi_masks
from src.eval_lowrank import paired_bootstrap

R, M = "outputs/repro/clip_b16", "outputs/medarc/holdout"
METHODS = {
    "Ridge (CLIP)": f"{R}/eval_s1_{{tag}}/ridge",
    "BraInCoRL-scratch (CLIP)": f"{R}/eval_s1_{{tag}}/icl",
    "PopEB-v2 (CLIP, K32)": f"{R}/popeb_s1_{{tag}}/lowrank_popeb",
    "MedARC-CLIP + ridge": f"{M}/s1_clip_b16_d192/eval_{{tag}}/lowrank",
    "MedARC-CLIP + PopEB": f"{M}/s1_clip_b16_d192/eval_{{tag}}/lowrank_popeb",
    "Ridge (3 backbone concat)": f"{M}/s1_concat3_d192/eval_{{tag}}/ridge",
    "MedARC-3bb + ridge": f"{M}/s1_concat3_d192/eval_{{tag}}/lowrank",
    "MedARC-3bb + PopEB": f"{M}/s1_concat3_d192/eval_{{tag}}/lowrank_popeb",
}
DRAWS = {30: ("scaling", 3), 100: ("ic100", 5), 300: ("scaling", 3), 1000: ("scaling", 3)}
good = roi_masks("data/braincorl_nsdgeneral/s1_voxel_meta.npz")["ncsnr>0.2"]


def load(method, n, metric):
    tag, reps = DRAWS[n]
    return np.mean([np.load(METHODS[method].format(tag=tag) + f"_num_ic={n}_rep={r}_{metric}.npy") for r in range(reps)], 0)


# pairing check: ridge on CLIP in the MedARC run must equal the E20 ridge (same support draws)
for n, (tag, reps) in DRAWS.items():
    a = np.load(f"{M}/s1_clip_b16_d192/eval_{tag}/ridge_num_ic={n}_rep=0_expl_var.npy")
    b = np.load(f"{R}/eval_s1_{tag}/ridge_num_ic={n}_rep=0_expl_var.npy")
    print(f"pairing check n={n}: max |ridge diff| = {np.abs(a - b).max():.2e}")

rows = []
for m in METHODS:
    row = {"method": m}
    for n in DRAWS:
        ev, r = load(m, n, "expl_var"), load(m, n, "pearson")
        row[f"n={n}"] = f"{ev.mean():.4f} / {r.mean():.3f} ({ev[good].mean():.4f})"
    rows.append(row)
print("EV all / Pearson all (EV ncsnr>0.2)")
print(pd.DataFrame(rows).to_string(index=False))

for ref in ["BraInCoRL-scratch (CLIP)", "PopEB-v2 (CLIP, K32)"]:
    print(f"\npaired difference vs {ref}: EV all [CI] | Pearson all [CI]")
    for m in [k for k in METHODS if k.startswith("MedARC")]:
        cells = []
        for n in DRAWS:
            out = []
            for metric in ["expl_var", "pearson"]:
                mean, lo, hi = paired_bootstrap(load(m, n, metric) - load(ref, n, metric))
                out.append(f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            cells.append(f"n={n}: " + " | ".join(out))
        print(m); print("  " + "\n  ".join(cells))
