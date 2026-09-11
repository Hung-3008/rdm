"""Aggregate the propose01 low-rank sweep (scripts/propose01_sweep.sh) into tables.

    python scripts/propose01_aggregate.py [outputs/propose01] [eval-dir suffix, default _lowrank]

Tables: (1) nested leave-one-training-subject-out EV per K (selection criterion);
(2) S1 EV per K; (3) paired bootstrap vs BraInCoRL for the selected K.
"""

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/propose01")
suffix = sys.argv[2] if len(sys.argv) > 2 else "_lowrank"
ROI = ["ev_faces", "ev_places", "ev_bodies", "ev_words", "ev_ncsnr>0.2"]


def collect(pattern, parse):
    rows = []
    for d in sorted(root.glob(pattern)):
        if not d.is_dir():
            continue
        for f in d.glob(f"eval{suffix}_s*/results.csv"):
            df = pd.read_csv(f)
            for k, v in parse(d.name).items():
                df[k] = v
            rows.append(df)
    return pd.concat(rows)


nested = collect("sweep_nested/K*_a1_h*", lambda n: dict(K=int(n.split("_")[0][1:]), held=int(n.split("_")[2][1:])))
sel = nested[nested.method != "ridge"].groupby("K")[ROI].mean()
print("## nested leave-one-training-subject-out (mean over S2/S5/S7 held-out, 100 support x 3 draws)")
print(sel.round(4).to_string())
best_K = int(sel["ev_ncsnr>0.2"].idxmax())
print(f"\nselected K={best_K}\n\nper held-out subject, K={best_K}:")
print(nested[nested.K == best_K].groupby(["held", "method"])[ROI].mean().round(4).to_string())

s1 = collect("sweep/K*_a1", lambda n: dict(K=int(n.split("_")[0][1:])))
print("\n## S1 (train S2/S5/S7), 100 support x 5 draws, by K")
print(s1.groupby(["K", "method"])[ROI].mean().round(4).to_string())

cfg = root / "sweep" / f"K{best_K}_a1"
print(f"\n## selected K on S1: {cfg.name}")
r = pd.read_csv(cfg / f"eval{suffix}_s1" / "results.csv")
print(r.groupby("method")[ROI].agg(["mean", "std"]).round(4).to_string())
c = pd.read_csv(cfg / f"eval{suffix}_s1" / "compare_braincorl.csv")
print("\npaired bootstrap vs BraInCoRL (mean over draws of per-draw diff and CI):")
print(c.groupby(["method", "roi"])[["ev_method", "ev_braincorl", "diff", "ci_lo", "ci_hi"]].mean().round(4).to_string())
