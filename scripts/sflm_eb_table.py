"""Paired 4-fold table for the SFLM-EB fast stage (scripts/sflm_fast_cont.py, scripts/sflm_fast.py) against a reference arm.

    python scripts/sflm_eb_table.py [--out outputs/sflm_ab] [--tag A_cont] [--ref eb_mackay] [--ref_out outputs/sflm_eb]

Per (fold, n): mean EV of both arms over the repeats present in both, the mean paired difference (same support indices by
construction), the number of positive differences; then the mean over folds per n and the acceptance rule of
docs/survey/sflm_constant_removal_2026-09-23.md §5 (mean paired difference >= -0.005 at every n on every fold), the
estimated tau2 per cell, and the streams at n = 100.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/sflm_ab")
ap.add_argument("--ref", default="eb_mackay")
ap.add_argument("--tag", default="A_cont")
ap.add_argument("--ref_out", default="outputs/sflm_eb", help="directory of the reference results")
ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 5, 7])
a = ap.parse_args()
pd.set_option("display.width", 220)

def load(root):
    frames = []
    for f in a.folds:
        p = Path(root) / f"h{f}" / "results.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["fold"] = f
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


df = load(a.out)
key = ["fold", "n", "rep"]
ref = load(a.ref_out or a.out)
ref = ref[ref.tag == a.ref].set_index(key)
arm = df[df.tag == a.tag].set_index(key)
both = ref.join(arm, how="inner", lsuffix="_ref", rsuffix="_arm")
both["d"] = both.ev_arm - both.ev_ref
g = both.groupby(["fold", "n"])
t = pd.DataFrame({"reps": g.size(), a.ref: g.ev_ref.mean(), a.tag: g.ev_arm.mean(), "d_mean": g.d.mean(),
                  "d_min": g.d.min(), "n_pos": g.d.apply(lambda x: int((x > 0).sum())),
                  "tau2": g.tau2_arm.mean(), "iters": g.iters_arm.mean(), "sec": g.sec_arm.mean()})
print(f"## paired EV (ncsnr > 0.2), {a.tag} vs {a.ref}, cells with both arms present")
print(t.round(4).to_string())
print("\n## mean over folds per n (cells present)")
m = both.groupby("n").agg(ref=("ev_ref", "mean"), arm=("ev_arm", "mean"), d_mean=("d", "mean"), d_min=("d", "min"),
                          n_pos=("d", lambda x: int((x > 0).sum())), n_cells=("d", "size"))
print(m.round(4).to_string())
tot = both.d
print(f"\ncells: {len(tot)}  positive: {int((tot > 0).sum())}  mean d: {tot.mean():+.4f}  min d: {tot.min():+.4f}")
fail = t[t.d_mean < -0.005]
print("acceptance (mean paired d >= -0.005 at every fold x n): " + ("PASS" if fail.empty and len(t) == 16 else
      f"{'FAIL' if not fail.empty else 'INCOMPLETE'} ({len(t)}/16 cells; failing: {list(fail.index)})"))
cols = [c for c in ("ev_early", "ev_mid", "ev_ventral", "ev_lateral", "ev_parietal") if f"{c}_arm" in both]
if cols:
    s = both.xs(100, level="n", drop_level=False) if 100 in both.index.get_level_values("n") else both.iloc[0:0]
    if len(s):
        print("\n## streams at n = 100, mean over folds x reps: arm - ref")
        print(pd.DataFrame({c: [s[f"{c}_ref"].mean(), s[f"{c}_arm"].mean(), (s[f"{c}_arm"] - s[f"{c}_ref"]).mean()] for c in cols},
                           index=[a.ref, a.tag, "d"]).round(4).to_string())
