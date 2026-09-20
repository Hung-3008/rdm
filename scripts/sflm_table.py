"""Comparison tables for an sflm_pilot.py run.

    python scripts/sflm_table.py --run outputs/sflm/h1 --holdout 1

Table A: EV on nsdgeneral (ncsnr > 0.2) by support size, mean and std over support draws, plus streams.
Table B: EV at each n inside the floc category ROIs (label > 0, t > 2, ncsnr > 0.2), the voxel selection of
         BraInCoRL Table 1 (arXiv 2505.15813). NSD has no food localizer, so that column is absent.
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/braincorl_nsdgeneral")
# BraInCoRL Table 1, S1/S2 columns: faces, places, bodies, words (100 support images; ridge = within-subject)
PAPER = {1: {"FullyTrained-9000": [0.19, 0.20, 0.28, 0.11], "Ridge-100": [0.10, 0.08, 0.16, 0.02],
             "Ridge-300": [0.13, 0.13, 0.22, 0.06], "FsAverage map": [0.13, 0.11, 0.09, 0.06],
             "BraInCoRL-100": [0.16, 0.16, 0.25, 0.07]},
         2: {"FullyTrained-9000": [0.16, 0.27, 0.24, 0.11], "Ridge-100": [0.07, 0.14, 0.12, 0.03],
             "Ridge-300": [0.10, 0.20, 0.16, 0.06], "FsAverage map": [0.06, 0.19, 0.08, 0.03],
             "BraInCoRL-100": [0.13, 0.23, 0.21, 0.08]}}
CATS = ["faces", "places", "bodies", "words"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--holdout", type=int, required=True)
    a = ap.parse_args()
    run = Path(a.run)
    pd.set_option("display.width", 200)

    df = pd.read_csv(run / "results.csv")
    cols = ["ev", "ev_early", "ev_mid", "ev_ventral", "ev_lateral", "ev_parietal"]
    g = df.groupby(["method", "n"])
    print("## A. nsdgeneral, ncsnr > 0.2: EV mean (std over support draws)")
    ev = (g["ev"].mean().map("{:.4f}".format) + " (" + g["ev"].std().map("{:.4f}".format) + ")").unstack("n")
    print(ev.to_string())
    print("\n## A2. streams, mean over draws")
    print(g[cols].mean().round(4).to_string())

    m = np.load(ROOT / f"s{a.holdout}_voxel_meta.npz")
    good = m["ncsnr"] > 0.2
    roi = {c: good & (m[f"floc_{c}_label"] > 0) & (m[f"floc_{c}_tval"] > 2) for c in CATS}
    print("\nvoxels per ROI:", {c: int(v.sum()) for c, v in roi.items()})
    rows = []
    for f in sorted(glob.glob(str(run / "*_ev.npy"))):
        meth, n, rep = re.match(r"(.+)_n=(\d+)_rep=(\d+)_ev", Path(f).stem).groups()
        e = np.load(f)
        rows.append(dict(method=meth, n=int(n), rep=int(rep), **{c: e[roi[c]].mean() for c in CATS}))
    t = pd.DataFrame(rows).groupby(["method", "n"])[CATS].mean()
    t["mean4"] = t[CATS].mean(1)
    print("\n## B. floc category ROIs (this run)")
    print(t.round(3).to_string())
    if a.holdout in PAPER:
        p = pd.DataFrame(PAPER[a.holdout], index=CATS).T
        p["mean4"] = p[CATS].mean(1)
        print(f"\n## B-paper. BraInCoRL Table 1, S{a.holdout} (published, 2 decimals)")
        print(p.round(3).to_string())


if __name__ == "__main__":
    main()
