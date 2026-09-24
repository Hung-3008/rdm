"""Tables of scripts/sflm_inpaint_cont.py over folds: per (scheme, n) the mean EV over the ncsnr > 0.2 voxels (every voxel is
predicted once per scheme, when its own mask is applied), paired differences per repeat, and EV by distance to the nearest
unmasked voxel.

    python scripts/sflm_inpaint_cont_table.py > outputs/sflm_ab/inpaint_table.txt
"""

import numpy as np
import pandas as pd

from pathlib import Path

DATA = Path("data/braincorl_nsdgeneral")
H_GRID = (2, 3, 5, 7, 10, 14)
BINS = (0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 1e9)


def main():
    rows, dist_rows = [], []
    for H in (1, 2, 5, 7):
        good = np.load(DATA / f"s{H}_voxel_meta.npz")["ncsnr"] > 0.2
        for f in sorted(Path(f"outputs/sflm_ab/h{H}/inpaint").glob("ev_*_n=*_rep=*.npz")):
            scheme = f.stem[3:].split("_n=")[0]
            n, rep = int(f.stem.split("_n=")[1].split("_")[0]), int(f.stem.split("_rep=")[1])
            d = np.load(f)
            sel = good & ~np.isnan(d["M0"])
            r = dict(fold=H, scheme=scheme, n=n, rep=rep, n_vox=int(sel.sum()))
            for m in ("M0", "NN", "KRIG", "own"):
                r[m] = float(d[m][sel].mean())
            r["KA_oracle"] = max(float(d[f"KA_{h}"][sel].mean()) for h in H_GRID)
            r["KAD_oracle"] = max(float(d[f"KAD_{h}"][sel].mean()) for h in H_GRID)
            rows.append(r)
            if scheme == "B_slab20":
                for lo, hi in zip(BINS[:-1], BINS[1:]):
                    s = sel & (d["d_nn"] >= lo) & (d["d_nn"] < hi)
                    dist_rows.append(dict(fold=H, n=n, rep=rep, bin=f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}", n_vox=int(s.sum()),
                                          M0=float(d["M0"][s].mean()), KRIG=float(d["KRIG"][s].mean()), own=float(d["own"][s].mean())))
    df = pd.DataFrame(rows)
    df["gain"] = df["KRIG"] - df["M0"]
    df["headroom"] = df["own"] - df["M0"]
    pd.set_option("display.width", 250)
    per_fold = df.groupby(["scheme", "n", "fold"]).agg(M0=("M0", "mean"), NN=("NN", "mean"), KA_or=("KA_oracle", "mean"),
                                                       KAD_or=("KAD_oracle", "mean"), KRIG=("KRIG", "mean"), own=("own", "mean"),
                                                       gain=("gain", "mean"), gain_min=("gain", "min"), reps=("rep", "count"))
    per_fold["frac"] = per_fold["gain"] / (per_fold["own"] - per_fold["M0"])
    print("# per fold (mean of repeats); gain = KRIG - M0 paired per repeat, gain_min over repeats; frac = gain / (own - M0)")
    print(per_fold.round(4).to_string())
    avg = per_fold.groupby(["scheme", "n"]).agg(M0=("M0", "mean"), NN=("NN", "mean"), KA_or=("KA_or", "mean"), KAD_or=("KAD_or", "mean"),
                                                KRIG=("KRIG", "mean"), own=("own", "mean"), gain=("gain", "mean"),
                                                folds_ge_0006=("gain", lambda g: int((g >= 0.006).sum())), frac=("frac", "mean"))
    print("\n# mean over folds; folds_ge_0006 = folds with mean gain >= 0.006 (H1 threshold)")
    print(avg.round(4).to_string())
    dd = pd.DataFrame(dist_rows)
    dd["gain"] = dd["KRIG"] - dd["M0"]
    print("\n# B_slab20: EV by distance (mm) to the nearest unmasked voxel, mean over folds and repeats")
    print(dd.groupby(["n", "bin"], sort=False).agg(n_vox=("n_vox", "mean"), M0=("M0", "mean"), KRIG=("KRIG", "mean"), own=("own", "mean"),
                                                   gain=("gain", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
