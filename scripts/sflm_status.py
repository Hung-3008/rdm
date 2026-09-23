"""Progress of the SFLM-EB runs (scripts/sflm_fast.py).   python scripts/sflm_status.py [--out outputs/sflm_eb] [--ref sflm_default]

Scans outputs/sflm_eb/h*/results.csv and prints one row per (tag, fold, n): repeats done, mean EV, the paired difference
against --ref per repeat (same support indices by construction, scripts/sflm_fast.py:support_offset), the estimated
hyperparameters (tau2_hat, bracket position, alpha, sig2 median), iterations / evaluations, D5 counters and wall time;
then the queue of cells (fold, n, rep) still missing for each tag over 4 folds x 4 n x 3 repeats, and the running jobs.
Reads under a shared lock on results.lock (writers take an exclusive lock), so it is safe while jobs are writing.
"""
import argparse
import fcntl
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/sflm_eb")
ap.add_argument("--ref", default="sflm_default")
ap.add_argument("--tags", nargs="*", default=None, help="restrict to these tags (default: all)")
ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 5, 7])
ap.add_argument("--n", type=int, nargs="+", default=[30, 100, 300, 1000])
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()


def read(fold):
    p = Path(a.out) / f"h{fold}" / "results.csv"
    if not p.exists():
        return None
    for _ in range(3):
        try:
            with open(p.with_suffix(".lock"), "a+") as lk:
                fcntl.flock(lk, fcntl.LOCK_SH)
                df = pd.read_csv(p)
                fcntl.flock(lk, fcntl.LOCK_UN)
            df["fold"] = fold
            return df
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
            continue
    print(f"  h{fold}/results.csv: could not parse (being written?)")
    return None


frames = [d for d in (read(f) for f in a.folds) if d is not None]
if not frames:
    print("no results.csv found")
    raise SystemExit
df = pd.concat(frames, ignore_index=True)
for c in ("tau2", "tau2_pos", "alpha", "sig2_med", "iters", "evals", "n_floor", "n_clamp", "n_clip", "n_guard", "sec", "converged"):
    if c not in df:
        df[c] = np.nan
tags = a.tags or sorted(df.tag.unique(), key=lambda t: (t != a.ref, t))
ref = df[df.tag == a.ref].set_index(["fold", "n", "rep"])["ev"] if (df.tag == a.ref).any() else None


def fmt(x, f="{:.4f}"):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


hdr = f"{'tag':<22} {'fold':>4} {'n':>5} {'reps':<7} {'ev':>7} {'d_vs_ref (per rep)':<30} {'mean_d':>7} {'tau2':>8} {'pos':>5} {'alpha':>6} " \
      f"{'sig2':>6} {'it/ev':>7} {'fl/cl/cp/gd':>12} {'sec':>6} {'conv':>4}"
print(hdr)
print("-" * len(hdr))
for tag in tags:
    for fold in a.folds:
        for n in a.n:
            g = df[(df.tag == tag) & (df.fold == fold) & (df.n == n)].sort_values("rep")
            if g.empty:
                continue
            reps = "".join(str(int(r)) for r in g.rep)
            d = []
            if ref is not None and tag != a.ref:
                for _, r in g.iterrows():
                    k = (fold, n, int(r.rep))
                    d.append(r.ev - ref[k] if k in ref.index else np.nan)
            dtxt = " ".join(f"{x:+.4f}" if not np.isnan(x) else "   -   " for x in d) if d else "-"
            md = np.nanmean(d) if d and not all(np.isnan(d)) else np.nan
            cnt = "/".join(fmt(g[c].sum() if g[c].notna().any() else np.nan, "{:.0f}") for c in ("n_floor", "n_clamp", "n_clip", "n_guard"))
            itev = f"{fmt(g.iters.mean(), '{:.0f}')}/{fmt(g.evals.mean(), '{:.0f}')}"
            conv = "-" if g.converged.isna().all() else ("y" if g.converged.astype(bool).all() else "N")
            print(f"{tag:<22} {fold:>4} {n:>5} {reps:<7} {g.ev.mean():>7.4f} {dtxt:<30} {fmt(md, '{:+.4f}'):>7} {fmt(g.tau2.mean(), '{:.5f}'):>8} "
                  f"{fmt(g.tau2_pos.mean(), '{:.2f}'):>5} {fmt(g.alpha.mean(), '{:.2f}'):>6} {fmt(g.sig2_med.mean(), '{:.3f}'):>6} {itev:>7} "
                  f"{cnt:>12} {g.sec.mean():>6.0f} {conv:>4}")

print("\nqueue (cells missing over folds x n x reps)")
for tag in tags:
    have = set(map(tuple, df[df.tag == tag][["fold", "n", "rep"]].astype(int).values))
    miss = [(f, n, r) for f in a.folds for n in a.n for r in range(a.reps) if (f, n, r) not in have]
    if miss:
        by_fold = {f: [(n, r) for (ff, n, r) in miss if ff == f] for f in a.folds}
        txt = "; ".join(f"h{f}: " + ",".join(f"n{n}r{r}" for n, r in c) for f, c in by_fold.items() if c)
        print(f"  {tag}: {len(miss)} missing  {txt}")
    else:
        print(f"  {tag}: complete")

ps = subprocess.run(["pgrep", "-af", "sflm_fast.py"], capture_output=True, text=True).stdout.strip()
print("\nrunning jobs")
print("  " + ps.replace("\n", "\n  ") if ps else "  none")
