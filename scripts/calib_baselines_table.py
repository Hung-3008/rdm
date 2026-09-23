"""Markdown tables from outputs/calib/baselines_s*/results.csv (scripts/calib_baselines.py).

    python scripts/calib_baselines_table.py
"""

import glob

import pandas as pd

def md(t):
    t = t.reset_index()
    rows = [list(map(str, t.columns)), ["---"] * t.shape[1]] + t.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


ORDER = ["null_n01", "ridge_homo", "bayes_global", "bayes_voxel", "bayes_voxel_t", "bayes_pop", "braincorl_homo"]

df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("outputs/calib/baselines_s*/results.csv")))
df["cov_err_max"] = df[["cov50", "cov80", "cov90"]].sub([0.5, 0.8, 0.9]).abs().max(axis=1) * 100   # points
print(f"subjects {sorted(df.subject.unique())}, seeds per cell {df.groupby(['method', 'subject', 'k']).size().unique()}\n")

# mean over seeds within a subject, then mean / min / max over subjects
ps = df.groupby(["method", "k", "subject"]).mean(numeric_only=True).reset_index()
for col, fmt in [("cov90", "{:.3f}"), ("cov_err_max", "{:.1f}"), ("cov90_vox_mae", "{:.3f}"), ("nll", "{:.4f}"),
                 ("crps", "{:.4f}"), ("r_nc02", "{:.3f}"), ("rho_var_1mNC", "{:.2f}")]:
    g = ps.groupby(["method", "k"])[col]
    t = g.mean().map(fmt.format) + " [" + g.min().map(fmt.format) + ", " + g.max().map(fmt.format) + "]"
    print(f"### {col}: mean over subjects [min, max]\n")
    print(md(t.unstack("k").reindex(ORDER)), "\n")
