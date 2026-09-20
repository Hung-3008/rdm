"""Aggregate outputs/foveacl/ceiling.csv into the H1-H3 tables of docs/propose/propose04.md section 6.

    python scripts/foveacl_report.py
Condition = run name without the seed suffix. `sel` = mean valuniq score over S2/S5/S7 (selection); `common` = EV on the
1,000 common images, mean over S1/S2/S5/S7 (reporting). Seed spread = max - min over seeds of the 4-subject mean.
"""
import pandas as pd

d = pd.read_csv("outputs/foveacl/ceiling.csv")
d = d[d.feats == "nsd_feats.npy"].copy()
d["cond"] = d.run.str.replace(r"_s\d+$", "", regex=True)
G = ["early", "higher", "ncsnr>0.2", "faces", "places", "bodies", "words"]
per_run = d.groupby(["cond", "run"])[[f"common_{g}" for g in G]].mean()
sel = d[d.subject != 1].groupby(["cond", "run"])[[f"valuniq_{g}" for g in G[:3]]].mean()
pd.set_option("display.width", 250)
print("== per run, common images, mean of 4 subjects"); print(per_run.round(4).to_string())
print("\n== per run, valuniq (S2/S5/S7), selection score"); print(sel.round(4).to_string())
m = per_run.groupby("cond").agg(["mean", lambda x: x.max() - x.min(), "count"])
print("\n== per condition: mean / seed range / n seeds")
print(pd.DataFrame({g: m[f"common_{g}"]["mean"].round(4).astype(str) + " (" + m[f"common_{g}"]["<lambda_0>"].round(4).astype(str) + ")" for g in G[:3]}
                   ).assign(n=m["common_early"]["count"]).to_string())
mean = per_run.groupby("cond").mean()
def diff(a, b): return (mean.loc[a] - mean.loc[b]).round(4)
print("\n== differences (common, 4-subject mean)")
for name, a, b in [("H2: R1 - R0", "R1", "R0"), ("H1: R2 - R1", "R2", "R1"), ("H1: R3a - R1", "R3a", "R1"), ("H1: R3b - R1", "R3b", "R1"), ("H3: R4 - R2", "R4", "R2")]:
    if a in mean.index and b in mean.index:
        print(name, diff(a, b).to_dict())
# per-subject paired differences of condition means
ps = d.groupby(["cond", "subject"])[[f"common_{g}" for g in G[:2]]].mean()
print("\n== per-subject differences (early, higher)")
for a, b in [("R1", "R0"), ("R2", "R1"), ("R3a", "R1"), ("R3b", "R1"), ("R4", "R2")]:
    if a in mean.index and b in mean.index:
        print(a, "-", b); print((ps.loc[a] - ps.loc[b]).round(4).T.to_string())
