"""Summarise outputs/propose01/holdout/s*/ (scripts/propose01_holdout.sh): EV (ncsnr > 0.2) of BraInCoRL, Ridge,
A (K16), PopEB (K16) and PopEB-v2 (K32) per hold-out subject and support size, plus the paired-bootstrap
difference PopEB-v2 - BraInCoRL with CI, and per-ROI numbers at 100 support.

    python scripts/propose01_holdout_aggregate.py [outputs/propose01/holdout]
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/propose01/holdout")
pd.set_option("display.width", 250)
rows, cmp_rows, roi_rows = [], [], []
for d in sorted(root.glob("s*")):
    subj = int(d.name[1:])
    for tag in ["ic100", "scaling"]:
        bc = d / f"braincorl_{tag}" / "results.csv"
        if not bc.exists():
            continue
        b = pd.read_csv(bc)
        for n, g in b.groupby("num_ic"):
            rows.append(dict(subject=subj, num_ic=n, method="BraInCoRL", ev=g["ev_ncsnr>0.2"].mean()))
        for f, names in [(d / f"popeb_k16_{tag}", {"lowrank": "A (K16)", "lowrank_popeb": "PopEB (K16)", "ridge": "Ridge"}),
                         (d / f"popeb_v2_{tag}", {"lowrank_popeb": "PopEB-v2 (K32)"})]:
            if not (f / "results.csv").exists():
                continue
            r = pd.read_csv(f / "results.csv")
            for (m, n), g in r.groupby(["method", "num_ic"]):
                if m in names:
                    rows.append(dict(subject=subj, num_ic=n, method=names[m], ev=g["ev_ncsnr>0.2"].mean()))
                    if n == 100:
                        roi_rows.append(dict(subject=subj, method=names[m], **{c: g[f"ev_{c}"].mean() for c in ["faces", "places", "bodies", "words"]}))
            c = pd.read_csv(f / "compare_braincorl.csv")
            for (m, n, roi), g in c.groupby(["method", "num_ic", "roi"]):
                if m in names:
                    cmp_rows.append(dict(subject=subj, num_ic=n, method=names[m], roi=roi, diff=g["diff"].mean(),
                                         ci_lo=g["ci_lo"].mean(), ci_hi=g["ci_hi"].mean()))
        if tag == "ic100":
            b100 = b[b.num_ic == 100]
            roi_rows.append(dict(subject=subj, method="BraInCoRL", **{c: b100[f"ev_{c}"].mean() for c in ["faces", "places", "bodies", "words"]}))

df = pd.DataFrame(rows)
order = ["Ridge", "A (K16)", "PopEB (K16)", "PopEB-v2 (K32)", "BraInCoRL"]
print("## EV ncsnr > 0.2 by hold-out subject and support size")
print(df.pivot_table(index=["subject", "num_ic"], columns="method", values="ev").reindex(columns=[m for m in order if m in df.method.unique()]).round(4).to_string())
if cmp_rows:
    c = pd.DataFrame(cmp_rows)
    c = c[(c.method == "PopEB-v2 (K32)") & (c.roi == "ncsnr>0.2")]
    print("\n## PopEB-v2 - BraInCoRL (ncsnr > 0.2), mean over draws of the per-draw paired bootstrap")
    print(c.pivot_table(index="subject", columns="num_ic", values="diff").round(4).to_string())
    print("mean over subjects:", c.groupby("num_ic")["diff"].mean().round(4).to_dict())
if roi_rows:
    print("\n## 100 support, by ROI")
    print(pd.DataFrame(roi_rows).set_index(["subject", "method"]).round(4).to_string())
