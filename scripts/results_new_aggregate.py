"""Tables for docs/results_new.md (plan P1-P7): EV per hold-out x support size x method, ROI table at n = 100, and the
paired voxel bootstrap PopEB-v2 - BraInCoRL-scratch on the same feature.

    python scripts/results_new_aggregate.py [--hvc]

Sources (missing ones are skipped):
  BraInCoRL-scratch <feat>   outputs/repro[_braincorl_hvc]/<feat>[_valuniq_h<H>][_seed1]/eval_s<H>_<tag>/ (method icl; ridge)
  PopEB-v2 pooled            outputs/repro/clip_b16/popeb_s<H>_<tag>/                             (method lowrank_popeb)
  PopEB-v2 PCA-512 <tag>     outputs/clip_layers/holdout[_braincorl_hvc]/<L>[_fit*]/s<H>/<run>_<tag>/ (lowrank_popeb; ridge)
Bootstrap: per-voxel EV averaged over the draws of each method, difference on voxels with ncsnr > 0.2, 2,000 resamples
of voxels; reported per hold-out and n.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROIS = ["faces", "places", "bodies", "words"]
L = "clip_pca512_clsmean_all"
FIT = {1: "257", 2: "157", 5: "127", 7: "125"}


def load(d, method, name):
    p = Path(d) / "results.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    df = df[df.method == method]
    return [dict(method=name, num_ic=n, ev=g["ev_ncsnr>0.2"].mean(), ev_all=g.expl_var.mean(), dir=str(d), vm=method,
                 **{c: g[f"ev_{c}"].mean() for c in ROIS}, reps=len(g)) for n, g in df.groupby("num_ic")]


def voxel_ev(d, method, n):
    fs = sorted(Path(d).glob(f"{method}_num_ic={n}_rep=*_expl_var.npy"))
    return np.mean([np.load(f) for f in fs], 0) if fs else None


def boot(a, b, mask, n_boot=2000, seed=0):
    d = (a - b)[mask]
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hvc", action="store_true", help="voxel set hvc (P7) instead of nsdgeneral")
    a = ap.parse_args()
    root = Path("data/braincorl_hvc" if a.hvc else "data/braincorl_nsdgeneral")
    rep = Path("outputs/repro_braincorl_hvc" if a.hvc else "outputs/repro")
    hold = Path("outputs/clip_layers/holdout_braincorl_hvc" if a.hvc else "outputs/clip_layers/holdout")
    pd.set_option("display.width", 250)
    rows = []
    for H in (1, 2, 5, 7):
        for tag in ("ic100", "scaling"):
            src = [(rep / f"clip_b16/eval_s{H}_{tag}", "icl", "BraInCoRL-scratch pooled"),
                   (rep / f"clip_b16/eval_s{H}_{tag}", "ridge", "Ridge pooled"),
                   (rep / f"clip_b16/popeb_s{H}_{tag}", "lowrank_popeb", "PopEB-v2 pooled"),
                   (rep / f"{L}/eval_s{H}_{tag}", "icl", "BraInCoRL-scratch PCA512"),
                   (rep / f"{L}_seed1/eval_s{H}_{tag}", "icl", "BraInCoRL-scratch PCA512 seed1"),
                   (rep / f"{L}_valuniq_h{H}/eval_s{H}_{tag}", "icl", "BraInCoRL-scratch PCA512 valuniq"),
                   (rep / f"{L}_valuniq_h{H}_seed1/eval_s{H}_{tag}", "icl", "BraInCoRL-scratch PCA512 valuniq seed1"),
                   (hold / f"{L}/s{H}/val600_{tag}", "lowrank_popeb", "PopEB-v2 PCA512 val600"),
                   (hold / f"{L}/s{H}/default_{tag}", "ridge", "Ridge PCA512"),
                   (hold / f"{L}/s{H}/default_{tag}", "lowrank_popeb", "PopEB-v2 PCA512"),
                   (hold / f"{L}/s{H}/default_seed1_{tag}", "lowrank_popeb", "PopEB-v2 PCA512 seed1"),
                   (hold / f"{L}/s{H}/default_seed2_{tag}", "lowrank_popeb", "PopEB-v2 PCA512 seed2"),
                   (hold / f"{L}/s{H}/nested_{tag}", "lowrank_popeb", "PopEB-v2 PCA512 nested-hp"),
                   (hold / f"{L}_fit{FIT[H]}/s{H}/default_{tag}", "lowrank_popeb", "PopEB-v2 PCA512 fit-train")]
            for d, m, name in src:
                rows += [dict(subject=H, **r) for r in load(d, m, name)]
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results"); return
    order = [m for m in dict.fromkeys(r["method"] for r in rows)]
    print("## EV, voxels ncsnr > 0.2")
    t = df.pivot_table(index=["subject", "num_ic"], columns="method", values="ev").reindex(columns=order)
    print(t.round(4).to_string())
    print("\n## mean over the 4 hold-outs (NaN if one is missing)")
    print(t.groupby("num_ic").agg(lambda x: x.mean() if x.notna().sum() == 4 else np.nan).round(4).T.to_string())
    print("\n## n = 100, 4-ROI table (S / ROI)")
    r = df[df.num_ic == 100].set_index(["method", "subject"])[ROIS]
    r["mean4"] = r.mean(axis=1)
    print(r.round(3).to_string())

    print("\n## paired voxel bootstrap, ncsnr > 0.2: diff [CI 95%]")
    pairs = [("PopEB-v2 PCA512 val600", "BraInCoRL-scratch PCA512 valuniq"),
             ("BraInCoRL-scratch PCA512 valuniq", "BraInCoRL-scratch PCA512"),
             ("PopEB-v2 PCA512 val600", "PopEB-v2 PCA512"),
             ("PopEB-v2 PCA512", "BraInCoRL-scratch PCA512"), ("PopEB-v2 pooled", "BraInCoRL-scratch pooled"),
             ("BraInCoRL-scratch PCA512", "BraInCoRL-scratch pooled"), ("PopEB-v2 PCA512 fit-train", "PopEB-v2 PCA512"),
             ("PopEB-v2 PCA512 nested-hp", "PopEB-v2 PCA512"), ("BraInCoRL-scratch PCA512 seed1", "BraInCoRL-scratch PCA512"),
             ("PopEB-v2 PCA512", "BraInCoRL-scratch PCA512 seed1")]
    idx = df.set_index(["method", "subject", "num_ic"])
    out = []
    for A, B in pairs:
        for (H, n) in sorted(set(zip(df.subject, df.num_ic))):
            if (A, H, n) not in idx.index or (B, H, n) not in idx.index:
                continue
            ra, rb = idx.loc[(A, H, n)], idx.loc[(B, H, n)]
            va, vb = voxel_ev(ra.dir, ra.vm, n), voxel_ev(rb.dir, rb.vm, n)
            if va is None or vb is None or va.shape != vb.shape:
                continue
            mask = np.load(root / f"s{H}_voxel_meta.npz")["ncsnr"] > 0.2
            dm, lo, hi = boot(va, vb, mask)
            out.append(dict(pair=f"{A} - {B}", subject=H, num_ic=n, diff=dm, ci=f"[{lo:+.4f}, {hi:+.4f}]", excl0=lo > 0 or hi < 0))
    if out:
        o = pd.DataFrame(out)
        for p, g in o.groupby("pair", sort=False):
            print(f"\n{p}")
            print(g.drop(columns="pair").to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
