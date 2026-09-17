"""Retinotopic-coordinate prior for early visual voxels (V1-hV4) in PopEB, and the MNI-prior gain split by ROI.

Voxel set nsdgeneral (docs/results.md E19), hold-out S1, basis K32 trained on S2/S5/S7, same support draws as
E19 (num_ic 100 x 5 reps; 30/300/1000 x 3 reps). Population weights w_vj of PopEB (propose01.md 9.6) are
multiplied by exp(-d^2 / 2) where d is one of:
    nospatial   none
    mni10       |MNI_v - MNI_j| / 10 mm                                (PopEB-v2, reproduces E19)
    mni10_early population restricted to early voxels of the training subjects, MNI weighting (control)
    prf{s}      |pRF_v - pRF_j| / s deg, pRF = ecc (cos a, sin a); population = early voxels only
    prf{s}_mni  sqrt(|pRF|^2 / s^2 + |MNI|^2 / 10^2), population = early voxels only
The prf* / *_early variants only re-predict the early voxels of S1 (prf-visualrois > 0); other voxels keep mni10.
Continuous pRF maps: data/nsd_prf/subj{XX}/prf_{eccentricity,angle,R2}.nii.gz (NSD S3, func1pt8mm).

    python scripts/propose01_retino.py --out outputs/propose01/retino
"""
import argparse, sys, time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.config import load_config  # noqa
from src.dataset import SOURCES  # noqa
from src.model.lowrank_basis import (FunctionalBasis, LowRankConfig, feature_stats, full_coefficients,  # noqa
                                     predict_lowrank)
from src.utils import explained_variance, unit_norm  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", type=int, default=1)
ap.add_argument("--train", type=int, nargs="+", default=[2, 5, 7])
ap.add_argument("--data", default="data/braincorl_nsdgeneral")
ap.add_argument("--basis", default="outputs/propose01/nsdgeneral/K32/basis.pt")
ap.add_argument("--config", default="src/config/popeb_clip.yaml")
ap.add_argument("--prf_sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
ap.add_argument("--out", default="outputs/propose01/retino")
ap.add_argument("--check_dir", default="outputs/propose01/nsdgeneral",
                help="E19 output root; mni10 per-voxel EV is compared with popeb_v2_* there")
args = ap.parse_args()

D, L = Path(args.data), Path(args.data) / "clip_b16"
NSD = Path("data/NSD/data/nsddata/ppdata")
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
h, tr = args.holdout, args.train
t0 = time.time()


# ------------------------------------------------------------------ per-voxel metadata
def voxel_info(s):
    m = np.load(D / f"s{s}_voxel_meta.npz"); idx = m["idx"]
    ld = lambda f: np.asarray(nib.load(f).dataobj, dtype=np.float32).ravel()[idx]
    ecc = ld(f"data/nsd_prf/subj{s:02d}/prf_eccentricity.nii.gz")
    ang = np.deg2rad(ld(f"data/nsd_prf/subj{s:02d}/prf_angle.nii.gz"))
    return dict(mni=np.load(L / f"s{s}_coords.npy").astype(np.float32),
                prf=np.stack([ecc * np.cos(ang), ecc * np.sin(ang)], 1).astype(np.float32),
                r2=ld(f"data/nsd_prf/subj{s:02d}/prf_R2.nii.gz"),
                vroi=ld(NSD / f"subj{s:02d}/func1pt8mm/roi/prf-visualrois.nii.gz").astype(int),
                streams=m["streams"].astype(int), ncsnr=m["ncsnr"],
                floc={c: (m[f"floc_{c}_label"] > 0) & (m[f"floc_{c}_tval"] > 2) for c in ["faces", "places", "bodies", "words"]})


info = {s: voxel_info(s) for s in [h] + tr}
early_h = info[h]["vroi"] > 0
print(f"S{h}: {len(early_h)} voxels, early (prf-visualrois>0) {early_h.sum()}, "
      + ", ".join(f"S{s} early {(info[s]['vroi'] > 0).sum()}" for s in tr))

# ------------------------------------------------------------------ basis, population
cfg = load_config(args.config, [f"data.source.data_dir={L}"])
fcfg = LowRankConfig(**cfg["fast"])
basis = FunctionalBasis.from_file(args.basis, dev)
pop, pop_mni, pop_prf, pop_early, imgs = [], [], [], [], []
for s in tr:
    img = unit_norm(np.load(L / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
    beta = np.load(L / f"s{s}_unique_nrn_feats_msk.npy")
    Cs = full_coefficients(basis, img, beta, dev); torch.cuda.empty_cache()
    good = info[s]["ncsnr"] > 0.2
    imgs.append(img); pop.append(Cs[:, good]); pop_mni.append(info[s]["mni"][good]); pop_prf.append(info[s]["prf"][good])
    pop_early.append((info[s]["vroi"] > 0)[good])
Cpop = torch.from_numpy(np.concatenate(pop, 1)).to(dev)
pop_mni, pop_prf, pop_early = map(np.concatenate, (pop_mni, pop_prf, pop_early))
fcfg.tau2 = float(fcfg.tau2_ratio * Cpop.var(1).mean().item())
F_stats = feature_stats(basis, imgs, dev)
T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
print(f"population {Cpop.shape[1]} voxels (early {pop_early.sum()}), tau2 {fcfg.tau2:.3e}, {time.time() - t0:.0f}s")

# ------------------------------------------------------------------ variants: (voxel mask of S1, population mask, scaled coords)
mni_h, prf_h = info[h]["mni"], info[h]["prf"]
all_h = np.ones(len(mni_h), bool)
variants = {
    "nospatial": (all_h, np.ones(len(pop_mni), bool), None),
    "mni10": (all_h, np.ones(len(pop_mni), bool), (mni_h / 10.0, pop_mni / 10.0)),
    "mni10_early": (early_h, pop_early, (mni_h / 10.0, pop_mni / 10.0)),
}
for s in args.prf_sigmas:
    variants[f"prf{s:g}"] = (early_h, pop_early, (prf_h / s, pop_prf / s))
    variants[f"prf{s:g}_mni"] = (early_h, pop_early, (np.c_[prf_h / s, mni_h / 10.0], np.c_[pop_prf / s, pop_mni / 10.0]))

# ------------------------------------------------------------------ hold-out, support draws as in E19
bundle = SOURCES.build(dict(name="npy_multisubj", data_dir=str(L), train_subjects=tr, val_subjects=tr[:1])).load_eval(h)
true = torch.from_numpy(bundle.query_beta.T).to(dev)
q_img = torch.from_numpy(bundle.query_img).to(dev)
draws = []
for tag, nics, reps in [("ic100", [100], 5), ("scaling", [30, 300, 1000], 3)]:
    start = 0
    for n in nics:
        for r in range(reps):
            draws.append((tag, n, r, bundle.icl_order[start:start + n])); start += n

ev = {}   # (variant, n, rep) -> per-voxel EV over the variant's voxel mask
for tag, n, r, idx in draws:
    ic_img, ic_beta = torch.from_numpy(bundle.ic_img[idx]).to(dev), torch.from_numpy(bundle.ic_beta[idx]).to(dev)
    for name, (vm, pm, sc) in variants.items():
        spatial = None if sc is None else (T(sc[0][vm]), T(sc[1][pm]), 1.0)
        pred, _ = predict_lowrank(ic_img, ic_beta[:, vm], q_img, basis, fcfg, Cpop=Cpop[:, torch.from_numpy(pm).to(dev)],
                                  spatial=spatial, F_stats=F_stats)
        e = explained_variance(pred, true[vm]).cpu().numpy()
        ev[(name, n, r)] = e
        np.save(out / f"{name}_num_ic={n}_rep={r}_expl_var.npy", e)
    ref = Path(args.check_dir) / f"popeb_v2_{tag}" / f"lowrank_popeb_num_ic={n}_rep={r}_expl_var.npy" if h == 1 else Path("/nonexistent")
    chk = f", |mni10 - E19| max {np.abs(ev[('mni10', n, r)] - np.load(ref)).max():.1e}" if ref.exists() else ""
    sub = lambda e: e[early_h] if len(e) == len(early_h) else e
    print(f"n={n} rep={r}: " + " ".join(f"{k}={sub(ev[(k, n, r)]).mean():.4f}" for k in variants)
          + f" (early voxels){chk}, {time.time() - t0:.0f}s")

# ------------------------------------------------------------------ tables
def boot(diff, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


good_h = info[h]["ncsnr"] > 0.2
st = info[h]["streams"]
rois = {"early": early_h, "V1": np.isin(info[h]["vroi"], [1, 2]), "V2": np.isin(info[h]["vroi"], [3, 4]),
        "V3": np.isin(info[h]["vroi"], [5, 6]), "hV4": info[h]["vroi"] == 7,
        "early_R2>20": early_h & (info[h]["r2"] > 20), "early_ecc<2": early_h & (np.linalg.norm(prf_h, axis=1) < 2),
        "early_ecc>=2": early_h & (np.linalg.norm(prf_h, axis=1) >= 2),
        "mid(streams2-4)": np.isin(st, [2, 3, 4]), "ventral": st == 5, "lateral": st == 6, "parietal": st == 7,
        **info[h]["floc"], "all": all_h}
rows = []
for n in sorted({d[1] for d in draws}):
    reps = sorted({d[2] for d in draws if d[1] == n})
    full = {k: np.mean([ev[(k, n, r)] for r in reps], 0) for k in ["nospatial", "mni10"]}          # (V,)
    early = {k: np.mean([ev[(k, n, r)] for r in reps], 0) for k in variants if k not in full}        # (V_early,)
    for roi, m in rois.items():
        for k, e in full.items():
            rows.append(dict(num_ic=n, roi=roi, variant=k, ev=e[m].mean(), ev_ncsnr=e[m & good_h].mean(), n_vox=int(m.sum())))
        d, lo, hi = boot(full["mni10"][m] - full["nospatial"][m])
        rows.append(dict(num_ic=n, roi=roi, variant="mni10-nospatial", ev=d, ci_lo=lo, ci_hi=hi, n_vox=int(m.sum())))
        if (m & ~early_h).any():
            continue
        me = m[early_h]; base = full["mni10"][early_h]
        for k, e in early.items():
            d, lo, hi = boot(e[me] - base[me])
            rows.append(dict(num_ic=n, roi=roi, variant=k, ev=e[me].mean(), ev_ncsnr=e[me & good_h[early_h]].mean(),
                             diff_vs_mni10=d, ci_lo=lo, ci_hi=hi, n_vox=int(m.sum())))
df = pd.DataFrame(rows)
df.to_csv(out / "summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("\n== MNI prior gain (mni10 - nospatial) by ROI, EV on all voxels of the ROI")
print(df[df.variant == "mni10-nospatial"].pivot(index="roi", columns="num_ic", values="ev").round(4))
print("\n== early voxels: EV by variant (rows) and n (cols)")
print(df[df.roi == "early"].pivot(index="variant", columns="num_ic", values="ev").round(4))
print("\n== early voxels: diff vs mni10 [CI 95%]")
e = df[(df.roi == "early") & df.diff_vs_mni10.notna()]
print(e.assign(s=lambda x: x.apply(lambda r: f"{r.diff_vs_mni10:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}]", axis=1))
      .pivot(index="variant", columns="num_ic", values="s"))
print("\n== by early sub-ROI at n=100: diff vs mni10")
print(df[(df.num_ic == 100) & df.diff_vs_mni10.notna()].pivot(index="variant", columns="roi", values="diff_vs_mni10").round(4))
print(f"done {time.time() - t0:.0f}s")
