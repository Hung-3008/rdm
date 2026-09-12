"""Fast stage + evaluation of the low-rank basis encoder on one held-out subject
(docs/propose/propose01.md, 3.2c and 5).

Support sets are the same consecutive slices of the source's support ordering as in
src.eval, so results are paired with a BraInCoRL run that used the same --num_ic /
--repeats. With --compare_dir pointing at that run's output, a paired bootstrap over
voxels gives a CI for the EV difference.

    python -m src.eval_lowrank --config src/config/lowrank_clip.yaml \
        --ckpt_dir outputs/propose01/sweep/K16_a1 --subject 1 --num_ic 100 --repeats 5 \
        --ridge --voxel_meta data/braincorl/s1_voxel_meta.npz \
        --compare_dir outputs/propose01/braincorl_s1_ic100

--prior_sigma SIGMA adds the method "lowrank_prior": ridge with the Gaussian anatomical prior mean
(sigma in mm) built from the training subjects of the config (their full-data coefficients in the
same basis and s{i}_coords.npy in data_dir).
--popeb adds the method "lowrank_popeb" (propose01.md section 9): PopEB posterior mean under the
population prior formed by the full-data coefficients of the training subjects' ncsnr > 0.2 voxels
(masks from --popeb_meta_dir/s{i}_voxel_meta.npz); tau2 / lam0 / spatial_sigma / std_global come from
the config's `fast` block (spatial weighting of the population by MNI distance and pooled feature
standardisation, propose01.md 9.6).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .dataset import SOURCES
from .eval import predict_ridge, roi_masks
from .model.lowrank_basis import (FunctionalBasis, LowRankConfig, feature_stats, full_coefficients,
                                  gaussian_anatomical_prior, predict_lowrank)
from .utils import explained_variance, point_mse, unit_norm


def paired_bootstrap(diff: np.ndarray, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--num_ic", type=int, nargs="+", default=[100])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--ridge", action="store_true")
    ap.add_argument("--voxel_meta", default=None)
    ap.add_argument("--compare_dir", default=None,
                    help="src.eval output dir with icl_num_ic=*_rep=*_expl_var.npy for paired bootstrap")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--prior_sigma", type=float, default=None,
                    help="mm; adds method lowrank_prior (Gaussian anatomical prior mean from the training subjects)")
    ap.add_argument("--popeb", action="store_true",
                    help="adds method lowrank_popeb (population-prior empirical Bayes, fast.tau2 / fast.lam0)")
    ap.add_argument("--popeb_meta_dir", default="data/braincorl",
                    help="directory with s{i}_voxel_meta.npz of the training subjects (ncsnr > 0.2 mask for the population)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir or ckpt_dir / f"eval_s{args.subject}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = SOURCES.build(cfg["data"]["source"]).load_eval(args.subject)
    basis = FunctionalBasis.from_file(ckpt_dir / "basis.pt", device)
    fcfg = LowRankConfig(**cfg.get("fast", {}))
    rois = roi_masks(args.voxel_meta) if args.voxel_meta else {}
    true = torch.from_numpy(bundle.query_beta.T).to(device)         # (V, Q)
    q_img = torch.from_numpy(bundle.query_img).to(device)

    C0, Cpop, spatial, F_stats, xyz_v = None, None, None, None, None
    if args.prior_sigma or args.popeb:
        src = cfg["data"]["source"]
        data_dir = Path(src["data_dir"])
        assert args.subject not in src["train_subjects"], "prior must not use the held-out subject"
        train, pop, pop_xyz, imgs = [], [], [], []
        for s in src["train_subjects"]:
            img = unit_norm(np.load(data_dir / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
            beta = np.load(data_dir / f"s{s}_unique_nrn_feats_msk.npy")
            Cs = full_coefficients(basis, img, beta, device)
            torch.cuda.empty_cache()
            imgs.append(img)
            if args.prior_sigma:
                train.append((np.load(data_dir / f"s{s}_coords.npy"), Cs))
            if args.popeb:
                good = np.load(Path(args.popeb_meta_dir) / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2
                pop.append(Cs[:, good])
                pop_xyz.append(np.load(data_dir / f"s{s}_coords.npy").astype(np.float32)[good])
        if args.prior_sigma:
            coords = np.load(data_dir / f"s{args.subject}_coords.npy")
            C0 = torch.from_numpy(gaussian_anatomical_prior(coords, train, args.prior_sigma)).to(device)
        if args.popeb:
            Cpop = torch.from_numpy(np.concatenate(pop, 1)).to(device)
            if fcfg.tau2_ratio > 0:
                fcfg.tau2 = float(fcfg.tau2_ratio * Cpop.var(1).mean().item())
            if fcfg.spatial_sigma or fcfg.match_sigma:
                xyz_v = torch.from_numpy(np.load(data_dir / f"s{args.subject}_coords.npy").astype(np.float32)).to(device)
            if fcfg.spatial_sigma:
                spatial = (xyz_v, torch.from_numpy(np.concatenate(pop_xyz, 0)).to(device), fcfg.spatial_sigma)
            if fcfg.std_global:
                F_stats = feature_stats(basis, imgs, device)
            print(f"[popeb] population: {Cpop.shape[1]} voxels from subjects {src['train_subjects']}, tau2={fcfg.tau2}, "
                  f"lam0={fcfg.lam0}, spatial_sigma={fcfg.spatial_sigma}, std_global={fcfg.std_global}, match_sigma={fcfg.match_sigma}")

    rows, cmp_rows, start = [], [], 0
    for num_ic in args.num_ic:
        for rep in range(args.repeats):
            idx = bundle.icl_order[start:start + num_ic]
            start += num_ic
            ic_img = torch.from_numpy(bundle.ic_img[idx]).to(device)
            ic_beta = torch.from_numpy(bundle.ic_beta[idx]).to(device)

            preds = {"lowrank": predict_lowrank(ic_img, ic_beta, q_img, basis, fcfg)}
            if C0 is not None:
                preds["lowrank_prior"] = predict_lowrank(ic_img, ic_beta, q_img, basis, fcfg, C0=C0)
            if Cpop is not None:
                preds["lowrank_popeb"] = predict_lowrank(ic_img, ic_beta, q_img, basis, fcfg, Cpop=Cpop,
                                                         spatial=spatial, F_stats=F_stats, xyz_v=xyz_v)
            if args.ridge:
                preds["ridge"] = (predict_ridge(bundle.ic_img[idx], bundle.ic_beta[idx], bundle.query_img, device)[0], {})

            ref = None
            if args.compare_dir:
                p = Path(args.compare_dir) / f"icl_num_ic={num_ic}_rep={rep}_expl_var.npy"
                ref = np.load(p) if p.exists() else None

            for method, (pred, info) in preds.items():
                ev = explained_variance(pred, true).cpu().numpy()
                np.save(out_dir / f"{method}_num_ic={num_ic}_rep={rep}_expl_var.npy", ev)
                row = dict(method=method, subject=args.subject, num_ic=num_ic, rep=rep,
                           expl_var=float(ev.mean()), pt_mse=point_mse(pred, true).mean().item(), **info)
                row.update({f"ev_{k}": float(ev[m].mean()) for k, m in rois.items()})
                rows.append(row)
                print(f"[{method}] num_ic={num_ic} rep={rep} "
                      + " ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)))
                if ref is not None:
                    sel = rois.get("ncsnr>0.2", np.ones(len(ev), bool))
                    mean, lo, hi = paired_bootstrap(ev[sel] - ref[sel])
                    cmp_rows.append(dict(method=method, num_ic=num_ic, rep=rep, roi="ncsnr>0.2",
                                         ev_method=float(ev[sel].mean()), ev_braincorl=float(ref[sel].mean()),
                                         diff=mean, ci_lo=lo, ci_hi=hi))
                    for c in ["faces", "places", "bodies", "words"]:
                        if c in rois:
                            mean, lo, hi = paired_bootstrap(ev[rois[c]] - ref[rois[c]])
                            cmp_rows.append(dict(method=method, num_ic=num_ic, rep=rep, roi=c,
                                                 ev_method=float(ev[rois[c]].mean()), ev_braincorl=float(ref[rois[c]].mean()),
                                                 diff=mean, ci_lo=lo, ci_hi=hi))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby(["method", "num_ic"]).mean(numeric_only=True).drop(columns=["rep", "subject"]).round(4))
    if cmp_rows:
        cdf = pd.DataFrame(cmp_rows)
        cdf.to_csv(out_dir / "compare_braincorl.csv", index=False)
        print(cdf.groupby(["method", "num_ic", "roi"])[["ev_method", "ev_braincorl", "diff", "ci_lo", "ci_hi"]]
              .mean().round(4))


if __name__ == "__main__":
    main()
