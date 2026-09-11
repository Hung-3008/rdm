"""Few-shot evaluation on one held-out subject.

For each num_ic in --num_ic and each of --repeats trials, a consecutive slice of
the source's support ordering is used as the in-context set; every voxel is
scored on the held-out query images. Also runs the within-subject ridge baseline
on the same support images when --ridge is given.

    python -m src.eval --config src/config/smoke_synthetic.yaml \
        --ckpt outputs/smoke_synthetic/checkpoints/last.ckpt --subject 3 --num_ic 30 100 --ridge

Adapted from BraInCoRL scripts/test_det.py (Apache-2.0).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import load_config
from .dataset import SOURCES
from .engine import IclLightningModule
from .model import build_model
from .model.checkpoint import load_original_checkpoint
from .utils import explained_variance, normed_cos_similarity, point_mse, ridge_cv_brainwise


class VoxelEvalDataset(Dataset):
    def __init__(self, ic_img, ic_beta, q_img, q_beta, gt, coord=None):
        self.ic_img, self.q_img = torch.from_numpy(ic_img), torch.from_numpy(q_img)
        self.ic_beta, self.q_beta = torch.from_numpy(ic_beta), torch.from_numpy(q_beta)
        self.gt = torch.from_numpy(gt)
        self.coord = torch.from_numpy(coord) if coord is not None else None

    def __len__(self):
        return self.q_beta.shape[1]

    def __getitem__(self, v):
        coord = self.coord[v] if self.coord is not None else torch.zeros(3)
        return self.ic_img, self.ic_beta[:, v], self.q_img, self.q_beta[:, v], self.gt[v], coord


@torch.inference_mode()
def predict_icl(model, ic_img, ic_beta, q_img, q_beta, gt, device, batch_size, coord=None):
    dl = DataLoader(VoxelEvalDataset(ic_img, ic_beta, q_img, q_beta, gt, coord), batch_size=batch_size)
    preds, weights = [], []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for a, b, c, _, _, x in tqdm(dl, leave=False):
            p, w = model(a.to(device), b.to(device), c.to(device),
                         coord=x.to(device) if coord is not None else None)
            preds.append(p.float()), weights.append(w.float())
    return torch.cat(preds), torch.cat(weights)          # (V, Q), (V, E+1)


def predict_ridge(ic_img, ic_beta, q_img, device):
    X = torch.from_numpy(ic_img).to(device)
    y = torch.from_numpy(ic_beta.T).to(device)
    w = ridge_cv_brainwise(X, y)                          # (V, E+1)
    Xq = torch.cat([torch.from_numpy(q_img).to(device), torch.ones((q_img.shape[0], 1), device=device)], 1)
    return (Xq @ w.T).T, w


def roi_masks(voxel_meta):
    """ROI masks from s{N}_voxel_meta.npz (floc label>0, t>2, ncsnr>0.2) as in BraInCoRL Table 1."""
    meta = np.load(voxel_meta)
    good = meta["ncsnr"] > 0.2
    rois = {}
    for c in ["faces", "places", "bodies", "words"]:
        rois[c] = (meta[f"floc_{c}_label"] > 0) & (meta[f"floc_{c}_tval"] > 2) & good
    rois["ncsnr>0.2"] = good
    return rois


def score(pred, weights, q_beta, gt, device):
    true = torch.from_numpy(q_beta.T).to(device)
    gt_t = torch.from_numpy(gt).to(device)
    return dict(expl_var=explained_variance(pred, true), pt_mse=point_mse(pred, true),
                cos_sim=normed_cos_similarity(weights, gt_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--num_ic", type=int, nargs="+", default=[100])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--ridge", action="store_true")
    ap.add_argument("--original_ckpt", action="store_true",
                    help="ckpt is a checkpoint released with the original BraInCoRL repo")
    ap.add_argument("--coord_mode", default="real", choices=["real", "none", "shuffle"],
                    help="ablation for coordinate-aware models: drop the coordinate token, or "
                         "permute coordinates across voxels (seeded)")
    ap.add_argument("--voxel_meta", default=None,
                    help="s{N}_voxel_meta.npz from src.tools.preprocess_nsd; enables ROI-wise EV "
                         "(floc label>0, t>2, ncsnr>0.2) as in BraInCoRL Table 1")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir or Path(cfg["out_dir"]) / f"eval_s{args.subject}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = SOURCES.build(cfg["data"]["source"]).load_eval(args.subject)
    if args.coord_mode == "none":
        bundle.coord = None
    elif args.coord_mode == "shuffle" and bundle.coord is not None:
        bundle.coord = bundle.coord[np.random.default_rng(0).permutation(len(bundle.coord))]
    if args.original_ckpt:
        model = build_model(cfg["module"]["model"])
        load_original_checkpoint(model, args.ckpt)
        model = model.to(device).eval()
    else:
        module = IclLightningModule.load_from_checkpoint(args.ckpt, map_location=device)
        model = module.model.to(device).eval()

    rois = roi_masks(args.voxel_meta) if args.voxel_meta else {}
    if rois:
        print("[roi] voxel counts:", {k: int(v.sum()) for k, v in rois.items()})

    rows, start = [], 0
    for num_ic in args.num_ic:
        for rep in range(args.repeats):
            idx = bundle.icl_order[start:start + num_ic]
            start += num_ic
            ic_img, ic_beta = bundle.ic_img[idx], bundle.ic_beta[idx]

            runs = {"icl": predict_icl(model, ic_img, ic_beta, bundle.query_img, bundle.query_beta,
                                       bundle.gt, device, args.batch_size, coord=bundle.coord)}
            if args.ridge:
                runs["ridge"] = predict_ridge(ic_img, ic_beta, bundle.query_img, device)

            for method, (pred, w) in runs.items():
                m = score(pred, w, bundle.query_beta, bundle.gt, device)
                ev = m["expl_var"].cpu().numpy()
                np.save(out_dir / f"{method}_num_ic={num_ic}_rep={rep}_expl_var.npy", ev)
                row = dict(method=method, subject=args.subject, ckpt=args.ckpt, coord_mode=args.coord_mode,
                           num_ic=num_ic, rep=rep, **{k: v.mean().item() for k, v in m.items()})
                row.update({f"ev_{k}": float(ev[v].mean()) for k, v in rois.items()})
                rows.append(row)
                print(f"[{method}] num_ic={num_ic} rep={rep} "
                      + " ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["method", "num_ic"]).mean(numeric_only=True).drop(columns=["rep", "subject"]).round(4))


if __name__ == "__main__":
    main()
