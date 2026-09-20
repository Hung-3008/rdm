"""PopEB-v2 with the MLP basis replaced by a PCA projection of the pooled CLIP features.

build:   top-K principal directions of the training subjects' unique-image features (no fMRI), saved as a
         FunctionalBasis with hidden_dim = 0 so that src.eval_lowrank loads it unchanged.
report:  EV on ncsnr > 0.2 voxels per n, PCA basis vs the MLP basis run on the same support draws
         (paired bootstrap over voxels, mean over draws), plus the BraInCoRL EV stored next to the MLP run.

    python scripts/propose01_pca_basis.py build --data_dir DIR --train 2 5 7 --K 32 --out OUT/basis.pt
    python scripts/propose01_pca_basis.py report --pca_dir A --mlp_dir B --voxel_meta s1_voxel_meta.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval_lowrank import paired_bootstrap
from src.model.lowrank_basis import FunctionalBasis
from src.utils import unit_norm


def build(args):
    X = np.concatenate([unit_norm(np.load(Path(args.data_dir) / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
                        for s in args.train]).astype(np.float64)
    X = np.unique(X, axis=0)                                   # images shared between subjects counted once
    evals, evecs = np.linalg.eigh(np.cov(X - X.mean(0), rowvar=False))
    order = np.argsort(evals)[::-1][:args.K]
    basis = FunctionalBasis(X.shape[1], 0, args.K)
    basis.lin2.weight.data = torch.from_numpy(evecs[:, order].T.astype(np.float32).copy())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cfg=basis.cfg, state_dict=basis.state_dict()), args.out)
    print(f"[pca] {len(X)} images, K={args.K}, variance kept {evals[order].sum() / evals.sum():.4f} -> {args.out}")


def report(args):
    good = np.load(args.voxel_meta)["ncsnr"] > 0.2
    for tag in ["ic100", "scaling"]:
        pca, mlp = Path(f"{args.pca_dir}_{tag}"), Path(f"{args.mlp_dir}_{tag}")
        for n in sorted({int(p.name.split("=")[1].split("_")[0]) for p in pca.glob("lowrank_popeb_num_ic=*_expl_var.npy")}):
            a = np.mean([np.load(p) for p in sorted(pca.glob(f"lowrank_popeb_num_ic={n}_rep=*_expl_var.npy"))], 0)[good]
            b = np.mean([np.load(p) for p in sorted(mlp.glob(f"lowrank_popeb_num_ic={n}_rep=*_expl_var.npy"))], 0)[good]
            r_a = np.mean([np.load(p) for p in sorted(pca.glob(f"lowrank_num_ic={n}_rep=*_expl_var.npy"))], 0)[good]
            r_b = np.mean([np.load(p) for p in sorted(mlp.glob(f"lowrank_num_ic={n}_rep=*_expl_var.npy"))], 0)[good]
            d, lo, hi = paired_bootstrap(a - b)
            print(f"{args.name} n={n}: popeb pca {a.mean():.4f} mlp {b.mean():.4f} diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                  f"ridge-in-basis pca {r_a.mean():.4f} mlp {r_b.mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data_dir", required=True); b.add_argument("--train", type=int, nargs="+", required=True)
    b.add_argument("--K", type=int, default=32); b.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--pca_dir", required=True); r.add_argument("--mlp_dir", required=True)
    r.add_argument("--voxel_meta", required=True); r.add_argument("--name", default="")
    args = ap.parse_args()
    build(args) if args.cmd == "build" else report(args)
