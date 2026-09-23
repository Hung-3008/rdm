"""npy_multisubj layouts that give BraInCoRL the data of scripts/sflm_pilot.py, one directory per hold-out subject.

    python scripts/sflm_braincorl_layout.py --holdout 1 2 5 7 [--root data/braincorl_nsdgeneral]

Written to <root>/clip_pca512_sflm_h<H>/ (real files, no symlinks):
  image feature   <root>/clip_pca512_clsmean_all/*_img_feats_normed.npy  (multi-layer CLIP ViT-B/16 PCA-512, the feature of the
                  popeb_v2_pca512 / ridge_pca512 rows of the pilot; BraInCoRL takes one vector per image, so the 49 x 3840 patch
                  tokens of SFLM itself cannot be its input)
  betas           <root>/s{i}_betas_{unique,common}.npy
  train / val     the three training subjects' unique images split 8,400 / 600 as sflm_pilot.split_8400_600 (seed 0, one rng
                  consumed in the order of the training subjects)
  support order   test_time_icl_idx.json = default_rng(100).permutation(9000), so `src.eval --num_ic 30 100 300 1000 --repeats 3`
                  draws the support sets of the pilot
  gt              ridge weights of clip_pca512_clsmean_all (validation metric weight_cos_l2 only)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

TRAIN = {1: [2, 5, 7], 2: [1, 5, 7], 5: [1, 2, 7], 7: [1, 2, 5]}


def build(root: Path, H: int, n_val: int = 600, seed: int = 0, order_seed: int = 100):
    feat, out = root / "clip_pca512_clsmean_all", root / f"clip_pca512_sflm_h{H}"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for s in TRAIN[H]:
        img = np.load(feat / f"s{s}_unique_img_feats_normed.npy")
        beta = np.load(root / f"s{s}_betas_unique.npy", mmap_mode="r")
        perm = rng.permutation(len(img))
        va, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        for suffix, idx in (("", tr), ("_val", va)):
            np.save(out / f"s{s}_unique_img_feats_normed{suffix}.npy", img[idx])
            np.save(out / f"s{s}_unique_nrn_feats_msk{suffix}.npy", np.ascontiguousarray(beta[idx]))
            shutil.copyfile(feat / f"s{s}_gt.npy", out / f"s{s}_gt{suffix}.npy")
        print(f"[layout] h{H} s{s}: train {len(tr)}, val {len(va)}, voxels {beta.shape[1]}")
    shutil.copyfile(feat / f"s{H}_unique_img_feats_normed.npy", out / f"s{H}_unique_img_feats_normed.npy")
    shutil.copyfile(feat / "common_img_feats_normed.npy", out / "common_img_feats_normed.npy")
    shutil.copyfile(feat / f"s{H}_gt.npy", out / f"s{H}_gt.npy")
    shutil.copyfile(root / f"s{H}_betas_unique.npy", out / f"s{H}_unique_nrn_feats_msk.npy")
    shutil.copyfile(root / f"s{H}_betas_common.npy", out / f"s{H}_common_nrn_feats_msk.npy")
    order = np.random.default_rng(order_seed).permutation(9000)
    (out / "test_time_icl_idx.json").write_text(json.dumps({"large_shuffled": order.tolist()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/braincorl_nsdgeneral")
    ap.add_argument("--holdout", type=int, nargs="+", default=[1, 2, 5, 7])
    a = ap.parse_args()
    for H in a.holdout:
        build(Path(a.root), H)


if __name__ == "__main__":
    main()
