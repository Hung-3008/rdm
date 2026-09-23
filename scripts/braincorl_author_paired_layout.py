"""npy_multisubj eval-only layout for the authors' BraInCoRL checkpoints on nsdgeneral with the support sets of
scripts/sflm_pilot.py (default_rng(100).permutation(9000), consecutive slices over n and rep).

    python scripts/braincorl_author_paired_layout.py [--root data/braincorl_nsdgeneral]

Written to <root>/clip_b16_paired/: unit-normed CLIP ViT-B/16 image_embeds (feats_clip_b16_73k.npy, the feature of the released
checkpoints), betas symlinked to <root>/s{i}_betas_{unique,common}.npy, gt = ridge weights of that feature (weight_cos_l2 metric
only), test_time_icl_idx.json. No train / val files: the layout serves `src.eval --original_ckpt` only.
"""
import argparse
import sys
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from src.utils import ridge_cv_brainwise, unit_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/braincorl_nsdgeneral")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.root) / "clip_b16_paired"
    out.mkdir(exist_ok=True)
    feats = np.load(root / "feats_clip_b16_73k.npy", mmap_mode="r")
    dev = torch.device("cuda")
    for s in a.subjects:
        ids = np.load(root / f"s{s}_nsdid_unique.npy")
        X = unit_norm(np.asarray(feats[ids]).astype(np.float32))
        np.save(out / f"s{s}_unique_img_feats_normed.npy", X)
        for kind in ("unique", "common"):
            link = out / f"s{s}_{kind}_nrn_feats_msk.npy"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.relpath(root / f"s{s}_betas_{kind}.npy", out), link)
        y = torch.from_numpy(np.load(root / f"s{s}_betas_unique.npy").T.copy()).to(dev)
        Xt = torch.from_numpy(X).to(dev)
        gt = np.concatenate([ridge_cv_brainwise(Xt, y[i:i + 4096]).cpu().numpy()
                             for i in tqdm(range(0, y.shape[0], 4096), desc=f"[gt] s{s}")]).astype(np.float32)
        np.save(out / f"s{s}_gt.npy", gt)
        print(f"[layout] s{s}: img {X.shape}, gt {gt.shape}")
    common = np.load(root / "s1_nsdid_common.npy")
    for s in a.subjects:
        assert np.array_equal(common, np.load(root / f"s{s}_nsdid_common.npy"))
    np.save(out / "common_img_feats_normed.npy", unit_norm(np.asarray(feats[common]).astype(np.float32)))
    order = np.random.default_rng(100).permutation(9000)
    (out / "test_time_icl_idx.json").write_text(json.dumps({"large_shuffled": order.tolist()}))
    print(f"[layout] wrote {out}")


if __name__ == "__main__":
    main()
