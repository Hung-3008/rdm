"""npy_multisubj layout whose validation split is the one PopEB-v2 uses, for a fair BraInCoRL comparison
(docs/research-log-2026-09.md section 8).

    python scripts/valuniq_layout.py --layout clip_pca512_clsmean_all --holdout 1 2 5 7 [--root data/braincorl_nsdgeneral]

For hold-out H the three training subjects keep `val_images` (500) of their 9,000 unique images as validation and
train on the remaining 8,500 — the same images as `src/train_lowrank.py:load_subjects` (one `np.random.default_rng(seed)`
consumed in the order of `train_subjects`), so BraInCoRL and PopEB-v2 see the same images and select on the same images.
The held-out subject's files (support) and the shared query images are symlinked unchanged, so evaluation is identical.
Written to <root>/<layout>_valuniq_h<H>/.
"""
import argparse
import os
from pathlib import Path

import numpy as np

TRAIN = {1: [2, 5, 7], 2: [1, 5, 7], 5: [1, 2, 7], 7: [1, 2, 5]}


def build(root: Path, layout: str, H: int, val_images: int, seed: int):
    src, out = root / layout, root / f"{layout}_valuniq_h{H}"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for s in TRAIN[H]:
        img = np.load(src / f"s{s}_unique_img_feats_normed.npy")
        beta = np.load(src / f"s{s}_unique_nrn_feats_msk.npy")
        perm = rng.permutation(len(img))                      # same draw order as load_subjects
        va, tr = np.sort(perm[:val_images]), np.sort(perm[val_images:])
        np.save(out / f"s{s}_unique_img_feats_normed.npy", img[tr])
        np.save(out / f"s{s}_unique_nrn_feats_msk.npy", beta[tr])
        np.save(out / f"s{s}_unique_img_feats_normed_val.npy", img[va])
        np.save(out / f"s{s}_unique_nrn_feats_msk_val.npy", beta[va])
        for f in [f"s{s}_gt.npy", f"s{s}_gt_val.npy", f"s{s}_coords.npy", f"s{s}_coords_val.npy"]:
            link(out, src, f)
        print(f"[valuniq] h{H} s{s}: train {len(tr)}, val {len(va)}")
    for f in [f"s{H}_unique_img_feats_normed.npy", f"s{H}_unique_nrn_feats_msk.npy", f"s{H}_common_nrn_feats_msk.npy",
              f"s{H}_gt.npy", f"s{H}_coords.npy", "common_img_feats_normed.npy", "test_time_icl_idx.json"]:
        link(out, src, f)
    (out / "README.txt").write_text(
        f"BraInCoRL layout for hold-out {H}: training subjects {TRAIN[H]} split {9000 - val_images} train / {val_images} val "
        f"unique images, exactly as src/train_lowrank.py (seed {seed}); gt and the held-out subject symlink to ../{layout}/\n")


def link(out: Path, src: Path, name: str):
    if not os.path.lexists(out / name):
        os.symlink(f"../{src.name}/{name}", out / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/braincorl_nsdgeneral")
    ap.add_argument("--layout", default="clip_pca512_clsmean_all")
    ap.add_argument("--holdout", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--val_images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for H in a.holdout:
        build(Path(a.root), a.layout, H, a.val_images, a.seed)


if __name__ == "__main__":
    main()
