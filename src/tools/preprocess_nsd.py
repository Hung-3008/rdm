"""Build the BraInCoRL `npy_multisubj` layout from raw NSD files.

    python -m src.tools.preprocess_nsd --subjects 1 2 5 7 --stage all
    python -m src.tools.preprocess_nsd --stage feats          # image features only

Stages (each writes to --out and can be re-run independently):
  mask    voxel set + per-voxel metadata          -> s{i}_voxel_meta.npz
  coords  MNI (mm) coordinate of every voxel      -> s{i}_coords_mni.npy
  betas   per-session z-scored betas, repeat-avg  -> s{i}_betas_{unique,common}.npy, s{i}_nsdid_{unique,common}.npy
  feats   backbone embeddings for all 73k images  -> feats_{backbone}_73k.npy
  gt      ridge weights on the 9000 unique images -> s{i}_gt.npy
  layout  files named as the original repo expects (see src/README.md)

Voxel set ("higher visual cortex"): streams ROI labels 2..7 (everything except
early visual) union the four floc ROIs. The paper reports ~20k voxels per
subject; streams>1 gives 21.7k (S1) / 20.6k (S2). Betas are z-scored per voxel
within each session, then averaged over the repeats of each image (paper Sec. 4.1).
"""

import argparse
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

FLOC = ["faces", "places", "bodies", "words"]
FLOC_TVAL = {"faces": "faces", "places": "places", "bodies": "bodies", "words": "word"}
BETAS_DIR = "betas_fithrf_GLMdenoise_RR"
N_SESSIONS, TRIALS_PER_SESSION = 40, 750


# ----------------------------------------------------------------------------- mask
def stage_mask(nsd: Path, out: Path, subj: int):
    s = f"subj{subj:02d}"
    roi = nsd / "nsddata/ppdata" / s / "func1pt8mm/roi"
    load = lambda p: np.asarray(nib.load(p).get_fdata(), dtype=np.float32)

    streams = load(roi / "streams.nii.gz")
    nsdgeneral = load(roi / "nsdgeneral.nii.gz") > 0
    floc_lbl = {c: load(roi / f"floc-{c}.nii.gz") for c in FLOC}
    floc_tval = {c: load(nsd / "nsddata/ppdata" / s / f"func1pt8mm/floc_{FLOC_TVAL[c]}tval.nii.gz") for c in FLOC}
    ncsnr = load(nsd / "nsddata_betas/ppdata" / s / "func1pt8mm" / BETAS_DIR / "ncsnr.nii.gz")

    mask = streams > 1
    for c in FLOC:
        mask |= floc_lbl[c] > 0
    idx = np.flatnonzero(mask)                       # flat index in nii (x, y, z) C-order
    ijk = np.stack(np.unravel_index(idx, mask.shape), 1)
    affine = nib.load(roi / "streams.nii.gz").affine
    xyz = (affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]   # scanner-space mm

    np.savez(out / f"s{subj}_voxel_meta.npz",
             idx=idx, ijk=ijk, xyz=xyz.astype(np.float32), shape=np.array(mask.shape),
             streams=streams.ravel()[idx].astype(np.int8), nsdgeneral=nsdgeneral.ravel()[idx],
             ncsnr=ncsnr.ravel()[idx],
             **{f"floc_{c}_label": floc_lbl[c].ravel()[idx].astype(np.int8) for c in FLOC},
             **{f"floc_{c}_tval": floc_tval[c].ravel()[idx] for c in FLOC})
    print(f"[mask] s{subj}: {len(idx)} voxels (streams>1: {(streams > 1).sum()}, "
          f"ncsnr>0.2 within mask: {(ncsnr.ravel()[idx] > 0.2).sum()})")


# --------------------------------------------------------------------------- coords
# MNI-to-func1pt8.nii.gz stores, for every func1pt8 voxel, its 1-based MNI-1mm index
# (9999 = undefined). Index -> mm conversion follows references/NRF common.py.
MNI_INDEX_TO_MM = np.array([[-1.0, 0.0, 0.0, 90.0], [0.0, 1.0, 0.0, -126.0],
                            [0.0, 0.0, 1.0, -72.0], [0.0, 0.0, 0.0, 1.0]], np.float32)


def stage_coords(nsd: Path, out: Path, subj: int):
    meta = np.load(out / f"s{subj}_voxel_meta.npz")
    t = np.asarray(nib.load(nsd / "nsddata/ppdata" / f"subj{subj:02d}/transforms/MNI-to-func1pt8.nii.gz").dataobj,
                   dtype=np.float32)                            # (x, y, z, 3)
    c = t.reshape(-1, 3)[meta["idx"]]
    bad = (c == 9999).any(1)
    c = (MNI_INDEX_TO_MM @ np.c_[c - 1.0, np.ones(len(c), np.float32)].T)[:3].T.astype(np.float32)
    if bad.any():
        c[bad] = np.nan
        raise RuntimeError(f"s{subj}: {bad.sum()} voxels in the mask have undefined MNI coordinates")
    np.save(out / f"s{subj}_coords_mni.npy", c)
    print(f"[coords] s{subj}: {c.shape}, MNI mm range x[{c[:,0].min():.0f},{c[:,0].max():.0f}] "
          f"y[{c[:,1].min():.0f},{c[:,1].max():.0f}] z[{c[:,2].min():.0f},{c[:,2].max():.0f}]")


# ---------------------------------------------------------------------------- betas
def stage_betas(nsd: Path, out: Path, subj: int):
    meta = np.load(out / f"s{subj}_voxel_meta.npz")
    shape = tuple(meta["shape"])                                # (x, y, z) as in nii
    mask = np.zeros(shape, bool); mask.ravel()[meta["idx"]] = True
    # hdf5 betas are stored (trial, z, y, x): transpose the mask and take the
    # matching flat indices, then reorder back to the nii voxel order.
    mask_h5 = mask.transpose(2, 1, 0)
    idx_h5 = np.flatnonzero(mask_h5)
    order = np.argsort(np.ravel_multi_index(np.unravel_index(idx_h5, mask_h5.shape)[::-1], shape))

    exp = sio.loadmat(nsd / "nsddata/experiments/nsd/nsd_expdesign.mat")
    master = exp["masterordering"].ravel() - 1                  # (30000,) index into subjectim
    subjectim = exp["subjectim"][subj - 1] - 1                  # (10000,) nsdId (0-based)
    sharedix = exp["sharedix"].ravel() - 1                      # (1000,)  nsdId (0-based)

    V = len(idx_h5)
    sums = np.zeros((10000, V), np.float64)
    counts = np.zeros(10000, np.int32)
    bdir = nsd / "nsddata_betas/ppdata" / f"subj{subj:02d}/func1pt8mm" / BETAS_DIR
    # subjects 3/6 have 32 sessions and 4/8 have 30 (others 40): use the sessions that exist
    n_sess = sum((bdir / f"betas_session{k:02d}.hdf5").exists() or (bdir / f"betas_session{k:02d}.nii.gz").exists()
                 for k in range(1, N_SESSIONS + 1))
    for sess in tqdm(range(1, n_sess + 1), desc=f"[betas] s{subj} ({n_sess} sessions)"):
        h5 = bdir / f"betas_session{sess:02d}.hdf5"
        if h5.exists():
            with h5py.File(h5) as f:
                b = f["betas"][:]                               # (750, z, y, x) int16
            b = b.reshape(b.shape[0], -1)[:, idx_h5][:, order]
        else:                                                   # nii.gz fallback: (x, y, z, 750) int16
            b = np.asarray(nib.load(bdir / f"betas_session{sess:02d}.nii.gz").dataobj)
            b = b.reshape(-1, b.shape[-1])[meta["idx"]].T
        b = b.astype(np.float32) / 300.0
        b = (b - b.mean(0)) / (b.std(0) + 1e-8)                 # z-score per voxel within session
        trials = master[(sess - 1) * TRIALS_PER_SESSION: sess * TRIALS_PER_SESSION]
        np.add.at(sums, trials, b)
        np.add.at(counts, trials, 1)
    shown = counts > 0
    if not shown.all():
        # subjects with < 40 sessions: unique images never shown are dropped; common images never shown
        # are kept as NaN rows so that the 1,000-image common order stays identical across subjects
        print(f"[betas] s{subj}: {(~shown).sum()} of 10000 images never shown ({n_sess} sessions); "
              f"dropping them from the unique set, NaN in the common set")
    betas = np.full((10000, V), np.nan, np.float32)
    betas[shown] = (sums[shown] / counts[shown, None]).astype(np.float32)   # (10000, V) in subjectim order

    is_shared = np.isin(subjectim, sharedix)
    uniq = np.flatnonzero(~is_shared & shown)
    # common set in sharedix order so that one feature file serves every subject
    common = np.array([np.flatnonzero(subjectim == n)[0] for n in sharedix])
    np.save(out / f"s{subj}_betas_unique.npy", betas[uniq])
    np.save(out / f"s{subj}_betas_common.npy", betas[common])
    np.save(out / f"s{subj}_nsdid_unique.npy", subjectim[uniq])
    np.save(out / f"s{subj}_nsdid_common.npy", subjectim[common])
    print(f"[betas] s{subj}: unique {betas[uniq].shape}, common {betas[common].shape}, "
          f"repeats min/max {counts.min()}/{counts.max()}")


# ---------------------------------------------------------------------------- feats
def stage_feats(nsd: Path, out: Path, backbone: str, batch_size: int, device: str):
    from transformers import (AutoImageProcessor, AutoModel, CLIPImageProcessor,
                              CLIPVisionModelWithProjection, SiglipImageProcessor, SiglipVisionModel)
    root = Path("checkpoints/backbones")
    if backbone == "clip_b16":
        proc = CLIPImageProcessor.from_pretrained(root / "openai--clip-vit-base-patch16")
        model = CLIPVisionModelWithProjection.from_pretrained(root / "openai--clip-vit-base-patch16")
        embed = lambda o: o.image_embeds
    elif backbone == "dinov2_base":
        proc = AutoImageProcessor.from_pretrained(root / "facebook--dinov2-base")
        model = AutoModel.from_pretrained(root / "facebook--dinov2-base")
        embed = lambda o: o.pooler_output
    elif backbone == "siglip_so400m":
        proc = SiglipImageProcessor.from_pretrained(root / "google--siglip-so400m-patch14-384")
        model = SiglipVisionModel.from_pretrained(root / "google--siglip-so400m-patch14-384")
        embed = lambda o: o.pooler_output
    else:
        raise ValueError(backbone)
    model = model.to(device).eval()

    path = out / f"feats_{backbone}_73k.npy"
    feats = None
    with h5py.File(nsd / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5") as f:
        brick = f["imgBrick"]
        n = brick.shape[0]
        for i in tqdm(range(0, n, batch_size), desc=f"[feats] {backbone}"):
            imgs = [im for im in brick[i:i + batch_size]]
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                e = embed(model(**proc(images=imgs, return_tensors="pt").to(device))).float().cpu().numpy()
            if feats is None:
                feats = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, e.shape[1]))
            feats[i:i + len(imgs)] = e
    feats.flush()
    print(f"[feats] wrote {path} {feats.shape}")


# ------------------------------------------------------------------------------- gt
def stage_gt(out: Path, subj: int, backbone: str, device: str):
    from ..utils import ridge_cv_brainwise, unit_norm
    feats = np.load(out / f"feats_{backbone}_73k.npy", mmap_mode="r")
    X = torch.from_numpy(unit_norm(np.asarray(feats[np.load(out / f"s{subj}_nsdid_unique.npy")]))).to(device)
    y = torch.from_numpy(np.load(out / f"s{subj}_betas_unique.npy").T.copy()).to(device)   # (V, n_unique)
    gt = []
    for i in tqdm(range(0, y.shape[0], 4096), desc=f"[gt] s{subj}"):
        gt.append(ridge_cv_brainwise(X, y[i:i + 4096]).cpu().numpy())
    gt = np.concatenate(gt).astype(np.float32)
    np.save(out / f"s{subj}_gt.npy", gt)
    # sanity: EV of gt on the shared images
    Xc = unit_norm(np.asarray(feats[np.load(out / f"s{subj}_nsdid_common.npy")]))
    pred = np.c_[Xc, np.ones(len(Xc), np.float32)] @ gt.T
    yc = np.load(out / f"s{subj}_betas_common.npy")
    ok = ~np.isnan(yc[:, 0])                                     # common images actually shown to this subject
    ev = 1 - np.var(yc[ok] - pred[ok], 0) / np.var(yc[ok], 0)
    print(f"[gt] s{subj}: {gt.shape}, EV on common images: mean {ev.mean():.4f}, "
          f"mean over ncsnr>0.2 {ev[np.load(out / f's{subj}_voxel_meta.npz')['ncsnr'] > 0.2].mean():.4f}")


# --------------------------------------------------------------------------- layout
def stage_layout(out: Path, subjects, backbone: str, seed: int):
    from ..utils import unit_norm
    feats = np.load(out / f"feats_{backbone}_73k.npy", mmap_mode="r")
    lay = out / backbone
    lay.mkdir(exist_ok=True)
    common_ids = None
    for s in subjects:
        ids_u = np.load(out / f"s{s}_nsdid_unique.npy")
        ids_c = np.load(out / f"s{s}_nsdid_common.npy")
        if common_ids is None:
            common_ids = ids_c
        elif not np.array_equal(common_ids, ids_c):
            raise RuntimeError("common image order differs between subjects")
        fu = unit_norm(np.asarray(feats[ids_u])).astype(np.float32)
        np.save(lay / f"s{s}_unique_img_feats_normed.npy", fu)
        np.save(lay / f"s{s}_unique_nrn_feats_msk.npy", np.load(out / f"s{s}_betas_unique.npy"))
        np.save(lay / f"s{s}_gt.npy", np.load(out / f"s{s}_gt.npy"))
        np.save(lay / f"s{s}_common_nrn_feats_msk.npy", np.load(out / f"s{s}_betas_common.npy"))
        # validation split for training-time monitoring: the subject's shared images
        np.save(lay / f"s{s}_unique_img_feats_normed_val.npy", unit_norm(np.asarray(feats[ids_c])).astype(np.float32))
        np.save(lay / f"s{s}_unique_nrn_feats_msk_val.npy", np.load(out / f"s{s}_betas_common.npy"))
        np.save(lay / f"s{s}_gt_val.npy", np.load(out / f"s{s}_gt.npy"))
        if (out / f"s{s}_coords_mni.npy").exists():
            np.save(lay / f"s{s}_coords.npy", np.load(out / f"s{s}_coords_mni.npy"))
            np.save(lay / f"s{s}_coords_val.npy", np.load(out / f"s{s}_coords_mni.npy"))
    np.save(lay / "common_img_feats_normed.npy", unit_norm(np.asarray(feats[common_ids])).astype(np.float32))
    # support ordering over the 9,000 unique images of the 40-session subjects; never overwritten once
    # created (subjects with fewer unique images are population-only, not evaluation subjects)
    if not (lay / "test_time_icl_idx.json").exists():
        order = np.random.default_rng(seed).permutation(9000).tolist()
        with open(lay / "test_time_icl_idx.json", "w") as f:
            json.dump({"large_shuffled": order}, f)
    print(f"[layout] wrote {lay}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsd_root", default="data/NSD/data")
    ap.add_argument("--out", default="data/braincorl")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--stage", default="all", choices=["all", "mask", "coords", "betas", "feats", "gt", "layout"])
    ap.add_argument("--backbone", default="clip_b16")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    nsd, out = Path(a.nsd_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = lambda st: a.stage in ("all", st)
    if run("mask"):
        for s in a.subjects: stage_mask(nsd, out, s)
    if run("coords"):
        for s in a.subjects: stage_coords(nsd, out, s)
    if run("betas"):
        for s in a.subjects: stage_betas(nsd, out, s)
    if run("feats"):
        stage_feats(nsd, out, a.backbone, a.batch_size, a.device)
    if run("gt"):
        for s in a.subjects: stage_gt(out, s, a.backbone, a.device)
    if run("layout"):
        stage_layout(out, a.subjects, a.backbone, a.seed)


if __name__ == "__main__":
    main()
