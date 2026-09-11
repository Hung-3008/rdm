"""Build fsaverage-vertex -> func1pt8-voxel correspondences from NSD transform files.
For each subject and each surface layer (white, layerB1..B3, pial), each fsaverage vertex is mapped to
its subject vertex (white-to-fsaverage, 1-based) and then to func1pt8 voxel coordinates
(func1pt8-to-<layer>, 1-based MATLAB convention, rounded to nearest voxel), then to the row of that voxel in
s{i}_voxel_meta.npz['idx'] (or -1 if outside the analysis mask).
Output: data/nsd_surf/vox2fsavg.npz with vox2fsavg_s{i} of shape (n_layers * 2 * 163842,).
"""
import numpy as np, nibabel as nib
from pathlib import Path

LAYERS = ["white", "layerB1", "layerB2", "layerB3", "pial"]
out = {}
for s in [1, 2, 5, 7]:
    meta = np.load(f"data/braincorl/s{s}_voxel_meta.npz"); shape = tuple(meta["shape"]); idx = meta["idx"]
    lut = -np.ones(int(np.prod(shape)), np.int64); lut[idx] = np.arange(len(idx))
    per_layer = []
    for layer in LAYERS:
        rows = []
        for hemi in ["lh", "rh"]:
            d = Path(f"data/nsd_surf/subj{s:02d}")
            w2f = np.asarray(nib.load(d / f"{hemi}.white-to-fsaverage.mgz").dataobj).reshape(-1).astype(np.int64) - 1
            f2s = np.asarray(nib.load(d / f"{hemi}.func1pt8-to-{layer}.mgz").dataobj).reshape(-1, 3)
            ijk = np.rint(f2s[w2f] - 1).astype(np.int64)                       # (163842, 3), 0-based
            ok = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
            flat = np.ravel_multi_index(ijk[ok].T, shape)
            r = -np.ones(len(w2f), np.int64); r[ok] = lut[flat]
            rows.append(r)
        per_layer.append(np.concatenate(rows))
    arr = np.concatenate(per_layer)
    out[f"vox2fsavg_s{s}"] = arr
    covered = np.zeros(len(idx), bool); covered[arr[arr >= 0]] = True
    good = meta["ncsnr"] > 0.2
    print(f"s{s}: vertices in mask {np.mean(arr >= 0)*100:.1f}%  mask voxels covered {covered.mean()*100:.1f}%  "
          f"ncsnr>0.2 voxels covered {covered[good].mean()*100:.1f}%")
np.savez("data/nsd_surf/vox2fsavg.npz", **out)
# sanity: MNI distance between S1 voxels and their S2 correspondences vs random
c1 = np.load("data/braincorl/clip_b16/s1_coords.npy"); c2 = np.load("data/braincorl/clip_b16/s2_coords.npy")
a, b = out["vox2fsavg_s1"], out["vox2fsavg_s2"]; ok = (a >= 0) & (b >= 0)
d = np.linalg.norm(c1[a[ok]] - c2[b[ok]], axis=1)
print(f"S1<->S2 vertex correspondences: {ok.sum()} pairs, MNI distance median {np.median(d):.1f} mm, 90% {np.percentile(d, 90):.1f} mm")
