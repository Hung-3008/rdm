"""Spatial multi-layer CLIP ViT-B/16 features for the NSD images seen by the given subjects.

    python -m src.tools.extract_clip_grid --out data/braincorl_nsdgeneral

Writes (image order = sorted union of s{i}_nsdid_{unique,common}.npy of --subjects):
  feats_clip_b16_grid7_ids.npy   (N,)               nsd ids (row index into nsd_stimuli.hdf5 imgBrick)
  feats_clip_b16_grid7.npy       (N, 5, 49, 768)    float16; patch tokens of hidden_states[--layers], 14x14 grid
                                                    average-pooled 2x2 to 7x7, row-major (y, x)
Same preprocessing as src.tools.extract_clip_layers (CLIPImageProcessor, bf16 autocast).
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F_
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsd_root", default="data/NSD/data")
    ap.add_argument("--out", default="data/braincorl_nsdgeneral")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 3, 6, 9, 12])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    root = Path("checkpoints/backbones/openai--clip-vit-base-patch16")
    proc = CLIPImageProcessor.from_pretrained(root)
    model = CLIPVisionModelWithProjection.from_pretrained(root).to(a.device).eval()
    D = model.config.hidden_size
    out = Path(a.out)

    ids = np.unique(np.concatenate([np.load(out / f"s{s}_nsdid_{k}.npy") for s in a.subjects for k in ("unique", "common")]))
    np.save(out / "feats_clip_b16_grid7_ids.npy", ids)
    with h5py.File(Path(a.nsd_root) / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5") as f:
        brick = f["imgBrick"]
        grid = np.lib.format.open_memmap(out / "feats_clip_b16_grid7.npy", mode="w+", dtype=np.float16,
                                         shape=(len(ids), len(a.layers), 49, D))
        for i in tqdm(range(0, len(ids), a.batch_size), desc="[clip grid]"):
            imgs = [im for im in brick[ids[i:i + a.batch_size]]]
            with torch.inference_mode(), torch.autocast(a.device, dtype=torch.bfloat16):
                o = model(**proc(images=imgs, return_tensors="pt").to(a.device), output_hidden_states=True)
                hs = torch.stack([o.hidden_states[l] for l in a.layers], 1).float()[:, :, 1:]      # (B, L, 196, D)
            B, L = hs.shape[:2]
            hs = F_.avg_pool2d(hs.reshape(B * L, 14, 14, D).permute(0, 3, 1, 2), 2)               # (B*L, D, 7, 7)
            grid[i:i + B] = hs.permute(0, 2, 3, 1).reshape(B, L, 49, D).half().cpu().numpy()
        grid.flush()
    print(f"wrote {out / 'feats_clip_b16_grid7.npy'} {grid.shape}")


if __name__ == "__main__":
    main()
