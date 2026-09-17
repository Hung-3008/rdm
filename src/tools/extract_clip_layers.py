"""Multi-layer CLIP ViT-B/16 features for the 73k NSD images.

    python -m src.tools.extract_clip_layers --out data/braincorl_nsdgeneral

Writes (float32 memmaps, image order = nsd_stimuli.hdf5 imgBrick):
  feats_clip_b16_cls_73k.npy    (73000, 13, 768)  CLS token of hidden_states[0..12] (0 = pre-transformer embedding)
  feats_clip_b16_mean_73k.npy   (73000, 13, 768)  mean over the 196 patch tokens of the same hidden states
  feats_clip_b16_73k.npy        (73000, 512)      image_embeds (post_layernorm + visual_projection), the pooled
                                                  feature every earlier experiment used; rewritten only if missing
                                                  or a broken symlink.
Same preprocessing as src.tools.preprocess_nsd stage_feats (CLIPImageProcessor, bf16 autocast).
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsd_root", default="data/NSD/data")
    ap.add_argument("--out", default="data/braincorl_nsdgeneral")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    root = Path("checkpoints/backbones/openai--clip-vit-base-patch16")
    proc = CLIPImageProcessor.from_pretrained(root)
    model = CLIPVisionModelWithProjection.from_pretrained(root).to(a.device).eval()
    L = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    out = Path(a.out)

    p_cls, p_mean, p_emb = out / "feats_clip_b16_cls_73k.npy", out / "feats_clip_b16_mean_73k.npy", out / "feats_clip_b16_73k.npy"
    write_emb = not p_emb.exists()            # exists() follows symlinks: False for a broken link
    if write_emb and p_emb.is_symlink():
        p_emb.unlink()
    with h5py.File(Path(a.nsd_root) / "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5") as f:
        brick = f["imgBrick"]
        n = brick.shape[0]
        cls = np.lib.format.open_memmap(p_cls, mode="w+", dtype=np.float32, shape=(n, L, D))
        mean = np.lib.format.open_memmap(p_mean, mode="w+", dtype=np.float32, shape=(n, L, D))
        emb = np.lib.format.open_memmap(p_emb, mode="w+", dtype=np.float32, shape=(n, model.config.projection_dim)) if write_emb else None
        for i in tqdm(range(0, n, a.batch_size), desc="[clip layers]"):
            imgs = [im for im in brick[i:i + a.batch_size]]
            with torch.inference_mode(), torch.autocast(a.device, dtype=torch.bfloat16):
                o = model(**proc(images=imgs, return_tensors="pt").to(a.device), output_hidden_states=True)
                hs = torch.stack(o.hidden_states, 1).float()          # (B, L, 197, D)
            cls[i:i + len(imgs)] = hs[:, :, 0].cpu().numpy()
            mean[i:i + len(imgs)] = hs[:, :, 1:].mean(2).cpu().numpy()
            if emb is not None:
                emb[i:i + len(imgs)] = o.image_embeds.float().cpu().numpy()
    cls.flush(); mean.flush()
    if emb is not None:
        emb.flush()
    print(f"wrote {p_cls} {cls.shape}, {p_mean} {mean.shape}" + (f", {p_emb} {emb.shape}" if emb is not None else ""))


if __name__ == "__main__":
    main()
