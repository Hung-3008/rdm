"""Features of a trained FoveaCL encoder for the NSD images, under the NSD viewing condition (one central fixation
covering the whole 8.4 deg image).

    python -m src.foveacl.extract_nsd --cache data/images/foveacl/nsd425 --build_cache        # once, needs h5py
    python -m src.foveacl.extract_nsd --cache data/images/foveacl/nsd425 --run outputs/foveacl/R2_s0

Cache: ids.npy (sorted 0-based nsdId of the images seen by S1/S2/S5/S7) + images.u8 (len(ids), 425, 425, 3).
Output: <run>/nsd_feats.npy, float16 (len(ids), D). Blocks = stem, layer1..layer4; each block is the feature map
average-pooled to a 4 x 4 grid plus its global mean, so a block with C channels has 17 * C dims. <run>/nsd_feats.json
holds the block boundaries. --untrained writes the features of the randomly initialised network (seed from the run).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.foveacl.train import Encoder
from src.foveacl.views import ViewMaker

STIM = "data/NSD/data/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
ROOT = "data/braincorl_nsdgeneral"


def build_cache(cache):
    import h5py
    cache.mkdir(parents=True, exist_ok=True)
    ids = np.unique(np.concatenate([np.load(f"{ROOT}/s{s}_nsdid_{t}.npy") for s in (1, 2, 5, 7) for t in ("unique", "common")]))
    np.save(cache / "ids.npy", ids)
    src = h5py.File(STIM, "r")["imgBrick"]
    mm = np.memmap(cache / "images.u8", np.uint8, "w+", shape=(len(ids), 425, 425, 3))
    for i in range(0, len(ids), 500):
        mm[i:i + 500] = src[ids[i:i + 500]]
    mm.flush()


@torch.no_grad()
def blocks(backbone, x):
    out = []
    x = backbone.relu(backbone.bn1(backbone.conv1(x))); out.append(x)
    x = backbone.maxpool(x)
    for layer in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):
        x = layer(x); out.append(x)
    return [torch.cat([F.adaptive_avg_pool2d(o, 4).flatten(1), o.mean((2, 3))], 1) for o in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--run")
    p.add_argument("--build_cache", action="store_true")
    p.add_argument("--untrained", action="store_true")
    p.add_argument("--batch", type=int, default=250)
    a = p.parse_args()
    cache = Path(a.cache)
    if a.build_cache:
        return build_cache(cache)

    run = Path(a.run)
    args = json.loads((run / "args.json").read_text())
    dev = torch.device("cuda")
    torch.manual_seed(args["seed"])
    model = Encoder().to(dev).eval()
    if not a.untrained:
        model.load_state_dict(torch.load(run / "last.pt", map_location=dev)["model"])
    model = model.to(memory_format=torch.channels_last)
    views = ViewMaker(args["n"], args["mode"], args["a_rel"])
    ids = np.load(cache / "ids.npy")
    mm = np.memmap(cache / "images.u8", np.uint8, "r", shape=(len(ids), 425, 425, 3))
    feats, dims = None, None
    for i in range(0, len(ids), a.batch):
        img = torch.from_numpy(mm[i:i + a.batch]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bl = blocks(model.backbone, views.test_view(img))
        if feats is None:
            dims = [b.shape[1] for b in bl]
            feats = np.lib.format.open_memmap(run / ("nsd_feats_untrained.npy" if a.untrained else "nsd_feats.npy"), "w+", np.float16, (len(ids), sum(dims)))
        feats[i:i + a.batch] = torch.cat(bl, 1).float().cpu().numpy().astype(np.float16)
    feats.flush()
    (run / "nsd_feats.json").write_text(json.dumps(dict(block_dims=dims, names=["stem", "layer1", "layer2", "layer3", "layer4"])))
    print(run, feats.shape)


if __name__ == "__main__":
    main()
