"""Build a uint8 image array (N, S, S, 3) + labels from the ImageNet-1K parquet files, restricted to the classes of
ImageNet-100 (class names read from the imagenet100 parquet metadata) or to all classes.

    python scripts/foveacl_prep_imagenet.py --out data/images/foveacl/in100_256 --size 256

Each image: resize so that the short side is S (antialiased), center crop S x S. Output files:
    {train,val}_images.u8   raw uint8, shape (N, S, S, 3)
    {train,val}_labels.npy  int16, class index 0..C-1 in the order of the ImageNet-100 class list
    meta.json               shapes and class names
"""

import argparse
import glob
import io
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("data/images")


def class_names(parquet):
    md = json.loads(pq.ParquetFile(parquet).schema_arrow.metadata[b"huggingface"])
    return md["info"]["features"]["label"]["names"]


def load_one(args):
    path, keep, size, shard = args
    table = pq.read_table(path, filters=[("label", "in", list(keep))])
    imgs = np.empty((table.num_rows, size, size, 3), np.uint8)
    labels = np.empty(table.num_rows, np.int16)
    for i, row in enumerate(table.to_pylist()):
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - size) // 2, (h - size) // 2
        imgs[i] = np.asarray(im.crop((l, t, l + size, t + size)))
        labels[i] = keep[row["label"]]
    np.save(shard, imgs)
    np.save(str(shard).replace(".npy", "_y.npy"), labels)
    return table.num_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--all_classes", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    out = Path(a.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)

    names1k = class_names(sorted(glob.glob(str(ROOT / "imagenet1k/data/train-*.parquet")))[0])
    names = names1k if a.all_classes else class_names(sorted(glob.glob(str(ROOT / "imagenet100/data/train-*.parquet")))[0])
    keep = {names1k.index(n): i for i, n in enumerate(names)}
    meta = {"size": a.size, "classes": names}

    for split, pattern in (("val", "validation-*.parquet"), ("train", "train-*.parquet")):
        files = sorted(glob.glob(str(ROOT / "imagenet1k/data" / pattern)))
        jobs = [(f, keep, a.size, out / "shards" / f"{split}_{i:04d}.npy") for i, f in enumerate(files)]
        with Pool(a.workers) as pool:
            counts = pool.map(load_one, jobs, chunksize=1)
        n = sum(counts)
        mm = np.memmap(out / f"{split}_images.u8", np.uint8, "w+", shape=(n, a.size, a.size, 3))
        ys, o = [], 0
        for (_, _, _, shard), c in zip(jobs, counts):
            mm[o:o + c] = np.load(shard)
            ys.append(np.load(str(shard).replace(".npy", "_y.npy")))
            o += c
            Path(shard).unlink()
            Path(str(shard).replace(".npy", "_y.npy")).unlink()
        mm.flush()
        np.save(out / f"{split}_labels.npy", np.concatenate(ys))
        meta[f"n_{split}"] = n
        print(split, n, flush=True)
    (out / "shards").rmdir()
    (out / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
