"""Linear ceiling of FoveaCL features on NSD, same ridge and ROI groups as scripts/clip_layers_ceiling.py.

    PYTHONPATH=. python scripts/foveacl_ceiling.py --runs outputs/foveacl/R0_s0 outputs/foveacl/R2_s0

Per run: blocks (stem, layer1..4) are standardised per dimension, unit-normed per block, concatenated / sqrt(#blocks) and
reduced to PCA-512 (fitted on 20,000 of the cached NSD images, stimulus-only), as the CLIP layout pca512_clsmean_all.
Two scores per subject:
  common : ridge on the 9,000 unique images, EV on the 1,000 common images (reporting)
  valuniq: ridge on 8,400 unique images, EV on the other 600 (seed-0 split); this is the score used to select settings
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.clip_layers_ceiling import groups, kernel_ridge_cv_predict
from src.utils import explained_variance, unit_norm

ROOT = Path("data/braincorl_nsdgeneral")


def pca_features(path, k=512, n_fit=20000):
    x = np.load(path, mmap_mode="r")
    dims = json.loads((path.parent / "nsd_feats.json").read_text())["block_dims"]
    out, o = [], 0
    for d in dims:
        b = torch.from_numpy(np.asarray(x[:, o:o + d], np.float32)).cuda()
        sd = b.std(0)
        b = (b - b.mean(0)) / torch.maximum(sd, 0.01 * sd.mean())
        out.append(torch.nn.functional.normalize(b, dim=1))
        o += d
    Z = torch.cat(out, 1) / len(out) ** 0.5
    fit = np.sort(np.random.default_rng(0).choice(len(Z), n_fit, replace=False))
    mu = Z[fit].mean(0, keepdim=True)
    Zf = Z[fit] - mu
    e, U = torch.linalg.eigh(Zf.T @ Zf)
    comps = U[:, torch.argsort(e, descending=True)[:k]]
    return unit_norm(((Z - mu) @ comps).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--cache", default="data/images/foveacl/nsd425")
    ap.add_argument("--feats", default="nsd_feats.npy")
    ap.add_argument("--out", default="outputs/foveacl/ceiling.csv")
    ap.add_argument("--save_voxel_ev", action="store_true", help="write <run>/ev_common_s<subject>.npy")
    a = ap.parse_args()
    dev = torch.device("cuda")
    cache_ids = np.load(Path(a.cache) / "ids.npy")
    rows = []
    for run in map(Path, a.runs):
        F = pca_features(run / a.feats)
        row_of = lambda ids: F[np.searchsorted(cache_ids, ids)]
        for s in a.subjects:
            ids_u, ids_c = np.load(ROOT / f"s{s}_nsdid_unique.npy"), np.load(ROOT / f"s{s}_nsdid_common.npy")
            yu = torch.from_numpy(np.load(ROOT / f"s{s}_betas_unique.npy")).to(dev)
            yc = np.load(ROOT / f"s{s}_betas_common.npy")
            ok = ~np.isnan(yc[:, 0])
            g = groups(np.load(ROOT / f"s{s}_voxel_meta.npz"))
            Xu, Xc = torch.from_numpy(row_of(ids_u)).to(dev), torch.from_numpy(row_of(ids_c)).to(dev)
            pred, _ = kernel_ridge_cv_predict(Xu, yu, Xc)
            ev = explained_variance(pred[ok].T, torch.from_numpy(yc[ok]).to(dev).T).cpu().numpy()
            if a.save_voxel_ev:
                np.save(run / f"ev_common_s{s}.npy", ev)
            perm = torch.from_numpy(np.random.default_rng(0).permutation(len(ids_u))).to(dev)
            tr, va = perm[:-600], perm[-600:]
            pred_v, _ = kernel_ridge_cv_predict(Xu[tr], yu[tr], Xu[va])
            ev_v = explained_variance(pred_v.T, yu[va].T).cpu().numpy()
            row = dict(run=run.name, feats=a.feats, subject=s, **{f"common_{k}": float(ev[m].mean()) for k, m in g.items()},
                       **{f"valuniq_{k}": float(ev_v[m].mean()) for k, m in g.items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
            del pred, pred_v; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    out = Path(a.out)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df]).drop_duplicates(["run", "feats", "subject"], keep="last")
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
