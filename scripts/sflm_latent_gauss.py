"""Option A slow stage + option B dictionaries (docs/sflm_eb_experiments_2026-09-23.md, part 3).

Trains the SFLM latent map with a continuous Gaussian receptive field per training voxel (src/sflm/latent.py GaussianReadout,
no mask dictionary), saves latent.pt, the population receptive fields (mu, sigma of the ncsnr > 0.2 voxels of the training
subjects) and k-means dictionaries of M Gaussian masks built from those receptive fields (features mu_y, mu_x, log sigma;
sklearn KMeans, n_init 10, seed 0).  The fast stage (scripts/sflm_fast.py --latent ... --dict dict_M<M>.pt) is unchanged.

    python scripts/sflm_latent_gauss.py --holdout 1 --out outputs/sflm_ab/h1 --M 35 70 140 280
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, ".")
from src.sflm.latent import DATA, DEV, GridStore, gaussian_dictionary, train_latent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--out", default="outputs/sflm_ab/h1")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--M", type=int, nargs="+", default=[35, 70, 140, 280])
    ap.add_argument("--skip_train", action="store_true", help="reuse <out>/latent.pt, only rebuild the dictionaries")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_subj = [s for s in (1, 2, 5, 7) if s != a.holdout]
    if not a.skip_train:
        torch.manual_seed(0)
        np.random.seed(0)
        _, _, _, info = train_latent(train_subj, GridStore(), None, a, out, readout="gauss")
        print(f"[gauss] best val_ev {info['best_val_ev']:.4f} at epoch {info['best_epoch']} in {info['train_sec']:.0f}s", flush=True)
    ck = torch.load(out / "latent.pt")
    mu, sig, subj = [], [], []
    for s in train_subj:
        good = np.load(DATA / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2
        m, g = ck["rf"][s]
        mu.append(m.numpy()[good])
        sig.append(g.numpy()[good])
        subj.append(np.full(int(good.sum()), s))
    mu, sig, subj = np.concatenate(mu), np.concatenate(sig), np.concatenate(subj)
    np.savez(out / "rf_pop.npz", mu=mu, sigma=sig, subject=subj)
    q = np.quantile(sig, [0.05, 0.25, 0.5, 0.75, 0.95])
    stats = dict(J=int(len(sig)), sigma_quantiles=q.tolist(), frac_sigma_ge_3=float((sig >= 3).mean()),
                 frac_sigma_le_1=float((sig <= 1).mean()), mu_mean=mu.mean(0).tolist(), mu_std=mu.std(0).tolist(),
                 frac_mu_outside_grid=float(((mu < 0) | (mu > 6)).any(1).mean()))
    print("[rf] " + json.dumps(stats), flush=True)
    X = np.column_stack([mu, np.log(sig)])
    for M in a.M:
        km = KMeans(n_clusters=M, n_init=10, random_state=0).fit(X)
        c = torch.from_numpy(km.cluster_centers_.astype(np.float32))
        A = gaussian_dictionary(c[:, :2], c[:, 2].exp())
        torch.save(dict(A=A, centres=c[:, :2], sigma=c[:, 2].exp(), inertia=float(km.inertia_), counts=np.bincount(km.labels_, minlength=M).tolist()),
                   out / f"dict_M{M}.pt")
        print(f"[dict] M={M} inertia {km.inertia_:.1f} sigma range {c[:, 2].exp().min():.2f}-{c[:, 2].exp().max():.2f} "
              f"min/max cluster size {np.bincount(km.labels_).min()}/{np.bincount(km.labels_).max()}", flush=True)
    stats["val_ev"] = None if a.skip_train else info["best_val_ev"]
    with open(out / "rf_stats.json", "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
