"""Few-shot feasibility of H3: joint Gaussian with a covariance estimated from the k context trials only.

    python scripts/calib_joint_fewshot.py --subjects 1 2 5 7

Mean model: 0 for every voxel (as `null_n01` in calib_baselines.py), so the comparison isolates the covariance.
  diag   N(0, diag(d)),                        d = context variance per voxel
  full   N(0, (1 - a) S_k + a diag(d)),        S_k = context covariance (rank <= k), a by 5-fold CV inside the context
Same marginals in both. Test: held-out trial (rep = 2) of the common images. Context: k distinct unique
images, one trial each, as in calib_baselines.py. Output: NLL in nats per voxel per trial.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_baselines import draw_context, load_subject  # noqa: E402

DEV = torch.device("cuda")
ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]


def nll(x, C):
    L = torch.linalg.cholesky(C)
    q = (torch.linalg.solve_triangular(L, x.T, upper=False) ** 2).sum(0).mean()
    return (0.5 * (2 * torch.log(torch.diagonal(L)).sum() + q) / x.shape[1] + 0.5 * math.log(2 * math.pi)).item()


def shrunk(x, a):
    S = x.T @ x / len(x)
    return (1 - a) * S + a * torch.diag(torch.diagonal(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/braincorl_nsdgeneral")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50, 100, 200, 500])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_vox", type=int, default=1500)
    ap.add_argument("--out", default="outputs/calib/joint_fewshot.json")
    a = ap.parse_args()
    rows = []
    for s in a.subjects:
        trials, meta = load_subject(Path(a.root), s)
        vox = np.sort(np.random.default_rng(0).permutation(trials.shape[1])[:a.n_vox])
        test = np.flatnonzero(meta["is_common"] & (meta["rep"] == 2))
        Yt = torch.from_numpy(np.asarray(trials[test])[:, vox]).to(DEV, torch.float64)
        for k in a.k:
            for seed in range(a.seeds):
                idx = draw_context(meta, k, np.random.default_rng(1000 * k + seed))
                Yc = torch.from_numpy(np.asarray(trials[idx])[:, vox]).to(DEV, torch.float64)
                folds = np.array_split(np.random.default_rng(seed).permutation(k), 5)
                cv = [np.mean([nll(Yc[f], shrunk(Yc[np.setdiff1d(np.arange(k), f)], al)) for f in folds]) for al in ALPHAS]
                al = ALPHAS[int(np.argmin(cv))]
                d = torch.diag(torch.diagonal(shrunk(Yc, 1.0)))
                row = dict(subject=s, k=k, seed=seed, alpha=al, nll_diag=nll(Yt, d), nll_full=nll(Yt, shrunk(Yc, al)),
                           nll_n01=nll(Yt, torch.eye(len(vox), device=DEV, dtype=torch.float64)))
                rows.append(row)
                print(f"[joint-fewshot] S{s} k={k} seed={seed} alpha={al} n01={row['nll_n01']:.4f} "
                      f"diag={row['nll_diag']:.4f} full={row['nll_full']:.4f}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("\n| k | N(0,1) | diag | full | full - N(0,1) [min, max over subjects] | alpha |\n| --- | --- | --- | --- | --- | --- |")
    for k in a.k:
        r = [x for x in rows if x["k"] == k]
        m = lambda key: np.mean([x[key] for x in r])
        per = [np.mean([x["nll_full"] - x["nll_n01"] for x in r if x["subject"] == s]) for s in a.subjects]
        print(f"| {k} | {m('nll_n01'):.4f} | {m('nll_diag'):.4f} | {m('nll_full'):.4f} | "
              f"{np.mean(per):+.4f} [{min(per):+.4f}, {max(per):+.4f}] | {sorted(set(x['alpha'] for x in r))} |")


if __name__ == "__main__":
    main()
