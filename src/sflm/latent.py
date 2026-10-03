"""Shared functional latent map (SFLM), slow stage: the latent map over the 7x7 CLIP patch grid, the mask dictionary
and the per-subject source readout, trained on the training subjects' unique images (docs/proposal-shared-functional-latent
-2026-09-20.md; pilot in scripts/sflm_pilot.py).

    latent map      z_p(x) = W2 SiLU(W1 phi_p(x) + b1) in R^K for each of the 49 cells of the 7x7 CLIP grid
    source readout  beta_v(x) = c_v' sum_m softmax(l_v)_m u_m(x) + b_v,   u_m(x) = sum_p A[m, p] z_p(x)
                    A = dictionary of M isotropic Gaussian masks over the grid (+ one uniform mask)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_

from src.utils.metrics import explained_variance

DATA = Path("data/braincorl_nsdgeneral")
DEV = torch.device("cuda")
STREAMS = {"early": [1], "mid": [2, 3, 4], "ventral": [5], "lateral": [6], "parietal": [7]}


def mask_dictionary(side: int = 7) -> torch.Tensor:
    """(M, side*side) rows summing to 1: Gaussians (sigma in cells) on nested centre grids + uniform."""
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    rows = []
    for sigma, centres in [(0.7, np.arange(side)), (1.4, np.arange(0, side, 2)), (2.5, np.array([1.5, 4.5]))]:
        for cy in centres:
            for cx in centres:
                g = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / sigma ** 2).ravel()
                rows.append(g / g.sum())
    rows.append(np.full(side * side, 1.0 / side ** 2))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


class GridStore:
    """Rows of feats_clip_b16_grid7.npy by nsd id, as (N, 49, L*768) float16 on the GPU."""

    def __init__(self):
        self.ids = np.load(DATA / "feats_clip_b16_grid7_ids.npy")
        self.grid = np.load(DATA / "feats_clip_b16_grid7.npy", mmap_mode="r")

    def __call__(self, nsdid: np.ndarray) -> torch.Tensor:
        rows = np.searchsorted(self.ids, nsdid)
        assert (self.ids[rows] == nsdid).all()
        order = np.argsort(rows)
        x = np.empty((len(rows),) + self.grid.shape[1:], np.float16)
        x[order] = self.grid[rows[order]]
        x = torch.from_numpy(x).to(DEV)                                  # (N, L, 49, D)
        return x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)


class LatentMap(nn.Module):
    def __init__(self, in_dim: int, hidden: int, K: int, mu: torch.Tensor, sd: torch.Tensor):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, K, bias=False)
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  # (B, 49, in_dim) float16 -> (B, 49, K)
        return self.lin2(F_.silu(self.lin1((x.float() - self.mu) / self.sd)))


class SourceReadout(nn.Module):
    def __init__(self, V: int, M: int, K: int):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(V, M))
        self.C = nn.Parameter(torch.randn(V, K) * 0.01)
        self.b = nn.Parameter(torch.zeros(V))

    def forward(self, U: torch.Tensor) -> torch.Tensor:                  # (B, M, K) -> (B, V)
        return (torch.einsum("bmk,vm->bvk", U, torch.softmax(self.logit, 1)) * self.C).sum(-1) + self.b


class GaussianReadout(nn.Module):
    """Continuous receptive field per voxel (option A): a_v(p) ∝ exp(-|p - mu_v|^2 / 2 sigma_v^2) over the side x side grid,
    beta_v(x) = c_v' sum_p a_v(p) z_p(x) + b_v.  mu_v (2, cells) starts at the grid centre, sigma_v (cells) at sigma0;
    sigma is clamped to [0.3, 6] cells (6 cells on a 7-cell grid is close to a uniform mask)."""

    def __init__(self, V: int, K: int, side: int = 7, sigma0: float = 2.0):
        super().__init__()
        yy, xx = torch.meshgrid(torch.arange(side, dtype=torch.float32), torch.arange(side, dtype=torch.float32), indexing="ij")
        self.register_buffer("grid", torch.stack([yy.ravel(), xx.ravel()], 1))       # (P, 2)
        self.mu = nn.Parameter(torch.full((V, 2), (side - 1) / 2.0))
        self.log_sigma = nn.Parameter(torch.full((V,), float(np.log(sigma0))))
        self.C = nn.Parameter(torch.randn(V, K) * 0.01)
        self.b = nn.Parameter(torch.zeros(V))
        self.side = side

    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp(0.3, 6.0)

    def masks(self) -> torch.Tensor:                                     # (V, P), rows sum to 1
        mu = self.mu.clamp(-1.0, self.side)
        d2 = ((self.grid[None] - mu[:, None]) ** 2).sum(-1)               # (V, P)
        return torch.softmax(-0.5 * d2 / self.sigma()[:, None] ** 2, 1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:                  # (B, P, K) -> (B, V)
        U = torch.einsum("bpk,vp->bvk", Z, self.masks())
        return (U * self.C).sum(-1) + self.b


def gaussian_dictionary(mu: torch.Tensor, sigma: torch.Tensor, side: int = 7) -> torch.Tensor:
    """(M, side*side) rows summing to 1: Gaussians at the given centres mu (M, 2) and widths sigma (M,) in cells."""
    yy, xx = torch.meshgrid(torch.arange(side, dtype=torch.float32), torch.arange(side, dtype=torch.float32), indexing="ij")
    grid = torch.stack([yy.ravel(), xx.ravel()], 1).to(mu.device)
    d2 = ((grid[None] - mu[:, None]) ** 2).sum(-1)
    return torch.softmax(-0.5 * d2 / sigma[:, None] ** 2, 1)


def split_8400_600(n_subj: int, seed: int = 0, n_img: int = 9000, n_val: int = 600):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_subj):
        perm = rng.permutation(n_img)
        out.append((np.sort(perm[n_val:]), np.sort(perm[:n_val])))
    return out


def train_latent(train_subj, store, A, a, out: Path, readout: str = "dict", data=None):
    """readout "dict": SourceReadout over the mask dictionary A (M, P); "gauss": GaussianReadout (continuous RF per voxel,
    A unused).  latent.pt stores the latent map; with "gauss" also rf = {subject: (mu (V, 2), sigma (V,))} at the best epoch.
    data: {subject: (grid features (N, 49, in_dim) float16 on DEV or on the CPU (batches are moved to DEV), betas (N, V) float32
    on DEV, (train idx, val idx))} in place of the NSD files and split_8400_600 (store unused).  An epoch has
    min_k(n_train_k) // batch steps."""
    if data is None:
        betas = [torch.from_numpy(np.load(DATA / f"s{s}_betas_unique.npy")).to(DEV) for s in train_subj]
        X = [store(np.load(DATA / f"s{s}_nsdid_unique.npy")) for s in train_subj]
        split = split_8400_600(len(train_subj))
    else:
        X, betas, split = ([data[s][i] for s in train_subj] for i in range(3))
    n_steps = min(len(tr) for tr, _ in split) // a.batch

    def grid_rows(k, idx):                                                 # rows idx of X[k] on DEV (X[k] may stay on the CPU)
        return X[k][idx.to(X[k].device) if torch.is_tensor(idx) else idx].to(DEV)
    samp = torch.cat([grid_rows(k, torch.randperm(x.shape[0], device=DEV)[:700]) for k, x in enumerate(X)]).float().reshape(-1, X[0].shape[-1])
    g = LatentMap(X[0].shape[-1], a.hidden, a.K, samp.mean(0), samp.std(0).clamp_min(1e-3)).to(DEV)
    if readout == "dict":
        heads = nn.ModuleList([SourceReadout(b.shape[1], A.shape[0], a.K) for b in betas]).to(DEV)
        slow = [h.logit for h in heads] + [h.b for h in heads]
    else:
        heads = nn.ModuleList([GaussianReadout(b.shape[1], a.K) for b in betas]).to(DEV)
        slow = [h.mu for h in heads] + [h.log_sigma for h in heads] + [h.b for h in heads]
    opt = torch.optim.AdamW([{"params": g.parameters(), "weight_decay": 0.0},
                             {"params": [h.C for h in heads], "weight_decay": 1e-4},
                             {"params": slow, "weight_decay": 0.0, "lr": 1e-2}],
                            lr=1e-3)

    def head(k, Z):
        return heads[k](torch.einsum("mp,bpk->bmk", A, Z)) if readout == "dict" else heads[k](Z)

    def val_ev():
        g.eval()
        with torch.no_grad():
            evs = []
            for k, (_, va) in enumerate(split):
                pred = torch.cat([head(k, g(grid_rows(k, va[i:i + 200]))) for i in range(0, len(va), 200)])
                evs.append(explained_variance(pred.T, betas[k][va].T).mean().item())
        g.train()
        return float(np.mean(evs)), evs

    best, best_ep, rows, t0 = -1e9, -1, [], time.time()
    for ep in range(a.max_epochs):
        perms = [torch.from_numpy(np.random.permutation(tr)).to(DEV) for tr, _ in split]
        tot = 0.0
        for i in range(n_steps):
            loss = 0.0
            for k in range(len(train_subj)):
                idx = perms[k][i * a.batch:(i + 1) * a.batch]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    Z = g(grid_rows(k, idx))
                loss = loss + torch.mean((head(k, Z.float()) - betas[k][idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        ev, evs = val_ev()
        rows.append(dict(epoch=ep, loss=tot / n_steps, val_ev=ev, sec=time.time() - t0))
        print(f"[latent] epoch {ep:3d} loss {rows[-1]['loss']:.4f} val_ev {ev:.4f} " + " ".join(f"{v:.4f}" for v in evs), flush=True)
        if ev > best:
            best, best_ep = ev, ep
            ck = dict(state=g.state_dict(), cfg=dict(in_dim=X[0].shape[-1], hidden=a.hidden, K=a.K), readout=readout)
            if readout == "gauss":
                ck["rf"] = {s: (h.mu.detach().clamp(-1.0, h.side).cpu(), h.sigma().detach().cpu()) for s, h in zip(train_subj, heads)}
            torch.save(ck, out / "latent.pt")
        elif ep - best_ep >= a.patience:
            break
    pd.DataFrame(rows).to_csv(out / "latent_log.csv", index=False)
    g.load_state_dict(torch.load(out / "latent.pt")["state"])
    g.eval()
    with torch.no_grad():                                                  # pooled channels of every training image
        U = [torch.cat([torch.einsum("mp,bpk->bmk", A, g(x[i:i + 500].to(DEV))) for i in range(0, x.shape[0], 500)]) for x in X] if A is not None else None
    del X
    torch.cuda.empty_cache()
    return g, U, betas, dict(best_val_ev=best, best_epoch=best_ep, train_sec=time.time() - t0)
