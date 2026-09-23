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


def split_8400_600(n_subj: int, seed: int = 0, n_img: int = 9000, n_val: int = 600):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_subj):
        perm = rng.permutation(n_img)
        out.append((np.sort(perm[n_val:]), np.sort(perm[:n_val])))
    return out


def train_latent(train_subj, store, A, a, out: Path):
    betas = [torch.from_numpy(np.load(DATA / f"s{s}_betas_unique.npy")).to(DEV) for s in train_subj]
    X = [store(np.load(DATA / f"s{s}_nsdid_unique.npy")) for s in train_subj]
    split = split_8400_600(len(train_subj))
    samp = torch.cat([x[torch.randperm(x.shape[0], device=DEV)[:700]] for x in X]).float().reshape(-1, X[0].shape[-1])
    g = LatentMap(X[0].shape[-1], a.hidden, a.K, samp.mean(0), samp.std(0).clamp_min(1e-3)).to(DEV)
    heads = nn.ModuleList([SourceReadout(b.shape[1], A.shape[0], a.K) for b in betas]).to(DEV)
    opt = torch.optim.AdamW([{"params": g.parameters(), "weight_decay": 0.0},
                             {"params": [h.C for h in heads], "weight_decay": 1e-4},
                             {"params": [h.logit for h in heads] + [h.b for h in heads], "weight_decay": 0.0, "lr": 1e-2}],
                            lr=1e-3)

    def val_ev():
        g.eval()
        with torch.no_grad():
            evs = []
            for k, (_, va) in enumerate(split):
                pred = torch.cat([heads[k](torch.einsum("mp,bpk->bmk", A, g(X[k][va[i:i + 200]]))) for i in range(0, len(va), 200)])
                evs.append(explained_variance(pred.T, betas[k][va].T).mean().item())
        g.train()
        return float(np.mean(evs)), evs

    best, best_ep, rows, t0 = -1e9, -1, [], time.time()
    for ep in range(a.max_epochs):
        perms = [torch.from_numpy(np.random.permutation(tr)).to(DEV) for tr, _ in split]
        tot = 0.0
        for i in range(len(perms[0]) // a.batch):
            loss = 0.0
            for k in range(len(train_subj)):
                idx = perms[k][i * a.batch:(i + 1) * a.batch]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    U = torch.einsum("mp,bpk->bmk", A, g(X[k][idx]))
                loss = loss + torch.mean((heads[k](U.float()) - betas[k][idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        ev, evs = val_ev()
        rows.append(dict(epoch=ep, loss=tot / (len(perms[0]) // a.batch), val_ev=ev, sec=time.time() - t0))
        print(f"[latent] epoch {ep:3d} loss {rows[-1]['loss']:.4f} val_ev {ev:.4f} " + " ".join(f"{v:.4f}" for v in evs), flush=True)
        if ev > best:
            best, best_ep = ev, ep
            torch.save(dict(state=g.state_dict(), cfg=dict(in_dim=X[0].shape[-1], hidden=a.hidden, K=a.K)), out / "latent.pt")
        elif ep - best_ep >= a.patience:
            break
    pd.DataFrame(rows).to_csv(out / "latent_log.csv", index=False)
    g.load_state_dict(torch.load(out / "latent.pt")["state"])
    g.eval()
    with torch.no_grad():                                                  # pooled channels of every training image
        U = [torch.cat([torch.einsum("mp,bpk->bmk", A, g(x[i:i + 500])) for i in range(0, x.shape[0], 500)]) for x in X]
    del X
    torch.cuda.empty_cache()
    return g, U, betas, dict(best_val_ev=best, best_epoch=best_ep, train_sec=time.time() - t0)
