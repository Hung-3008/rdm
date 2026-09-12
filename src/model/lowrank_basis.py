"""Low-rank basis encoder: the baseline kept from docs/propose/propose01.md.

Slow stage (trained once on the training subjects, plain multi-output regression):
    FunctionalBasis   f(x) = SiLU(x U1 + b1) U2  in R^K, one linear head per training subject
Fast stage (new subject, closed form):
    ridge_loo         C = (F'F + lam I)^-1 F'B with one lambda for all voxels chosen by
                      leave-one-out MSE; prediction  B_mean + (f(x) - f_mean) C
Optional anatomical prior (docs/propose/propose01.md, addendum 2026-09-11):
    gaussian_anatomical_prior   C0(v) = mean over training subjects of the Gaussian-weighted (MNI
                      distance, sigma mm) average of their full-data coefficients; the fast stage
                      then fits ridge to the residual  B - F C0  and predicts with  C0 + C_res.
PopEB, population-prior empirical Bayes (docs/propose/propose01.md, section 9):
    popeb             posterior-mean coefficients under a kernel-density prior over the full-data
                      coefficients of the training subjects' voxels; per-voxel shrinkage
                      lambda_v = sigma_v^2 / tau2 toward a softmax-weighted population mean.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_


class FunctionalBasis(nn.Module):
    """Shared K-dimensional functional basis over image embeddings."""

    def __init__(self, embed_dim: int, hidden_dim: int, basis_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, basis_dim, bias=False)
        self.cfg = dict(embed_dim=embed_dim, hidden_dim=hidden_dim, basis_dim=basis_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(F_.silu(self.lin1(x)))

    @classmethod
    def from_file(cls, path, device) -> "FunctionalBasis":
        ck = torch.load(path, map_location=device)
        cfg = {k: ck["cfg"][k] for k in ("embed_dim", "hidden_dim", "basis_dim")}   # older files also carry alpha/beta (=SiLU)
        m = cls(**cfg)
        m.load_state_dict(ck["state_dict"])
        return m.to(device).eval()


class SubjectHeads(nn.Module):
    """One linear head C_k (K -> N_k) per training subject; discarded after training."""

    def __init__(self, basis_dim: int, num_voxels: Sequence[int]):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(basis_dim, n) for n in num_voxels])

    def forward(self, f: torch.Tensor, subject: int) -> torch.Tensor:
        return self.heads[subject](f)


def ridge_loo(F: torch.Tensor, B: torch.Tensor, lambdas: Sequence[float]) -> Tuple[torch.Tensor, float]:
    """Ridge with one lambda for all voxels chosen by total leave-one-out MSE (closed form
    via the hat matrix). F: (n, K) centred design; B: (n, V) centred targets.
    Returns C (K, V) and the chosen lambda."""
    n, K = F.shape
    evals, Q = torch.linalg.eigh(F.T @ F)                      # F'F = Q diag(evals) Q'
    FQ = F @ Q
    QtFtB = FQ.T @ B
    best_mse, best_C, best_lam = float("inf"), None, None
    for lam in lambdas:
        inv = 1.0 / (evals + lam)
        h = ((FQ ** 2) * inv).sum(1)                           # diag of F (F'F + lam I)^-1 F'
        if (1.0 - h).min() < 0.05:                             # near-interpolating (K >= n): LOO unstable
            continue
        C = Q @ (inv[:, None] * QtFtB)
        mse = (((B - F @ C) / (1.0 - h)[:, None]) ** 2).mean().item()
        if mse < best_mse:
            best_mse, best_C, best_lam = mse, C, lam
    return best_C, best_lam


@dataclass
class LowRankConfig:
    lambdas: Sequence[float] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
    tau2: float = 1e-3      # PopEB kernel bandwidth (prior variance per basis dimension), nested-calibrated
    tau2_ratio: float = 0.0     # if > 0: tau2 = tau2_ratio x mean per-dimension variance of the population (basis-independent)
    lam0: float = 1.0       # PopEB ridge for the support estimate C_hat and its LOO noise variance
    spatial_sigma: float = 0.0   # PopEB: Gaussian weighting of population voxels by MNI distance (mm); 0 = off
    std_global: bool = False     # PopEB: standardise basis features with pooled training-subject statistics
    match_sigma: float = 0.0     # PopEB: population weights computed from C_hat smoothed over the new subject's
                                 # neighbouring voxels (Gaussian, mm); the posterior keeps the voxel's own C_hat. 0 = off


@torch.inference_mode()
def smooth_over_voxels(C: torch.Tensor, xyz: torch.Tensor, sigma: float, chunk: int = 1024) -> torch.Tensor:
    """Gaussian (sigma mm) weighted average of C (K, V) over the voxels xyz (V, 3) of the same subject."""
    out = torch.empty_like(C)
    pn = (xyz ** 2).sum(1)
    for a in range(0, xyz.shape[0], chunk):
        xv = xyz[a:a + chunk]
        d2 = (xv ** 2).sum(1, keepdim=True) - 2 * xv @ xyz.T + pn[None, :]
        out[:, a:a + chunk] = (torch.softmax(-0.5 * d2 / sigma ** 2, 1) @ C.T).T
    return out


@torch.inference_mode()
def popeb(F: torch.Tensor, B: torch.Tensor, Cpop: torch.Tensor, tau2: float, lam0: float = 1.0,
          spatial: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
          chunk: int = 512, match_smooth: Optional[Tuple[torch.Tensor, float]] = None) -> torch.Tensor:
    """PopEB: posterior-mean coefficients under a population kernel prior.
    F (n, K) standardised design, B (n, V) centred targets, Cpop (K, J) full-data coefficients of
    the training subjects' voxels in the same standardised basis.
        prior       p(C) = 1/J sum_j N(C_j, tau2 I)
        likelihood  C_hat_v ~ N(C, sig2_v G^-1),  G = F'F,  C_hat_v = (G + lam0 I)^-1 F'B_v,
                    sig2_v = leave-one-out MSE of that ridge
        posterior   m_v = (G/sig2_v + I/tau2)^-1 (G C_hat_v/sig2_v + Cbar_v/tau2)
                    Cbar_v = sum_j w_vj C_j,   w_vj ∝ N(C_hat_v; C_j, sig2_v G^-1 + tau2 I)
    i.e. per-voxel ridge with lambda_v = sig2_v/tau2 toward a softmax-weighted population mean.
    spatial = (xyz_v (V, 3), xyz_pop (J, 3), sigma_mm): w_vj additionally ∝ exp(-|xyz_v - xyz_j|^2 / 2 sigma^2)
    (propose01.md 9.6). The precision (sig2 G^-1 + tau2 I)^-1 is formed as (sig2 I + tau2 G)^-1 G so
    that K >= n (singular G) is handled.
    match_smooth = (xyz_v (V, 3), sigma_mm): the weights w_vj use C_hat averaged over the new subject's
    neighbouring voxels (docs/results.md E18); the likelihood term keeps the voxel's own C_hat.
    Returns m (K, V)."""
    n, K = F.shape
    G = F.T @ F
    evals, Q = torch.linalg.eigh(G)
    FQ = F @ Q
    inv = 1.0 / (evals + lam0)
    h = ((FQ ** 2) * inv).sum(1)
    Chat = Q @ (inv[:, None] * (FQ.T @ B))                                  # (K, V)
    sig2 = (((B - F @ Chat) / (1.0 - h)[:, None]) ** 2).mean(0)             # (V,)
    Cmatch = Chat if match_smooth is None else smooth_over_voxels(Chat, match_smooth[0], match_smooth[1])
    I = torch.eye(K, device=F.device)
    CC = (Cpop[:, None, :] * Cpop[None, :, :]).reshape(K * K, -1)           # (K*K, J) outer products
    if spatial is not None:
        xyz_v, xyz_p, sigma = spatial
        pn = (xyz_p ** 2).sum(1)
    out = torch.empty_like(Chat)
    for a in range(0, Chat.shape[1], chunk):
        s2 = sig2[a:a + chunk]
        c = Cmatch[:, a:a + chunk].T                                        # (v, K), weights only
        M = torch.linalg.solve(s2[:, None, None] * I[None] + (tau2 * G)[None], G[None].expand(len(s2), K, K))
        cM = torch.einsum("vi,vij->vj", c, M)
        logits = -0.5 * ((cM * c).sum(1, keepdim=True) - 2 * cM @ Cpop + M.reshape(len(s2), -1) @ CC)   # (v, J)
        if spatial is not None:
            xv = xyz_v[a:a + chunk]
            logits = logits - 0.5 * ((xv ** 2).sum(1, keepdim=True) - 2 * xv @ xyz_p.T + pn[None, :]) / sigma ** 2
        Cbar = torch.softmax(logits, 1) @ Cpop.T                            # (v, K)
        A = G[None] / s2[:, None, None] + I[None] / tau2
        rhs = (G @ Chat[:, a:a + chunk]).T / s2[:, None] + Cbar / tau2
        out[:, a:a + chunk] = torch.linalg.solve(A, rhs[:, :, None])[:, :, 0].T
    return out


@torch.inference_mode()
def predict_lowrank(ic_img: torch.Tensor, ic_beta: torch.Tensor, q_img: torch.Tensor,
                    basis: FunctionalBasis, cfg: LowRankConfig = LowRankConfig(),
                    C0: Optional[torch.Tensor] = None,
                    Cpop: Optional[torch.Tensor] = None,
                    spatial: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
                    F_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    xyz_v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """ic_img (n, E), ic_beta (n, V), q_img (q, E) on one device; optional prior mean C0 (K, V) or
    PopEB population coefficients Cpop (K, J) (mutually exclusive; the combination is untested).
    spatial (PopEB only): (xyz of the new subject's voxels, xyz of the population voxels, sigma mm).
    F_stats: (mean, std) of the basis features used for standardisation instead of the support's own
    (propose01.md 9.6: pooled statistics of the training subjects). xyz_v (V, 3): the new subject's MNI
    coordinates, needed when cfg.match_sigma > 0. Returns pred (V, q) and diagnostics."""
    B_mean = ic_beta.mean(0, keepdim=True)
    F_sup, F_q = basis(ic_img), basis(q_img)
    if F_stats is None:
        F_mean, F_std = F_sup.mean(0, keepdim=True), F_sup.std(0, keepdim=True).clamp_min(1e-6)
    else:
        F_mean, F_std = F_stats
    F_n = (F_sup - F_mean) / F_std
    if Cpop is not None:
        if C0 is not None:
            raise ValueError("C0 and Cpop together are untested; pass one of them")
        ms = (xyz_v, cfg.match_sigma) if cfg.match_sigma > 0 else None
        C = popeb(F_n, ic_beta - B_mean, Cpop, cfg.tau2, cfg.lam0, spatial=spatial, match_smooth=ms)
        return (B_mean + ((F_q - F_mean) / F_std) @ C).T, dict(tau2=cfg.tau2)
    target = ic_beta - B_mean if C0 is None else ic_beta - B_mean - F_n @ C0
    C, lam = ridge_loo(F_n, target, cfg.lambdas)
    if C0 is not None:
        C = C0 + C
    return (B_mean + ((F_q - F_mean) / F_std) @ C).T, dict(lam=lam)


@torch.inference_mode()
def feature_stats(basis: FunctionalBasis, imgs: Sequence[np.ndarray], device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pooled (mean, std) of the basis features over the given image sets (global standardisation)."""
    F = torch.cat([basis(torch.from_numpy(np.asarray(im, dtype=np.float32)).to(device)) for im in imgs], 0)
    return F.mean(0, keepdim=True), F.std(0, keepdim=True).clamp_min(1e-6)


@torch.inference_mode()
def full_coefficients(basis: FunctionalBasis, img: np.ndarray, beta: np.ndarray, device,
                      lam: float = 1.0) -> np.ndarray:
    """Ridge over all images of one training subject in the standardised basis -> C (K, V), numpy."""
    F = basis(torch.from_numpy(np.asarray(img, dtype=np.float32)).to(device))
    F = (F - F.mean(0, keepdim=True)) / F.std(0, keepdim=True).clamp_min(1e-6)
    B = torch.from_numpy(np.asarray(beta, dtype=np.float32)).to(device)
    C = torch.linalg.solve(F.T @ F + lam * torch.eye(F.shape[1], device=device), F.T @ (B - B.mean(0, keepdim=True)))
    return C.cpu().numpy()


def gaussian_anatomical_prior(coords: np.ndarray, train: List[Tuple[np.ndarray, np.ndarray]],
                              sigma: float, rmax: Optional[float] = None) -> np.ndarray:
    """coords (V, 3) MNI mm of the new subject; train = [(coords_s (V_s, 3), C_s (K, V_s)), ...].
    Returns C0 (K, V): for each training subject the exp(-d^2 / 2 sigma^2)-weighted mean of C_s over
    its voxels within rmax (default 3 sigma) mm, nearest voxel if none, averaged over subjects."""
    from scipy.spatial import cKDTree
    rmax = rmax or 3.0 * sigma
    C0 = np.zeros((train[0][1].shape[0], len(coords)), np.float32)
    for xyz, Cs in train:
        tree = cKDTree(xyz)
        nb = tree.query_ball_point(coords, rmax)
        _, nn = tree.query(coords)
        for v, lst in enumerate(nb):
            if not lst:
                C0[:, v] += Cs[:, nn[v]]
                continue
            d = np.linalg.norm(xyz[lst] - coords[v], axis=1)
            w = np.exp(-0.5 * (d / sigma) ** 2)
            C0[:, v] += Cs[:, lst] @ (w / w.sum())
    return C0 / len(train)
