"""Batched ridge regression with per-voxel lambda selected by k-fold CV.

Adapted from BraInCoRL scripts/utils.py::ridge_fast_brainwise (Apache-2.0).
Used as the within-subject baseline in evaluation.
"""

from typing import Sequence

import torch

from .metrics import explained_variance


def ridge_cv_brainwise(X: torch.Tensor, y: torch.Tensor,
                       lambdas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5),
                       num_fold: int = 5) -> torch.Tensor:
    """
    X: (num_img, E) image features (one design matrix shared by all voxels).
    y: (num_vxl, num_img) voxel responses.
    Returns b: (num_vxl, E+1) = [w; bias], averaged over folds at each voxel's best lambda.
    """
    device = X.device
    num_img, emb_len = X.shape
    num_vxl = y.shape[0]
    num_lamb = len(lambdas)

    ev_stats = torch.empty((num_fold, num_lamb, num_vxl), device=device)
    b_stats = torch.empty((num_fold, num_lamb, num_vxl, emb_len + 1), device=device)

    fold_size = num_img // num_fold
    idx = torch.arange(num_img, device=device)

    for fold in range(num_fold):
        te = idx[fold * fold_size:(fold + 1) * fold_size]
        tr = torch.cat([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])

        X_tr = torch.cat([X[tr], torch.ones((tr.numel(), 1), device=device)], dim=1)   # (n_tr, E+1)
        X_te = torch.cat([X[te], torch.ones((te.numel(), 1), device=device)], dim=1)
        y_tr = y[:, tr, None]                                                          # (V, n_tr, 1)
        y_te = y[:, te]

        Xt = X_tr.T                                                                    # (E+1, n_tr)
        _, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
        e = S ** 2
        p = Xt @ y_tr                                                                  # (V, E+1, 1)
        Q = Xt @ Vh.T                                                                  # (E+1, E+1)
        r = Q.T @ p                                                                    # (V, E+1, 1)

        for li, lam in enumerate(lambdas):
            b = (1.0 / lam) * (p - Q @ (r / (e.unsqueeze(-1) + lam)))                  # (V, E+1, 1)
            b = b.squeeze(-1)
            b_stats[fold, li] = b
            ev_stats[fold, li] = explained_variance(b @ X_te.T, y_te)

    best = torch.argmax(ev_stats.mean(dim=0), dim=0)                                   # (V,)
    fold_idx = torch.arange(num_fold, device=device)[:, None]
    vxl_idx = torch.arange(num_vxl, device=device)
    return b_stats[fold_idx, best, vxl_idx, :].mean(dim=0)                             # (V, E+1)
