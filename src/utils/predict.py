"""Linear voxel readout and feature normalisation helpers.

Adapted from BraInCoRL scripts/utils.py (Apache-2.0).
"""

import numpy as np
import torch


def linear_readout(weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """weights: (B, E+1) = [w; b]; x: (B, S, E). Returns (B, S) = x @ w + b."""
    w = weights[:, :-1].unsqueeze(1)   # (B, 1, E)
    b = weights[:, -1].unsqueeze(-1)   # (B, 1)
    return torch.sum(w * x, dim=-1) + b


def unit_norm(a):
    if isinstance(a, np.ndarray):
        return a / np.linalg.norm(a, axis=-1, keepdims=True)
    if isinstance(a, torch.Tensor):
        return a / torch.norm(a, p=2, dim=-1, keepdim=True)
    raise TypeError("expected numpy array or torch tensor")


def is_unit_norm(a, dim=-1, tol=1e-5) -> bool:
    if isinstance(a, np.ndarray):
        n = np.linalg.norm(a, axis=dim)
        return bool(np.allclose(n, np.ones_like(n), atol=tol))
    if isinstance(a, torch.Tensor):
        n = torch.norm(a, p=2, dim=dim)
        return bool(torch.allclose(n, torch.ones_like(n), atol=tol))
    raise TypeError("expected numpy array or torch tensor")
