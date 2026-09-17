"""Voxel-wise metrics. All functions take (num_voxel, num_sample) and reduce over dim 1.

Adapted from BraInCoRL scripts/utils.py (Apache-2.0).
"""

import numpy as np
import torch


def _check(pred, tar):
    if type(pred) is not type(tar):
        raise TypeError("pred and tar must both be numpy arrays or both be torch tensors")


def point_mse(pred, tar):
    _check(pred, tar)
    if isinstance(pred, np.ndarray):
        return np.mean((tar - pred) ** 2, axis=1)
    return torch.mean((tar - pred) ** 2, dim=1)


def explained_variance(pred, tar):
    """1 - var(tar - pred) / var(tar), per row."""
    _check(pred, tar)
    if isinstance(pred, np.ndarray):
        return 1 - np.var(tar - pred, axis=1) / np.var(tar, axis=1)
    return 1 - torch.var(tar - pred, dim=1) / torch.var(tar, dim=1)


def pearson(pred, tar):
    """Pearson correlation between pred and tar, per row."""
    _check(pred, tar)
    if isinstance(pred, np.ndarray):
        p = pred - pred.mean(axis=1, keepdims=True)
        t = tar - tar.mean(axis=1, keepdims=True)
        return np.sum(p * t, axis=1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    p = pred - pred.mean(dim=1, keepdim=True)
    t = tar - tar.mean(dim=1, keepdim=True)
    return torch.sum(p * t, dim=1) / (torch.norm(p, dim=1) * torch.norm(t, dim=1) + 1e-12)


def normed_cos_similarity(pred, tar):
    _check(pred, tar)
    if isinstance(pred, np.ndarray):
        p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
        t = tar / np.linalg.norm(tar, axis=1, keepdims=True)
        return np.sum(p * t, axis=1)
    p = pred / torch.norm(pred, dim=1, keepdim=True)
    t = tar / torch.norm(tar, dim=1, keepdim=True)
    return torch.sum(p * t, dim=1)
