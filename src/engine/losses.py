"""Losses over (pred_beta, query_beta, pred_weights, gt_weights).

Signature of every registered loss: loss(pred, target, weights, gt_weights) -> scalar.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import Registry

LOSSES = Registry("loss")


@LOSSES.register("pmse")
class PredMSE(nn.Module):
    """MSE between predicted and measured query betas (BraInCoRL fine-tuning loss)."""

    def forward(self, pred, target, weights, gt_weights):
        return F.mse_loss(pred, target)


@LOSSES.register("weight_cos_l2")
class WeightCosL2(nn.Module):
    """cosine(w, w_gt) on the non-bias part + lambda * MSE on full [w; b].
    This is the validation criterion in the original repo."""

    def __init__(self, lambda_l2: float = 1.0):
        super().__init__()
        self.lambda_l2 = lambda_l2

    def forward(self, pred, target, weights, gt_weights):
        w, g = weights[:, :-1], gt_weights[:, :-1]
        cos = F.cosine_embedding_loss(F.normalize(w, dim=-1), F.normalize(g, dim=-1),
                                      torch.ones(w.shape[0], device=w.device))
        return cos + self.lambda_l2 * F.mse_loss(weights, gt_weights)


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    return LOSSES.build(cfg)
