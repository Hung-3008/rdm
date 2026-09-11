"""BraInCoRL + anatomical coordinate token.

Adds one token gamma(x) built from Fourier features of the voxel's coordinate in a
common space (MNI mm), following the positional encoding of NRF (Chen et al.,
ICLR 2026, Sec. 3.1). Token order: [register tokens, coordinate token, context].
With an empty context the hypernetwork still emits weights, so the model can be
queried zero-shot from the coordinate alone.

Training-time augmentations: Gaussian jitter of the coordinate (mm) and random
dropout of the coordinate token, so the model keeps working without coordinates.
"""

import math

import torch
import torch.nn as nn

from .braincorl import BraInCoRL
from .factory import MODELS


class FourierCoordEncoder(nn.Module):
    """x (B, 3) in mm -> token (B, out_dim). Frequencies are fixed at init."""

    def __init__(self, out_dim: int, num_freq: int = 64, sigma: float = 2.0,
                 coord_scale_mm: float = 100.0, hidden_dim: int | None = None):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.register_buffer("B", torch.randn(3, num_freq, generator=gen) * sigma)
        self.coord_scale_mm = coord_scale_mm
        hidden_dim = hidden_dim or out_dim
        self.mlp = nn.Sequential(nn.Linear(2 * num_freq, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (x / self.coord_scale_mm) @ self.B      # (B, F)
        return self.mlp(torch.cat([proj.cos(), proj.sin()], dim=-1))


@MODELS.register("braincorl_coord")
class BraInCoRLCoord(BraInCoRL):
    def __init__(self, num_freq: int = 64, fourier_sigma: float = 2.0, coord_scale_mm: float = 100.0,
                 jitter_mm: float = 2.0, coord_token_dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.coord_encoder = FourierCoordEncoder(self.internal_dim, num_freq, fourier_sigma, coord_scale_mm)
        self.jitter_mm = jitter_mm
        self.coord_token_dropout = coord_token_dropout
        # learned stand-in used when the coordinate token is dropped / absent
        self.no_coord_token = nn.Parameter(torch.zeros(1, 1, self.internal_dim))

    def predict_weights(self, ic_img, ic_beta, coord=None, need_attn: bool = False):
        B = ic_img.shape[0]
        if coord is None:
            tok = self.no_coord_token.expand(B, -1, -1)
        else:
            if self.training and self.jitter_mm > 0:
                coord = coord + self.jitter_mm * torch.randn_like(coord)
            tok = self.coord_encoder(coord).unsqueeze(1)                  # (B, 1, D)
            if self.training and self.coord_token_dropout > 0:
                drop = (torch.rand(B, 1, 1, device=tok.device) < self.coord_token_dropout)
                tok = torch.where(drop, self.no_coord_token.expand(B, -1, -1), tok)

        x = self.early_layers(torch.cat([ic_img, ic_beta.unsqueeze(-1)], dim=-1))
        x = torch.cat([self.reg_tokens.expand(B, -1, -1), tok, x], dim=1)
        x = self.input_dropout(x)
        for blk in self.blocks:
            x = blk(x, need_weights=need_attn)
        return self.weight_pred(x[:, 0, :])

    def forward(self, ic_img, ic_beta, query_img, coord=None):
        from ..utils.predict import linear_readout
        weights = self.predict_weights(ic_img, ic_beta, coord)
        return linear_readout(weights, query_img), weights
