"""BraInCoRL: in-context hypernetwork that emits a linear voxel readout.

Input tokens are [image_embedding ; beta] pairs from the support set. Learnable
register tokens are prepended; the first one is read out after the transformer
and mapped to (E+1) weights = [w ; bias] of a linear encoder for that voxel.

Adapted from BraInCoRL scripts/model.py (Apache-2.0).
"""

from typing import Optional

import torch
import torch.nn as nn

from ..utils.predict import linear_readout
from .factory import MODELS
from .layers import ResidualBlock, SwiGLUAttentionBlock

# Architecture presets from the original repo; keyed by image backbone.
PRESETS = {
    "CLIP":   dict(embed_dim=512,  internal_dim=560),
    "DINO":   dict(embed_dim=768,  internal_dim=800),
    "SIGLIP": dict(embed_dim=1152, internal_dim=1200),
}


@MODELS.register("braincorl")
class BraInCoRL(nn.Module):
    def __init__(
        self,
        preset: Optional[str] = None,
        embed_dim: Optional[int] = None,
        internal_dim: Optional[int] = None,
        num_layers: int = 20,
        ffn_hidden_dim: int = 2048,
        num_heads: int = 10,
        num_reg_tokens: int = 4,
        num_early_layers: int = 1,
        num_pred_layers: int = 1,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"unknown preset {preset}; choose from {list(PRESETS)}")
            embed_dim = embed_dim or PRESETS[preset]["embed_dim"]
            internal_dim = internal_dim or PRESETS[preset]["internal_dim"]
        if embed_dim is None or internal_dim is None:
            raise ValueError("embed_dim and internal_dim are required when no preset is given")
        if internal_dim % num_heads != 0:
            raise ValueError(f"internal_dim={internal_dim} must be divisible by num_heads={num_heads}")
        if num_early_layers < 1 or num_pred_layers < 1:
            raise ValueError("num_early_layers and num_pred_layers must be >= 1")
        mlp_hidden_dim = mlp_hidden_dim or 2 * internal_dim

        self.embed_dim = embed_dim
        self.internal_dim = internal_dim
        self.num_reg_tokens = num_reg_tokens

        # token embedding: (E+1) -> internal
        self.early_layers = nn.Sequential(
            ResidualBlock(embed_dim + 1, internal_dim, mlp_hidden_dim, dropout),
            *[ResidualBlock(internal_dim, internal_dim, mlp_hidden_dim, dropout)
              for _ in range(num_early_layers - 1)],
        )

        reg = torch.randn(1, num_reg_tokens, internal_dim) / float(internal_dim + 1) ** 0.5
        self.reg_tokens = nn.Parameter(reg)

        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SwiGLUAttentionBlock(internal_dim, ffn_hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # weight head: internal -> (E+1)
        self.weight_pred = nn.Sequential(
            *[ResidualBlock(internal_dim, internal_dim, mlp_hidden_dim, dropout)
              for _ in range(num_pred_layers - 1)],
            ResidualBlock(internal_dim, embed_dim + 1, mlp_hidden_dim, dropout),
        )

    def predict_weights(self, ic_img: torch.Tensor, ic_beta: torch.Tensor,
                        need_attn: bool = False) -> torch.Tensor:
        """ic_img: (B, S, E); ic_beta: (B, S) -> weights (B, E+1)."""
        B = ic_img.shape[0]
        x = self.early_layers(torch.cat([ic_img, ic_beta.unsqueeze(-1)], dim=-1))
        x = torch.cat([self.reg_tokens.expand(B, -1, -1), x], dim=1)
        x = self.input_dropout(x)
        for blk in self.blocks:
            x = blk(x, need_weights=need_attn)
        return self.weight_pred(x[:, 0, :])

    def forward(self, ic_img: torch.Tensor, ic_beta: torch.Tensor, query_img: torch.Tensor,
                coord: Optional[torch.Tensor] = None):
        """query_img: (B, Q, E) -> (pred (B, Q), weights (B, E+1)). `coord` is ignored here."""
        weights = self.predict_weights(ic_img, ic_beta)
        return linear_readout(weights, query_img), weights
