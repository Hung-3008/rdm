"""Building blocks shared by hypernetwork models.

Adapted from BraInCoRL scripts/model.py (Apache-2.0).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1, x2 = self.linear1(x).chunk(2, dim=-1)
        return self.linear2(F.silu(x1) * x2)


class SwiGLUAttentionBlock(nn.Module):
    """Pre-norm self-attention + SwiGLU FFN.

    Query is scaled by log(sequence length) so the attention temperature stays
    stable when the in-context length varies at test time (BraInCoRL Eq. 4).
    """

    def __init__(self, embed_dim: int, ffn_hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, ffn_hidden_dim, embed_dim)
        self.last_attn = None

    def forward(self, x, need_weights: bool = False):
        h = self.norm1(x)
        log_scale = math.log(h.shape[-2])
        attn_out, attn_w = self.attn(log_scale * h, h, h, need_weights=need_weights)
        self.last_attn = attn_w
        x = x + self.attn_dropout(attn_out)
        return x + self.ffn(self.norm2(x))


class ResidualBlock(nn.Module):
    """Pre-activation residual MLP block (LayerNorm + LeakyReLU)."""

    def __init__(self, feat_in: int, feat_out: int, feat_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(feat_in), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(feat_in, feat_hidden),
            nn.LayerNorm(feat_hidden), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(feat_hidden, feat_out),
        )
        self.bypass = nn.Linear(feat_in, feat_out) if feat_in != feat_out else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.bypass(x)
