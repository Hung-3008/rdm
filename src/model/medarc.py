"""MedARC Algonauts 2025 feature encoder adapted to static images.

Adapted from references/algonauts2025/src/models.py::MultiSubjectConvLinearEncoder and
src/layers.py (MIT, Copyright (c) 2025 Paul Scotti). Changes for NSD images:
  - no temporal axis: the per-feature LinearConv (linear + depthwise temporal conv) becomes a
    Linear (the reference's own kernel_size <= 1 branch); the temporal hidden model is dropped;
  - one pooled embedding per backbone, input (N, D_i) instead of (N, T, L_i, D_i);
  - subjects have different voxel counts, so the shared decoder requires equal counts.

    features x_i (N, D_i) --Linear--> (N, d) --pool over i--> (N, d)
      --> shared Linear(d, V) + subject Linear(d, V_s)
"""

from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearPoolLatent(nn.Module):
    """Learned per-feature, per-channel weights, summed over features. (N, L, d) -> (N, d)."""

    def __init__(self, embed_dim: int, feat_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(feat_size, embed_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.weight, dim=-2)


class AttentionPoolLatent(nn.Module):
    """One learned query attending over the feature tokens. (N, L, d) -> (N, d).
    The reference constructs this with (embed_dim, total_feat_size, num_heads=...), which does not
    match its signature (embed_dim, num_heads); here the signature is used as defined."""

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.empty(embed_dim))
        self.kv = nn.Linear(embed_dim, 2 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, C = x.shape
        h = self.num_heads
        q = self.query.expand(N, 1, C).reshape(N, 1, h, C // h).transpose(1, 2)
        k, v = self.kv(x).reshape(N, L, 2, h, C // h).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v).reshape(N, C)
        return self.proj(x)


class MultiFeatureEncoder(nn.Module):
    def __init__(
        self,
        feat_dims: Sequence[int],
        num_voxels: Sequence[int],
        embed_dim: int = 192,
        global_pool: Literal["avg", "linear", "attn"] = "avg",
        pool_num_heads: int = 4,
        with_shared_decoder: bool = True,
        with_subject_decoders: bool = True,
    ):
        assert with_shared_decoder or with_subject_decoders
        assert not with_shared_decoder or len(set(num_voxels)) == 1, "shared decoder needs equal voxel counts"
        super().__init__()
        self.global_pool = global_pool
        self.feat_embeds = nn.ModuleList([nn.Linear(d, embed_dim) for d in feat_dims])

        if global_pool == "avg":
            self.feat_pool = None
        elif global_pool == "linear":
            self.feat_pool = LinearPoolLatent(embed_dim, len(feat_dims))
        elif global_pool == "attn":
            self.feat_pool = AttentionPoolLatent(embed_dim, num_heads=pool_num_heads)
        else:
            raise NotImplementedError(f"global_pool {global_pool} not implemented.")

        self.shared_decoder = nn.Linear(embed_dim, num_voxels[0]) if with_shared_decoder else None
        self.subject_decoders = (nn.ModuleList([nn.Linear(embed_dim, v) for v in num_voxels])
                                 if with_subject_decoders else None)
        self.apply(_init_weights)

    def embed(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        # features: list of (N, D_i) -> (N, d)
        embeds = [embed(x) for x, embed in zip(features, self.feat_embeds)]
        if self.feat_pool is None:
            return sum(embeds)           # reference: sum over features (mean over layers of each feature)
        return self.feat_pool(torch.stack(embeds, dim=1))

    def forward(self, features: Sequence[torch.Tensor], subject: int = 0) -> torch.Tensor:
        # features: list of (N, D_i) -> (N, V_subject)
        embed = self.embed(features)
        out = 0.0
        if self.shared_decoder is not None:
            out = out + self.shared_decoder(embed)
        if self.subject_decoders is not None:
            out = out + self.subject_decoders[subject](embed)
        return out


class MedarcTrunk(nn.Module):
    """Feature embeddings + pooling of a trained MultiFeatureEncoder, used as the basis f(x) (N, d) of
    src.eval_lowrank. Input: the backbone features concatenated along dim 1 (any scaling per block);
    each block is unit-normed again, as in training."""

    def __init__(self, encoder: MultiFeatureEncoder, feat_dims: Sequence[int]):
        super().__init__()
        self.encoder = encoder
        self.feat_dims = list(feat_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [F.normalize(b, dim=1) for b in torch.split(x, self.feat_dims, dim=1)]
        return self.encoder.embed(feats)

    @classmethod
    def from_file(cls, path, device) -> "MedarcTrunk":
        ck = torch.load(path, map_location=device)
        assert not ck["cfg"]["data"]["standardize"], "trunk input standardisation is not stored"
        enc = MultiFeatureEncoder(ck["feat_dims"], ck["num_voxels"], **ck["cfg"]["model"])
        enc.load_state_dict(ck["state_dict"])
        return cls(enc, ck["feat_dims"]).to(device).eval()


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
