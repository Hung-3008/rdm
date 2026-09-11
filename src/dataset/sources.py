"""Data sources: anything that yields SubjectArrays for a split, and an
evaluation bundle for one held-out subject.

Register a new source with @SOURCES.register("<name>"). A source must implement:
    load_split(split: "train" | "val") -> List[SubjectArrays]
    load_eval(subject) -> EvalBundle
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..utils.predict import is_unit_norm, unit_norm
from .factory import SOURCES
from .icl_dataset import SubjectArrays


@dataclass
class EvalBundle:
    """Arrays for few-shot evaluation on one subject."""
    ic_img: np.ndarray      # (num_pool_img, E)  candidate support images (unit-norm)
    ic_beta: np.ndarray     # (num_pool_img, V)
    query_img: np.ndarray   # (num_query, E)     held-out test images (unit-norm)
    query_beta: np.ndarray  # (num_query, V)
    gt: np.ndarray          # (V, E+1)
    icl_order: np.ndarray   # permutation of pool indices; support sets are consecutive slices
    coord: Optional[np.ndarray] = None   # (V, 3) mm in a common space


# --------------------------------------------------------------------------- #
# BraInCoRL preprocessed NSD layout
# --------------------------------------------------------------------------- #
@SOURCES.register("npy_multisubj")
class NpyMultiSubjectSource:
    """File layout (per subject index i, under data_dir):
        s{i}_unique_img_feats_normed[_val].npy   (num_img, E)
        s{i}_unique_nrn_feats_msk[_val].npy      (num_img, num_vxl)
        s{i}_gt[_val].npy                        (num_vxl, E+1)
        s{i}_coords[_val].npy                    (num_vxl, 3)   only read when with_coords=True
      shared test set:
        common_img_feats_normed.npy              (num_common, E)
        s{i}_common_nrn_feats_msk.npy            (num_common, num_vxl)
        test_time_icl_idx.json                   {"large_shuffled": [...]}
    """

    def __init__(self, data_dir: str, train_subjects: List[int], val_subjects: List[int],
                 mmap: bool = True, with_coords: bool = False):
        self.data_dir = Path(data_dir)
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.mmap_mode = "r" if mmap else None
        self.with_coords = with_coords

    def _coords(self, i: int, suffix: str = "") -> Optional[np.ndarray]:
        if not self.with_coords:
            return None
        p = self.data_dir / f"s{i}_coords{suffix}.npy"
        if not p.exists():
            raise FileNotFoundError(f"with_coords=True but {p} is missing (run preprocess_nsd --stage coords layout)")
        return np.load(p).astype(np.float32)

    def _load(self, i: int, suffix: str) -> SubjectArrays:
        d = self.data_dir
        img = np.load(d / f"s{i}_unique_img_feats_normed{suffix}.npy", mmap_mode=self.mmap_mode)
        beta = np.load(d / f"s{i}_unique_nrn_feats_msk{suffix}.npy", mmap_mode=self.mmap_mode)
        gt = np.load(d / f"s{i}_gt{suffix}.npy", mmap_mode=self.mmap_mode)
        if not is_unit_norm(np.asarray(img[:256])):
            raise ValueError(f"image features for subject {i} are not unit-normed")
        return SubjectArrays(img=img, beta=beta, gt=gt, coord=self._coords(i, suffix))

    def load_split(self, split: str) -> List[SubjectArrays]:
        if split == "train":
            return [self._load(i, "") for i in self.train_subjects]
        if split == "val":
            return [self._load(i, "_val") for i in self.val_subjects]
        raise ValueError(split)

    def load_eval(self, subject: int) -> EvalBundle:
        import json
        d = self.data_dir
        ic_img = unit_norm(np.load(d / f"s{subject}_unique_img_feats_normed.npy").astype(np.float32))
        ic_beta = np.load(d / f"s{subject}_unique_nrn_feats_msk.npy").astype(np.float32)
        q_img = unit_norm(np.load(d / "common_img_feats_normed.npy").astype(np.float32))
        q_beta = np.load(d / f"s{subject}_common_nrn_feats_msk.npy").astype(np.float32)
        gt = np.load(d / f"s{subject}_gt.npy").astype(np.float32)
        with open(d / "test_time_icl_idx.json") as f:
            order = np.asarray(json.load(f)["large_shuffled"], dtype=np.int64)
        return EvalBundle(ic_img, ic_beta, q_img, q_beta, gt, order, coord=self._coords(subject))


# --------------------------------------------------------------------------- #
# Synthetic linear voxels (analysis-by-synthesis; BraInCoRL pretraining stage)
# --------------------------------------------------------------------------- #
MNI_BOX = np.array([[-70.0, 70.0], [-100.0, 10.0], [-50.0, 70.0]], np.float32)   # mm, from NSD S1/2/5/7


def _make_subject(rng: np.random.Generator, num_img: int, num_vxl: int, embed_dim: int,
                  noise_std: float, num_common: int = 0, with_coords: bool = False,
                  coord_noise_frac: float = 1.0, field_num_freq: int = 32, field_sigma: float = 1.0,
                  beta_std: float = 1.0):
    """Random unit-norm image features, random linear weights, beta = x.w + b + noise.

    With coordinates, each voxel gets a position in an MNI-like box and its weights are
    w = sqrt(1-rho) * W_field(x) + sqrt(rho) * n_v, where W_field is a random smooth
    function of position (random Fourier features) and n_v is voxel-specific noise;
    rho = coord_noise_frac (1.0 -> no spatial structure). Betas are scaled to `beta_std`.
    """
    E = embed_dim
    n_v = rng.standard_normal((num_vxl, E + 1), dtype=np.float32)
    coord = None
    if with_coords:
        coord = (MNI_BOX[:, 0] + rng.random((num_vxl, 3), dtype=np.float32) * (MNI_BOX[:, 1] - MNI_BOX[:, 0])).astype(np.float32)
        B = rng.standard_normal((3, field_num_freq), dtype=np.float32) * field_sigma
        proj = 2 * np.pi * (coord / 100.0) @ B
        phi = np.concatenate([np.cos(proj), np.sin(proj)], 1) / np.sqrt(field_num_freq)   # unit variance
        A = rng.standard_normal((2 * field_num_freq, E + 1), dtype=np.float32)
        w_field = phi @ A
        gt = (np.sqrt(1 - coord_noise_frac) * w_field + np.sqrt(coord_noise_frac) * n_v).astype(np.float32)
    else:
        gt = n_v
    gt[:, -1] *= 0.1  # small bias
    scale = beta_std / np.sqrt(1.0 + noise_std ** 2)
    gt *= scale

    def responses(n):
        x = unit_norm(rng.standard_normal((n, E), dtype=np.float32))
        beta = x @ gt[:, :-1].T + gt[:, -1] + scale * noise_std * rng.standard_normal((n, num_vxl), dtype=np.float32)
        return x.astype(np.float32), beta.astype(np.float32)

    img, beta = responses(num_img)
    common = responses(num_common) if num_common > 0 else None
    return SubjectArrays(img=img, beta=beta, gt=gt, coord=coord), common


@SOURCES.register("synthetic")
class SyntheticSource:
    """Deterministic synthetic subjects. Signal variance is 1 (unit-norm x, w ~ N(0, I)),
    so the explained-variance ceiling is 1 / (1 + noise_std**2)."""

    def __init__(self, embed_dim: int = 512, num_img: int = 1000, num_vxl: int = 256,
                 num_train_subjects: int = 2, num_val_subjects: int = 1,
                 noise_std: float = 1.0, num_common: int = 200, seed: int = 0,
                 with_coords: bool = False, coord_noise_frac: float = 1.0,
                 field_num_freq: int = 32, field_sigma: float = 1.0, beta_std: float = 1.0):
        self.embed_dim = embed_dim
        self.num_img = num_img
        self.num_vxl = num_vxl
        self.num_train_subjects = num_train_subjects
        self.num_val_subjects = num_val_subjects
        self.noise_std = noise_std
        self.num_common = num_common
        self.seed = seed
        self.gen_kwargs = dict(with_coords=with_coords, coord_noise_frac=coord_noise_frac,
                               field_num_freq=field_num_freq, field_sigma=field_sigma, beta_std=beta_std)

    def _subject(self, subject: int, num_common: int = 0):
        rng = np.random.default_rng(self.seed * 1000 + subject)
        return _make_subject(rng, self.num_img, self.num_vxl, self.embed_dim, self.noise_std, num_common,
                             **self.gen_kwargs)

    def load_split(self, split: str) -> List[SubjectArrays]:
        if split == "train":
            ids = range(self.num_train_subjects)
        elif split == "val":
            ids = range(self.num_train_subjects, self.num_train_subjects + self.num_val_subjects)
        else:
            raise ValueError(split)
        return [self._subject(i)[0] for i in ids]

    def load_eval(self, subject: int) -> EvalBundle:
        s, (q_img, q_beta) = self._subject(subject, num_common=self.num_common)
        order = np.random.default_rng(self.seed + 7).permutation(s.num_img)
        return EvalBundle(s.img, s.beta, q_img, q_beta, s.gt, order, coord=s.coord)
