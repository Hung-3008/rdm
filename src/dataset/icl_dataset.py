"""Voxel-as-task dataset for in-context training.

Each item is one voxel: a support set of (image embedding, beta) pairs plus a
query set of (image embedding, beta) pairs drawn from the same subject, and the
subject's reference ridge weights for that voxel (used only for logging / the
optional weight-space loss).

Adapted from BraInCoRL scripts/dataset.py (Apache-2.0).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SubjectArrays:
    """One subject's arrays. May be np.memmap; must be float32 and unit-normed images."""
    img: np.ndarray    # (num_img, E), rows unit-norm
    beta: np.ndarray   # (num_img, num_vxl)
    gt: np.ndarray     # (num_vxl, E+1) reference linear weights [w; bias]
    coord: Optional[np.ndarray] = None   # (num_vxl, 3) voxel coordinate in a common space (mm)

    def __post_init__(self):
        if self.img.shape[0] != self.beta.shape[0]:
            raise ValueError(f"img has {self.img.shape[0]} images but beta has {self.beta.shape[0]}")
        if self.gt.shape != (self.beta.shape[1], self.img.shape[1] + 1):
            raise ValueError(f"gt shape {self.gt.shape} != (num_vxl={self.beta.shape[1]}, E+1={self.img.shape[1] + 1})")
        if self.coord is not None and self.coord.shape != (self.beta.shape[1], 3):
            raise ValueError(f"coord shape {self.coord.shape} != (num_vxl={self.beta.shape[1]}, 3)")

    @property
    def num_img(self):
        return self.img.shape[0]

    @property
    def num_vxl(self):
        return self.beta.shape[1]


class IclVoxelDataset(Dataset):
    def __init__(self, subjects: List[SubjectArrays], max_in_context: int, max_query: int):
        if not subjects:
            raise ValueError("need at least one subject")
        self.subjects = subjects
        self.max_in_context = max_in_context
        self.max_query = max_query
        for s in subjects:
            if s.num_img < max_in_context + max_query:
                raise ValueError(f"subject has {s.num_img} images < max_in_context + max_query "
                                 f"= {max_in_context + max_query}")
        self.cumsum = np.cumsum([0] + [s.num_vxl for s in subjects])

    def __len__(self):
        return int(self.cumsum[-1])

    def _locate(self, glob_idx: int):
        subj = int(np.searchsorted(self.cumsum, glob_idx, side="right") - 1)
        return subj, glob_idx - int(self.cumsum[subj])

    def __getitem__(self, glob_idx: int):
        subj_idx, v = self._locate(glob_idx)
        s = self.subjects[subj_idx]

        # torch RNG is re-seeded per DataLoader worker; numpy's is not.
        sel = torch.randperm(s.num_img)[:self.max_in_context + self.max_query].numpy()
        ic, q = np.sort(sel[:self.max_in_context]), np.sort(sel[self.max_in_context:])

        ic_img = torch.from_numpy(np.asarray(s.img[ic], dtype=np.float32))
        q_img = torch.from_numpy(np.asarray(s.img[q], dtype=np.float32))
        ic_beta = torch.from_numpy(np.asarray(s.beta[ic, v], dtype=np.float32))
        q_beta = torch.from_numpy(np.asarray(s.beta[q, v], dtype=np.float32))
        gt = torch.from_numpy(np.asarray(s.gt[v], dtype=np.float32))
        # coord is zeros when the source has none; models that need it must be given a
        # source with `with_coords: true` (see sources.py).
        coord = torch.from_numpy(np.asarray(s.coord[v], dtype=np.float32)) if s.coord is not None else torch.zeros(3)
        return ic_img, ic_beta, q_img, q_beta, gt, coord
