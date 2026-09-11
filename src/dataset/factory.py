from typing import Any, Dict

import lightning as pl
from torch.utils.data import DataLoader

from ..registry import Registry
from .icl_dataset import IclVoxelDataset

SOURCES = Registry("data source")


class IclDataModule(pl.LightningDataModule):
    def __init__(self, source: Dict[str, Any], batch_size: int, max_in_context: int,
                 max_query: int, num_workers: int = 4):
        super().__init__()
        self.source_cfg = source
        self.batch_size = batch_size
        self.max_in_context = max_in_context
        self.max_query = max_query
        self.num_workers = num_workers
        self.source = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        if self.source is None:
            self.source = SOURCES.build(self.source_cfg)
        if self.train_ds is None:
            self.train_ds = IclVoxelDataset(self.source.load_split("train"),
                                            self.max_in_context, self.max_query)
            self.val_ds = IclVoxelDataset(self.source.load_split("val"),
                                          self.max_in_context, self.max_query)

    def _loader(self, ds, shuffle):
        # "fork" so that np.memmap arrays are shared copy-on-write; the default
        # forkserver/spawn start method (Python >= 3.14) pickles them as full copies.
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0,
                          multiprocessing_context="fork" if self.num_workers > 0 else None)

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)


def build_datamodule(cfg: Dict[str, Any]) -> IclDataModule:
    """cfg: {'source': {'name': ..., ...}, 'batch_size', 'max_in_context', 'max_query', ...}"""
    return IclDataModule(**cfg)
