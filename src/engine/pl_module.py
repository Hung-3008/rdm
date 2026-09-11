"""LightningModule wrapping any registered model with in-context sampling.

Per step: the batch carries `max_in_context` support pairs per voxel; a random
subset of size U(min_in_context, max_in_context) is passed to the model so the
model learns to be length-robust (BraInCoRL context-extension stage).

Adapted from BraInCoRL scripts/model.py::HyperweightsPredictorTrainerComb (Apache-2.0).
"""

from typing import Any, Dict, Optional

import lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric

from ..model import build_model
from ..model.checkpoint import load_model_weights
from ..utils.metrics import explained_variance
from .losses import build_loss


class IclLightningModule(pl.LightningModule):
    def __init__(self, model: Dict[str, Any], loss: Dict[str, Any], val_loss: Dict[str, Any],
                 min_in_context: int, max_in_context: int,
                 lr: float = 1e-4, weight_decay: float = 1e-1,
                 sched_factor: float = 0.1, sched_patience: int = 5, min_lr: float = 1e-5,
                 warmup_steps: int = 0, init_weights: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        if min_in_context > max_in_context:
            raise ValueError("min_in_context must be <= max_in_context")
        self.model = build_model(model)
        if init_weights:
            # released BraInCoRL weights or a previous-stage Lightning ckpt; new params stay at init
            missing, unexpected = load_model_weights(self.model, init_weights, strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected keys in {init_weights}: {unexpected[:5]}")
            print(f"[INFO] init from {init_weights}; newly initialised params: {missing}")
        self.train_loss_fn = build_loss(loss)
        self.val_loss_fn = build_loss(val_loss)
        self.min_in_context = min_in_context
        self.max_in_context = max_in_context
        self.lr, self.weight_decay = lr, weight_decay
        self.sched_factor, self.sched_patience, self.min_lr = sched_factor, sched_patience, min_lr
        self.warmup_steps = warmup_steps

        names = ["loss", "expl_var", "cos_sim", "mse"]
        self.train_metrics = torch.nn.ModuleDict({n: MeanMetric() for n in names})
        self.val_metrics = torch.nn.ModuleDict({n: MeanMetric() for n in names})

    def forward(self, ic_img, ic_beta, query_img, coord=None):
        return self.model(ic_img, ic_beta, query_img, coord=coord)

    def _sample_context(self, ic_img, ic_beta):
        S = ic_img.shape[1]
        n = int(torch.randint(self.min_in_context, min(S, self.max_in_context) + 1, (1,)))
        idx = torch.randperm(S, device=ic_img.device)[:n]
        return ic_img[:, idx], ic_beta[:, idx]

    def _step(self, batch, loss_fn, metrics):
        ic_img, ic_beta, q_img, q_beta, gt, coord = batch
        ic_img, ic_beta = self._sample_context(ic_img, ic_beta)
        pred, weights = self(ic_img, ic_beta, q_img, coord=coord)
        loss = loss_fn(pred, q_beta, weights, gt)

        with torch.no_grad():
            metrics["loss"].update(loss.detach())
            metrics["expl_var"].update(explained_variance(pred, q_beta).mean())
            metrics["cos_sim"].update(F.cosine_similarity(weights, gt, dim=-1).mean())
            metrics["mse"].update(F.mse_loss(pred, q_beta))
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        # linear warmup on top of ReduceLROnPlateau (which only rescales the current lr)
        if self.warmup_steps and self.global_step < self.warmup_steps:
            scale = (self.global_step + 1) / self.warmup_steps
            for g in self.trainer.optimizers[0].param_groups:
                g["lr"] = self.lr * scale

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_loss_fn, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, self.val_loss_fn, self.val_metrics)

    def _log_epoch(self, prefix, metrics):
        for n, m in metrics.items():
            self.log(f"{prefix}_{n}", m.compute(), prog_bar=n in ("loss", "expl_var"), sync_dist=True)
            m.reset()

    def on_train_epoch_end(self):
        self._log_epoch("train", self.train_metrics)

    def on_validation_epoch_end(self):
        self._log_epoch("val", self.val_metrics)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.sched_factor, patience=self.sched_patience,
            threshold=1e-4, cooldown=2, min_lr=self.min_lr)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_loss", "interval": "epoch"}}
