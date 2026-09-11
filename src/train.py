"""Train an in-context voxel encoder.

    python -m src.train --config src/config/smoke_synthetic.yaml
    python -m src.train --config src/config/braincorl_clip_3090.yaml trainer.max_epochs=1
"""

import argparse
from pathlib import Path

import lightning as pl
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .config import load_config
from .dataset import build_datamodule
from .engine import IclLightningModule


def build_trainer(cfg, out_dir: Path):
    tcfg = dict(cfg["trainer"])
    ckpt_cfg = tcfg.pop("checkpoint")
    patience = tcfg.pop("early_stopping_patience", 0)

    callbacks = [ModelCheckpoint(dirpath=out_dir / "checkpoints",
                                 filename="{epoch:03d}-{val_expl_var:.4f}",
                                 save_last=True, **ckpt_cfg)]
    if patience > 0:
        callbacks.append(EarlyStopping(monitor=ckpt_cfg["monitor"], mode=ckpt_cfg["mode"],
                                       patience=patience))
    return pl.Trainer(default_root_dir=out_dir, logger=CSVLogger(out_dir, name="logs"),
                      callbacks=callbacks, accelerator="gpu", devices=1, **tcfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    ap.add_argument("overrides", nargs="*", help="key.sub=value")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    pl.seed_everything(cfg.get("seed", 0), workers=True)
    torch.set_float32_matmul_precision("high")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    dm = build_datamodule(cfg["data"])
    module = IclLightningModule(**cfg["module"])
    n_params = sum(p.numel() for p in module.model.parameters())
    print(f"[INFO] model params: {n_params / 1e6:.1f}M")

    trainer = build_trainer(cfg, out_dir)
    trainer.fit(module, datamodule=dm, ckpt_path=args.resume)
    if torch.cuda.is_available():
        print(f"[INFO] peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
