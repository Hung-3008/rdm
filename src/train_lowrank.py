"""Slow stage of the low-rank basis encoder (docs/propose/propose01.md, 3.1a).

Trains the shared functional basis f by multi-output regression with one linear head per
training subject (no support/query split, no hypernetwork). Early stopping uses
`val_images` held-out unique images of the training subjects only.

    python -m src.train_lowrank --config src/config/lowrank_clip.yaml [basis.basis_dim=32 ...]

Outputs in <out_dir>: basis.pt, log.csv, config.yaml, summary.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .config import load_config
from .dataset import SOURCES
from .model.lowrank_basis import FunctionalBasis, SubjectHeads
from .utils import explained_variance


def load_subjects(cfg, device):
    """Returns per-subject (img_train, beta_train, img_val, beta_val) on `device`."""
    subs = SOURCES.build(cfg["data"]["source"]).load_split("train")
    rng = np.random.default_rng(cfg["seed"])
    out = []
    for s in subs:
        perm = rng.permutation(s.num_img)
        va, tr = np.sort(perm[: cfg["basis"]["val_images"]]), np.sort(perm[cfg["basis"]["val_images"]:])
        img = torch.from_numpy(np.asarray(s.img, dtype=np.float32)).to(device)
        beta = torch.from_numpy(np.asarray(s.beta, dtype=np.float32)).to(device)
        out.append((img[tr], beta[tr], img[va], beta[va]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    out_dir = Path(args.out_dir or cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = load_subjects(cfg, device)
    E = subjects[0][0].shape[1]
    bc = cfg["basis"]

    basis = FunctionalBasis(E, bc["hidden_dim"], bc["basis_dim"]).to(device)
    heads = SubjectHeads(bc["basis_dim"], [b.shape[1] for _, b, _, _ in subjects]).to(device)
    opt = torch.optim.AdamW([{"params": basis.parameters(), "weight_decay": 0.0},
                             {"params": heads.parameters(), "weight_decay": bc["head_weight_decay"]}], lr=bc["lr"])

    def val_ev():
        basis.eval()
        with torch.no_grad():
            evs = [explained_variance((heads(basis(iv), k)).T, bv.T).mean().item()
                   for k, (_, _, iv, bv) in enumerate(subjects)]
        basis.train()
        return float(np.mean(evs)), evs

    best, best_epoch, rows, t0 = -float("inf"), -1, [], time.time()
    for epoch in range(bc["max_epochs"]):
        loss_sum, steps = 0.0, 0
        perms = [torch.randperm(it.shape[0], device=device) for it, _, _, _ in subjects]
        n_steps = max(len(p) for p in perms) // bc["batch_size"]
        for i in range(n_steps):
            loss = 0.0
            for k, (it, bt, _, _) in enumerate(subjects):
                idx = perms[k][(i * bc["batch_size"]) % len(perms[k]):][: bc["batch_size"]]
                loss = loss + torch.mean((heads(basis(it[idx]), k) - bt[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum, steps = loss_sum + loss.item(), steps + 1
        ev, evs = val_ev()
        rows.append(dict(epoch=epoch, train_loss=loss_sum / steps, val_ev=ev,
                         **{f"val_ev_s{k}": v for k, v in enumerate(evs)}, sec=time.time() - t0))
        print(f"epoch {epoch:3d} loss {loss_sum / steps:.4f} val_ev {ev:.4f} " + " ".join(f"{v:.4f}" for v in evs))
        if ev > best:
            best, best_epoch = ev, epoch
            torch.save(dict(cfg=basis.cfg, state_dict=basis.state_dict()), out_dir / "basis.pt")
        elif epoch - best_epoch >= bc["patience"]:
            break
    pd.DataFrame(rows).to_csv(out_dir / "log.csv", index=False)

    summary = dict(best_val_ev=best, best_epoch=best_epoch, epochs_run=len(rows),
                   basis_params=sum(p.numel() for p in basis.parameters()), train_sec=time.time() - t0)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
