"""Training of the MedARC multi-feature encoder (src/model/medarc.py) on NSD images.

For each subject in `data.subjects`: train on its unique images minus `val_images` (seeded hold-out
used for early stopping on mean voxel Pearson, as in references/algonauts2025/train_feature_encoder.py),
test on its 1,000 shared images. With several subjects, every step sums the MSE of one batch per
subject (one decoder per subject; the shared decoder needs equal voxel counts). With `eval.ridge`,
ridge_cv_brainwise is fit on the same training images for every backbone and their concatenation.
model.pt holds the trained encoder; `MedarcTrunk.from_file` turns it into a basis for src.eval_lowrank
(new-subject evaluation).

    python -m src.train_medarc --config src/config/medarc_s1.yaml [train.lr=1.0e-3 ...]

Outputs in <out_dir>: model.pt, log.csv, results.csv, s<i>_<method>_{expl_var,pearson}.npy, config.yaml, summary.json
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .config import load_config
from .eval import roi_masks
from .model.medarc import MultiFeatureEncoder
from .utils import explained_variance, pearson, ridge_cv_brainwise, unit_norm


def load_data(dc, s, seed):
    """Returns (train, val, test) dicts with 'feats' (list of (N, D_i) unit-norm) and 'beta' (N, V)."""
    root = Path(dc["root"])
    feats_u = [unit_norm(np.load(root / bb / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
               for bb in dc["backbones"]]
    feats_c = [unit_norm(np.load(root / bb / "common_img_feats_normed.npy").astype(np.float32))
               for bb in dc["backbones"]]
    beta_u = np.load(root / f"s{s}_betas_unique.npy").astype(np.float32)
    beta_c = np.load(root / f"s{s}_betas_common.npy").astype(np.float32)
    ok = ~np.isnan(beta_c).any(1)                      # common images never shown (subjects < 40 sessions)
    perm = np.random.default_rng(seed).permutation(len(beta_u))
    va, tr = np.sort(perm[: dc["val_images"]]), np.sort(perm[dc["val_images"]:])
    split = lambda f, b, i: dict(feats=[x[i] for x in f], beta=b[i])
    return split(feats_u, beta_u, tr), split(feats_u, beta_u, va), split(feats_c, beta_c, ok)


def to_device(d, device):
    return dict(feats=[torch.from_numpy(x).to(device) for x in d["feats"]], beta=torch.from_numpy(d["beta"]).to(device))


def standardizer(train_feats):
    """Per-dimension z-score with training statistics (the reference feeds raw LLM hidden states)."""
    stats = [(x.mean(0), x.std(0) + 1e-6) for x in train_feats]
    return lambda feats: [(x - m) / s for x, (m, s) in zip(feats, stats)]


@torch.no_grad()
def predict(model, feats, subject, batch_size=1024):
    model.eval()
    out = torch.cat([model([x[i:i + batch_size] for x in feats], subject)
                     for i in range(0, feats[0].shape[0], batch_size)])
    model.train()
    return out


def scores(pred, beta):
    """Per-voxel EV and Pearson; pred, beta: (N, V)."""
    return explained_variance(pred.T, beta.T).cpu().numpy(), pearson(pred.T, beta.T).cpu().numpy()


def predict_ridge(X_tr, y_tr, X_te):
    w = ridge_cv_brainwise(X_tr, y_tr.T)                                      # (V, E+1)
    return torch.cat([X_te, torch.ones(X_te.shape[0], 1, device=X_te.device)], 1) @ w.T


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
    dc, mc, tc = cfg["data"], cfg["model"], cfg["train"]
    subjects = dc["subjects"]
    data = []                                          # per subject: (train, val, test) with normalised features
    for s in subjects:
        train, val, test = (to_device(d, device) for d in load_data(dc, s, cfg["seed"]))
        norm = standardizer(train["feats"]) if dc["standardize"] else (lambda feats: feats)
        for d in (train, val, test):
            d["x"] = norm(d["feats"])
        data.append((train, val, test))
        print(f"s{s}: train {train['beta'].shape[0]}, val {val['beta'].shape[0]}, test {test['beta'].shape[0]}, "
              f"voxels {train['beta'].shape[1]}")
    feat_dims = [x.shape[1] for x in data[0][0]["x"]]
    num_voxels = [tr["beta"].shape[1] for tr, _, _ in data]
    print(f"feat dims {feat_dims}")

    model = MultiFeatureEncoder(feat_dims, num_voxels, **mc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    bs = tc["batch_size"]
    best, best_epoch, best_state, rows, t0 = -float("inf"), -1, None, [], time.time()
    for epoch in range(tc["max_epochs"]):
        perms = [torch.randperm(tr["beta"].shape[0], device=device) for tr, _, _ in data]
        n_steps = max(len(p) for p in perms) // bs
        loss_sum = 0.0
        for i in range(n_steps):
            loss = 0.0
            for k, (tr, _, _) in enumerate(data):
                idx = perms[k][(i * bs) % len(perms[k]):][:bs]
                loss = loss + F.mse_loss(model([x[idx] for x in tr["x"]], k), tr["beta"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        val = [scores(predict(model, va["x"], k), va["beta"]) for k, (_, va, _) in enumerate(data)]
        r_mean = float(np.mean([r.mean() for _, r in val]))
        rows.append(dict(epoch=epoch, train_loss=loss_sum / n_steps, val_pearson=r_mean,
                         val_ev=float(np.mean([ev.mean() for ev, _ in val])),
                         **{f"val_pearson_s{s}": float(r.mean()) for s, (_, r) in zip(subjects, val)},
                         sec=time.time() - t0))
        print(f"epoch {epoch:3d} loss {loss_sum / n_steps:.4f} val_r {r_mean:.4f} "
              + " ".join(f"{r.mean():.4f}" for _, r in val))
        if r_mean > best:
            best, best_epoch, best_state = r_mean, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= tc["patience"]:
            break
    train_sec = time.time() - t0
    pd.DataFrame(rows).to_csv(out_dir / "log.csv", index=False)
    model.load_state_dict(best_state)
    torch.save(dict(cfg=cfg, state_dict=best_state, epoch=best_epoch, feat_dims=feat_dims, num_voxels=num_voxels),
               out_dir / "model.pt")

    res = []
    for k, (s, (train, _, test)) in enumerate(zip(subjects, data)):
        preds = {"medarc": predict(model, test["x"], k)}
        if cfg["eval"]["ridge"]:
            Xtr_n, Xte_n = train["feats"], test["feats"]   # ridge on unit-norm features, as for s{i}_gt.npy
            for bb, a, b in zip(dc["backbones"], Xtr_n, Xte_n):
                preds[f"ridge_{bb}"] = predict_ridge(a, train["beta"], b)
            if len(Xtr_n) > 1:
                preds["ridge_concat"] = predict_ridge(torch.cat(Xtr_n, 1), train["beta"], torch.cat(Xte_n, 1))
                torch.cuda.empty_cache()
        rois = roi_masks(Path(dc["root"]) / f"s{s}_voxel_meta.npz")
        for method, pred in preds.items():
            ev, r = scores(pred, test["beta"])
            np.save(out_dir / f"s{s}_{method}_expl_var.npy", ev)
            np.save(out_dir / f"s{s}_{method}_pearson.npy", r)
            row = dict(method=method, subject=s)
            row.update({f"ev_{c}": float(ev[m].mean()) for c, m in rois.items()})
            row.update({f"r_{c}": float(r[m].mean()) for c, m in rois.items()})
            res.append(row)
    df = pd.DataFrame(res)
    df.to_csv(out_dir / "results.csv", index=False)
    print(df[["method", "subject", "ev_all", "ev_ncsnr>0.2", "r_all", "r_ncsnr>0.2"]].round(4).to_string(index=False))

    test = {int(r.subject): {k: v for k, v in r.items() if k not in ("method", "subject")}
            for _, r in df[df.method == "medarc"].iterrows()}
    summary = dict(best_val_pearson=best, best_epoch=best_epoch, epochs_run=len(rows), params=n_params,
                   train_sec=train_sec, test=test)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
