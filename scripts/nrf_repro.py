"""NRF (Chen et al., ICLR 2026; references/NRF @ b950f91) on the paired few-shot protocol of the SFLM / BraInCoRL table.

Model classes, loss, Fourier features and hyper-parameters come from references/NRF (training_config.yaml). What differs
from the authors' pipeline, and why:
  data        betas, MNI coordinates, voxel set of data/braincorl_nsdgeneral (same recipe as NRF: z-score per session, mean
              over repeats; coordinates use the same index->mm matrix), so every voxel and image is paired with the table.
  images      the authors' S{s}_stimuli_227.h5py; CLIP ViT-B/16 is frozen, so its outputs are computed once (stage feats,
              fp16 on disk) instead of once per step.
  selection   the authors early-stop base training and finetuning on shared-1k images, which are the test set here.
              Base models: 8,400 / 600 split of the unique images (default_rng(0)). Finetuning: the last 20% of the
              support set is the early-stopping set, the first 80% is trained on.
  support     default_rng(100).permutation(9000), consecutive slices over n and rep, as src.eval.

Stages (run from the repo root):
  feats  --subjects 1 2 5 7        CLIP layer-3/6 tokens + image embedding for the 10,000 images of each subject
  base   --subjects 1 2 5 7        one single-subject NRF per subject (shared by the folds that use it as a source)
  ft     --holdout 1 2 5 7         finetune the 3 source models on each support set, save support / test predictions
  table  --holdout 1 2 5 7         voxel-wise regression ensemble (NRF 3.2), EV / Pearson on ncsnr > 0.2, results.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
NRF = REPO / "references/NRF"
sys.path.insert(0, str(NRF))
from model.Feature_merger import Feature_merger  # noqa: E402
from model.nrf_encoder import NRF_Encoder  # noqa: E402
from train.training_utils import compute_loss, gaussian_fourier_feature  # noqa: E402

CFG = yaml.safe_load((NRF / "training_config.yaml").read_text())
TRAIN = {1: [2, 5, 7], 2: [1, 5, 7], 5: [1, 2, 7], 7: [1, 2, 5]}
DEV = torch.device("cuda")


# ── data ──

def load_subject(a, s):
    root = Path(a.root)
    rows = np.load(root / f"s{s}_stim_rows.npz")
    xyz = np.load(root / f"s{s}_coords_mni.npy").astype(np.float32)
    xyz = np.stack([xyz[:, 0] / 182 + 92 / 182, xyz[:, 1] / 218 + 126 / 218, xyz[:, 2] / 182 + 72 / 182], 1)  # NRF dataloader
    return dict(rows=rows, coords=torch.from_numpy(xyz).to(DEV),
                early=np.load(Path(a.feats) / f"s{s}_early.npy", mmap_mode="r"),
                higher=np.load(Path(a.feats) / f"s{s}_higher.npy", mmap_mode="r"),
                unique=np.load(root / f"s{s}_betas_unique.npy", mmap_mode="r"),
                common=np.load(root / f"s{s}_betas_common.npy", mmap_mode="r"))


def take(d, kind, idx):
    """CLIP features (pinned fp16) and betas of rows idx of the unique / common set, loaded into RAM."""
    r = d["rows"][kind][idx]
    o = np.argsort(r)
    inv = np.argsort(o)
    early = torch.from_numpy(np.ascontiguousarray(d["early"][r[o]])[inv]).pin_memory()
    higher = torch.from_numpy(np.ascontiguousarray(d["higher"][r[o]])[inv])
    beta = torch.from_numpy(np.ascontiguousarray(d[kind][np.sort(idx)])[np.argsort(np.argsort(idx))])
    return early, higher, beta


# ── model ──

def build():
    fm, pe = CFG["feature_merger"], CFG["positional_embed"]["gaussian"]
    merger = Feature_merger(fm["early_feat_input_dim"], fm["num_patches"], fm["downsample_dim"]).to(DEV)
    mlp = NRF_Encoder(2 * fm["downsample_dim"] + fm["higher_feat_input_dim"], 2 * pe["num_features"], **CFG["model"]).to(DEV)
    b = np.random.default_rng(seed=42).normal(scale=pe["std"], size=(3, pe["num_features"])).astype(np.float32)  # get_embed_fn
    return merger, mlp, gaussian_fourier_feature(torch.tensor(b, device=DEV))


def forward(merger, mlp, embed, early, higher, coords):
    """early (B, 2, 196, 768), higher (B, 512), coords (B, V, 3) -> (B, V); NRFTrainer.run_model."""
    early, higher = early.to(DEV, torch.float32, non_blocking=True), higher.to(DEV, torch.float32)
    img = torch.cat([merger([early[:, 0], early[:, 1]]), higher], -1)
    pe = embed(coords)
    x = torch.cat([img[:, None].expand(-1, pe.shape[1], -1), pe], -1)
    return mlp(x.reshape(-1, x.shape[-1])).reshape(len(img), -1)


@torch.no_grad()
def predict(merger, mlp, embed, early, higher, coords, rows=32768):
    merger.eval(), mlp.eval()
    step = max(1, rows // len(coords))
    out = [forward(merger, mlp, embed, early[i:i + step], higher[i:i + step], coords[None].expand(len(early[i:i + step]), -1, -1))
           for i in range(0, len(early), step)]
    return torch.cat(out).float()


def pearson_cols(p, t):
    p, t = p - p.mean(0), t - t.mean(0)
    r = (p * t).sum(0) / (p.norm(dim=0) * t.norm(dim=0))
    return torch.nan_to_num(r)


def fit(a, merger, mlp, embed, coords, trn, val, out, max_epochs, tag):
    """NRFTrainer.train: AdamW, per-image voxel subsampling, early stopping on the median val Pearson."""
    tc = CFG["training"]
    opt = torch.optim.AdamW(list(mlp.parameters()) + list(merger.parameters()), lr=a.lr, weight_decay=tc["wdecay"])
    (early, higher, beta), (v_early, v_higher, v_beta) = trn, val
    beta, v_beta = beta.to(DEV), v_beta.to(DEV)
    best, best_ep, bad, start = -1.0, -1, 0, 0
    last = out / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=DEV)
        mlp.load_state_dict(ck["mlp"]), merger.load_state_dict(ck["merger"]), opt.load_state_dict(ck["opt"])
        best, best_ep, bad, start = ck["best"], ck["best_ep"], ck["bad"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng"]), torch.cuda.set_rng_state(ck["cuda_rng"])
    V, nv = len(coords), tc["num_voxels"]
    for ep in range(start, max_epochs):
        if bad >= tc["patience"]:
            break
        merger.train(), mlp.train()
        t0, losses = time.time(), []
        perm = torch.randperm(len(early))
        for i in range(0, len(perm), tc["batch_size"]):
            b = perm[i:i + tc["batch_size"]]
            opt.zero_grad()
            total = 0.0
            for j in range(0, len(b), a.micro or len(b)):      # both loss terms are means over images: the weighted sum of
                m = b[j:j + (a.micro or len(b))]               # micro-batch losses has the gradient of the full-batch loss
                sub = torch.rand(len(m), V, device=DEV).topk(min(nv, V), dim=1).indices      # 2,000 voxels per image
                pred = forward(merger, mlp, embed, early[m], higher[m], coords[sub])
                loss = compute_loss(pred, beta[m.to(DEV)[:, None], sub], tc["loss_type"]) * (len(m) / len(b))
                loss.backward()
                total += loss.item()
            opt.step()
            losses.append(total)
        r = pearson_cols(predict(merger, mlp, embed, v_early, v_higher, coords), v_beta).median().item()
        improved = r > best
        if improved:
            best, best_ep, bad = r, ep, 0
            torch.save(dict(mlp=mlp.state_dict(), merger=merger.state_dict(), epoch=ep, val_corr=r), out / "best.pt")
        else:
            bad += 1
        rec = dict(tag=tag, epoch=ep, loss=float(np.mean(losses)), val_corr=r, best=best, best_ep=best_ep, sec=time.time() - t0)
        print(json.dumps(rec), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if a.resume:
            torch.save(dict(mlp=mlp.state_dict(), merger=merger.state_dict(), opt=opt.state_dict(), epoch=ep, best=best,
                            best_ep=best_ep, bad=bad, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), last)
    ck = torch.load(out / "best.pt", map_location=DEV)
    mlp.load_state_dict(ck["mlp"]), merger.load_state_dict(ck["merger"])
    return best, best_ep


# ── stages ──

def stage_feats(a):
    import h5py
    from model.image_encoder import CLIPImageEncoder
    ie = CFG["image_embedding"]
    enc = CLIPImageEncoder(ie["model_name"], ie["selected_layers"]).to(DEV).eval()
    Path(a.feats).mkdir(parents=True, exist_ok=True)
    for s in a.subjects:
        with h5py.File(Path(a.stimuli) / f"S{s}_stimuli_227.h5py") as f:
            stim = f["stimuli"]
            N = len(stim)
            early = np.lib.format.open_memmap(Path(a.feats) / f"s{s}_early.npy", "w+", np.float16, (N, 2, 196, 768))
            higher = np.lib.format.open_memmap(Path(a.feats) / f"s{s}_higher.npy", "w+", np.float16, (N, 512))
            for i in range(0, N, 250):
                with torch.no_grad():
                    e, h = enc(torch.from_numpy(stim[i:i + 250]))
                early[i:i + 250], higher[i:i + 250] = e.half().cpu().numpy(), h.half().cpu().numpy()
                print(f"[feats] s{s} {i + 250}/{N} max|early| {e.abs().max().item():.1f}", flush=True)
            early.flush(), higher.flush()


def stage_base(a):
    for s in a.subjects:
        out = Path(a.out) / f"base_s{s}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "done.json").exists():
            continue
        torch.manual_seed(a.seed)
        d = load_subject(a, s)
        perm = np.random.default_rng(0).permutation(9000)
        va, tr = np.sort(perm[:600]), np.sort(perm[600:])
        merger, mlp, embed = build()
        best, ep = fit(a, merger, mlp, embed, d["coords"], take(d, "unique", tr), take(d, "unique", va), out, a.epochs, f"base_s{s}")
        e, h, beta = take(d, "common", np.arange(1000))
        pred = predict(merger, mlp, embed, e, h, d["coords"])
        r = pearson_cols(pred, beta.to(DEV))
        (out / "done.json").write_text(json.dumps(dict(val_corr=best, best_epoch=ep, test_median_pearson_all_voxels=r.median().item())))
        print(f"[base] s{s}: best val {best:.4f} at epoch {ep}; test median Pearson (all nsdgeneral voxels) {r.median().item():.4f}")


def support_sets(ns, reps):
    order, start = np.random.default_rng(100).permutation(9000), 0
    for n in ns:
        for rep in range(reps):
            yield n, rep, order[start:start + n]
            start += n


def stage_ft(a):
    for H in a.holdout:
        d = load_subject(a, H)
        test = take(d, "common", np.arange(1000))
        for n, rep, idx in support_sets(a.ns, a.reps):
            n_val = max(round(0.2 * n), 5)
            sup = take(d, "unique", idx)
            trn, val = tuple(x[:n - n_val] for x in sup), tuple(x[n - n_val:] for x in sup)
            for b in TRAIN[H]:
                out = Path(a.out) / f"ft_h{H}" / f"n{n}_rep{rep}_src{b}"
                if (out / "done.json").exists():
                    continue
                out.mkdir(parents=True, exist_ok=True)
                torch.manual_seed(a.seed)
                merger, mlp, embed = build()
                ck = torch.load(Path(a.out) / f"base_s{b}" / "best.pt", map_location=DEV)
                mlp.load_state_dict(ck["mlp"]), merger.load_state_dict(ck["merger"])
                if n == a.ns[0] and rep == 0:                                              # un-finetuned source model
                    np.save(out.parent / f"zeroshot_src{b}_test.npy", predict(merger, mlp, embed, test[0], test[1], d["coords"]).half().cpu().numpy())
                best, ep = fit(a, merger, mlp, embed, d["coords"], trn, val, out, a.ft_epochs, f"h{H}_n{n}_r{rep}_src{b}")
                np.save(out / "pred_support.npy", predict(merger, mlp, embed, sup[0], sup[1], d["coords"]).half().cpu().numpy())
                np.save(out / "pred_test.npy", predict(merger, mlp, embed, test[0], test[1], d["coords"]).half().cpu().numpy())
                (out / "best.pt").unlink()
                (out / "done.json").write_text(json.dumps(dict(val_corr=best, best_epoch=ep, n_train=n - n_val, n_val=n_val)))


def stage_table(a):
    import pandas as pd
    rows = []
    for H in a.holdout:
        root = Path(a.root)
        good = torch.from_numpy(np.load(root / f"s{H}_voxel_meta.npz")["ncsnr"] > 0.2).to(DEV)
        uniq = np.load(root / f"s{H}_betas_unique.npy", mmap_mode="r")
        y = torch.from_numpy(np.load(root / f"s{H}_betas_common.npy")).to(DEV)

        def score(method, n, rep, pred):
            ev = 1 - (y - pred).var(0, correction=0) / y.var(0, correction=0)
            r = pearson_cols(pred, y)
            np.save(Path(a.out) / f"ft_h{H}" / f"{method}_n{n}_rep{rep}_expl_var.npy", ev.cpu().numpy())
            rows.append(dict(holdout=H, method=method, n=n, rep=rep, ev=ev[good].mean().item(), pearson=r[good].mean().item(),
                             median_pearson_all=r.median().item()))

        zs = [Path(a.out) / f"ft_h{H}" / f"zeroshot_src{b}_test.npy" for b in TRAIN[H]]
        if all(p.exists() for p in zs):
            score("nrf_zeroshot_avg", 0, 0, torch.stack([torch.from_numpy(np.load(p)).float().to(DEV) for p in zs]).mean(0))
        for n, rep, idx in support_sets(a.ns, a.reps):
            dirs = [Path(a.out) / f"ft_h{H}" / f"n{n}_rep{rep}_src{b}" for b in TRAIN[H]]
            if not all((p / "done.json").exists() for p in dirs):
                continue
            ys = torch.from_numpy(np.ascontiguousarray(uniq[np.sort(idx)])[np.argsort(np.argsort(idx))]).to(DEV)
            ps = torch.stack([torch.from_numpy(np.load(p / "pred_support.npy")).float().to(DEV) for p in dirs])   # (3, n, V)
            pt = torch.stack([torch.from_numpy(np.load(p / "pred_test.npy")).float().to(DEV) for p in dirs])
            n_tr = json.loads((dirs[0] / "done.json").read_text())["n_train"]
            for name, sl in (("nrf_ft_ens", slice(0, n_tr)), ("nrf_ft_ens_allsup", slice(0, n))):
                X = torch.cat([ps[:, sl], torch.ones_like(ps[:1, sl])]).permute(2, 1, 0)                          # (V, n, 4)
                w = torch.linalg.lstsq(X, ys[sl].T[..., None]).solution                                           # voxel_regression.py, intercept
                score(name, n, rep, (torch.cat([pt, torch.ones_like(pt[:1])]).permute(2, 1, 0) @ w)[..., 0].T)
            score("nrf_ft_avg", n, rep, pt.mean(0))
            for b, p in zip(TRAIN[H], pt):
                score(f"nrf_ft_src{b}", n, rep, p)
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "results.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["holdout", "method", "n"])[["ev", "pearson", "median_pearson_all"]].mean().round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "base", "ft", "table"])
    ap.add_argument("--root", default="data/braincorl_nsdgeneral")
    ap.add_argument("--stimuli", default="data/nrf/stimuli")
    ap.add_argument("--feats", default="data/nrf/feats")
    ap.add_argument("--out", default="outputs/nrf")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--holdout", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--ns", type=int, nargs="+", default=[30, 100, 300, 1000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=CFG["training"]["learning_rate"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--ft_epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--micro", type=int, default=0, help="images per micro-batch (0 = whole batch); same gradient, less GPU memory")
    ap.add_argument("--cpu", action="store_true", help="table stage on the CPU (GPU shared with another job)")
    a = ap.parse_args()
    if a.cpu:
        global DEV
        DEV = torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = not a.no_tf32
    {"feats": stage_feats, "base": stage_base, "ft": stage_ft, "table": stage_table}[a.stage](a)


if __name__ == "__main__":
    main()
