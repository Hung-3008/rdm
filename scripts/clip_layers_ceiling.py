"""Full-data linear ceiling (ridge on the 9,000 unique images, EV on the 1,000 common images) for
multi-layer CLIP ViT-B/16 features, per subject and per feature variant.

    PYTHONPATH=. python scripts/clip_layers_ceiling.py --subjects 1 2 5 7 --out outputs/clip_layers/ceiling

Ridge is the dual (kernel) form of src.utils.ridge_cv_brainwise: same 5 contiguous folds, same lambda
grid, lambda on [X, 1] (bias penalised as there), per-voxel lambda by mean fold EV, final prediction =
mean of the fold solutions at the selected lambda. Features are standardised per dimension with the
statistics of all 73k images (stimulus-only), unit-normed per block, and blocks concatenated / sqrt(#blocks).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils import explained_variance, unit_norm

LAMBDAS = (1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5)


def kernel_ridge_cv_predict(X, y, Xq, lambdas=LAMBDAS, num_fold=5, chunk=8192):
    """X (n, E) train, y (n, V), Xq (q, E). Returns (pred_q (q, V), best_lambda (V,)) as ridge_cv_brainwise."""
    X, y, Xq = X.float(), y.float(), Xq.float()
    dev = X.device
    n = X.shape[0]
    Xa = torch.cat([X, torch.ones(n, 1, device=dev)], 1)
    Xqa = torch.cat([Xq, torch.ones(Xq.shape[0], 1, device=dev)], 1)
    K, Kq = Xa @ Xa.T, Xqa @ Xa.T                         # (n, n), (q, n)
    fold_size = n // num_fold
    idx = torch.arange(n, device=dev)
    folds = []
    for f in range(num_fold):
        te = idx[f * fold_size:(f + 1) * fold_size]
        tr = torch.cat([idx[:f * fold_size], idx[(f + 1) * fold_size:]])
        e, U = torch.linalg.eigh(K[tr][:, tr].double())    # float64: the block-standardised kernels are ill-conditioned
        folds.append((tr, te, e.float(), U.float()))
    V = y.shape[1]
    pred_q = torch.empty(Xq.shape[0], V, device=dev)
    best_all = torch.empty(V, dtype=torch.long, device=dev)
    for c0 in range(0, V, chunk):
        yc = y[:, c0:c0 + chunk]
        ev = torch.zeros(num_fold, len(lambdas), yc.shape[1], device=dev)
        pq = torch.zeros(num_fold, len(lambdas), Xq.shape[0], yc.shape[1], device=dev)
        for f, (tr, te, e, U) in enumerate(folds):
            Uy = U.T @ yc[tr]                                  # (n_tr, Vc)
            for li, lam in enumerate(lambdas):
                alpha = U @ (Uy / (e[:, None] + lam))          # (n_tr, Vc) dual coefficients
                ev[f, li] = explained_variance((K[te][:, tr] @ alpha).T, yc[te].T)
                pq[f, li] = Kq[:, tr] @ alpha
        best = ev.mean(0).argmax(0)                            # (Vc,)
        best_all[c0:c0 + chunk] = best
        sel = pq[:, best, :, torch.arange(yc.shape[1], device=dev)]   # advanced indices -> (Vc, num_fold, q)
        pred_q[:, c0:c0 + chunk] = sel.mean(1).T
    return pred_q, torch.tensor(lambdas, device=dev)[best_all]


def load_stats(path, ids=None, n_chunk=8192):
    """Per-(layer, dim) mean/std over all 73k images (or the sorted subset `ids`), from a (73000, L, D) memmap."""
    m = np.load(path, mmap_mode="r")
    ids = np.arange(m.shape[0]) if ids is None else ids
    s = np.zeros(m.shape[1:], np.float64); s2 = np.zeros(m.shape[1:], np.float64)
    for i in range(0, len(ids), n_chunk):
        x = np.asarray(m[ids[i:i + n_chunk]], np.float64)
        s += x.sum(0); s2 += (x ** 2).sum(0)
    mu = s / len(ids)
    sd = np.sqrt(np.maximum(s2 / len(ids) - mu ** 2, 1e-12))
    return mu.astype(np.float32), sd.astype(np.float32)


class Feats:
    def __init__(self, root, fit_subjects=None):
        """fit_subjects: fit standardisation and PCA only on the unique images of these subjects (default: all 73k images)."""
        self.root = Path(root)
        self.fit_ids = None if not fit_subjects else np.unique(np.concatenate(
            [np.load(self.root / f"s{s}_nsdid_unique.npy") for s in fit_subjects]))
        self.cls = np.load(self.root / "feats_clip_b16_cls_73k.npy", mmap_mode="r")
        self.mean = np.load(self.root / "feats_clip_b16_mean_73k.npy", mmap_mode="r")
        self.pooled = np.load(self.root / "feats_clip_b16_73k.npy", mmap_mode="r")
        self.stats = {"cls": load_stats(self.root / "feats_clip_b16_cls_73k.npy", self.fit_ids),
                      "mean": load_stats(self.root / "feats_clip_b16_mean_73k.npy", self.fit_ids)}
        self.pca = {}

    def block(self, kind, layer, ids):
        m = self.cls if kind == "cls" else self.mean
        mu, sd = self.stats[kind]
        sd_l = np.maximum(sd[layer], 0.01 * sd[layer].mean())      # floor: near-constant dims (e.g. CLS at layer 0) stay ~0
        x = (np.asarray(m[ids, layer], np.float32) - mu[layer]) / sd_l
        return unit_norm(x)

    def build(self, variant, ids):
        """variant: 'pooled512' | '<kind>_L<l>' | '<kinds>_<layers>' e.g. 'cls_all', 'clsmean_3_6_9_12' | 'pca<k>_<base variant>'
        | 'lpca<k>_<base>' (one PCA to k dims per (kind, layer) block) | 'lpcaJ<k>_<base>' (one PCA per layer, CLS and mean joint)
        | 'gpca<k>g<size>_<base>' (one PCA to k dims per group of `size` consecutive transformer layers, CLS and mean joint; level 0 joins the first group)."""
        if variant == "pooled512":
            return unit_norm(np.asarray(self.pooled[ids], np.float32))
        if variant.startswith(("lpca", "gpca")):
            head, base = variant.split("_", 1)
            joint = head.startswith("lpcaJ")
            blocks = self._blocks(base)
            if head.startswith("gpca"):
                k, size = (int(v) for v in head[4:].split("g"))
                chunk = lambda l: max(l - 1, 0) // size
                groups = [[b for b in blocks if chunk(b[1]) == c] for c in sorted({chunk(b[1]) for b in blocks})]
            else:
                k = int(head[5 if joint else 4:])
                groups = [[b for b in blocks if b[1] == l] for l in sorted({b[1] for b in blocks})] if joint else [[b] for b in blocks]
            out = []
            for g in groups:
                def build_g(i, g=g):
                    return np.concatenate([self.block(kd, l, i) for kd, l in g], 1) / np.sqrt(len(g))
                if tuple(g) not in self.pca:
                    self.pca[tuple(g)] = self._fit_pca(None, k_max=max(k, 256), build=build_g)   # cached per group, reused for other k
                mu, comps = self.pca[tuple(g)]
                assert comps.shape[0] >= k, "this block PCA was fitted with a smaller k"
                out.append(unit_norm(((build_g(ids) - mu) @ comps[:k].T).astype(np.float32)))
            return np.concatenate(out, 1) / np.sqrt(len(out))
        if variant.startswith("pca"):
            k, base = variant.split("_", 1)
            k = int(k[3:])
            if base not in self.pca:
                self.pca[base] = self._fit_pca(base, k_max=max(k, 1024))
            mu, comps = self.pca[base]
            z = (self.build(base, ids) - mu) @ comps[:k].T
            return unit_norm(z.astype(np.float32))
        blocks = [self.block(kd, l, ids) for kd, l in self._blocks(variant)]
        return np.concatenate(blocks, 1) / np.sqrt(len(blocks))

    @staticmethod
    def _blocks(variant):
        kinds, layers = variant.split("_", 1)
        kinds = ["cls", "mean"] if kinds == "clsmean" else [kinds]
        layers = list(range(13)) if layers == "all" else [int(v.lstrip("L")) for v in layers.split("_")]
        return [(kd, l) for kd in kinds for l in layers if not (kd == "cls" and l == 0)]   # CLS at layer 0 is constant

    def _fit_pca(self, base, k_max, n_fit=20000, seed=0, build=None):
        pool = np.arange(73000) if self.fit_ids is None else self.fit_ids
        ids = np.sort(np.random.default_rng(seed).choice(pool, min(n_fit, len(pool)), replace=False))
        Z = torch.from_numpy((build or (lambda i: self.build(base, i)))(ids)).cuda()
        mu = Z.mean(0, keepdim=True)
        Z -= mu
        e, U = torch.linalg.eigh(Z.T @ Z)                     # (E, E) covariance eigenvectors; SVD of (n, E) does not fit in memory
        comps = U[:, torch.argsort(e, descending=True)[:k_max]].T
        del Z; torch.cuda.empty_cache()
        return mu.cpu().numpy(), comps.contiguous().cpu().numpy()


def groups(meta):
    good = meta["ncsnr"] > 0.2
    g = {"all": np.ones(len(good), bool), "ncsnr>0.2": good,
         "early": good & (meta["streams"] == 1), "higher": good & (meta["streams"] > 1)}
    for c in ["faces", "places", "bodies", "words"]:
        g[c] = (meta[f"floc_{c}_label"] > 0) & (meta[f"floc_{c}_tval"] > 2) & good
    return g


def build_layout(feats: Feats, variant: str, subjects=(1, 2, 5, 7), name=None):
    """data/braincorl_nsdgeneral/clip_<variant>/ in the npy_multisubj layout (as scripts/medarc_holdout.sh layout):
    image features of this variant; betas, coords, gt and the support ordering symlinked to clip_b16/."""
    import os
    lay = feats.root / (name or f"clip_{variant}")
    lay.mkdir(exist_ok=True)
    ids_c = np.load(feats.root / "s1_nsdid_common.npy")
    np.save(lay / "common_img_feats_normed.npy", feats.build(variant, ids_c).astype(np.float32))
    links = ["test_time_icl_idx.json"]
    for s in subjects:
        ids_u = np.load(feats.root / f"s{s}_nsdid_unique.npy")
        assert np.array_equal(np.load(feats.root / f"s{s}_nsdid_common.npy"), ids_c)
        fu = feats.build(variant, ids_u).astype(np.float32)
        np.save(lay / f"s{s}_unique_img_feats_normed.npy", fu)
        # gt placeholder with the (V, E+1) shape that npy_multisubj checks; train_lowrank / eval_lowrank never read its values
        nv = np.load(feats.root / f"s{s}_betas_common.npy", mmap_mode="r").shape[1]
        if (lay / f"s{s}_gt.npy").is_symlink():
            (lay / f"s{s}_gt.npy").unlink()
        np.save(lay / f"s{s}_gt.npy", np.zeros((nv, fu.shape[1] + 1), np.float32))
        links += [f"s{s}_unique_nrn_feats_msk.npy", f"s{s}_common_nrn_feats_msk.npy", f"s{s}_coords.npy"]
    for f in links:
        if not os.path.lexists(lay / f):
            os.symlink(f"../clip_b16/{f}", lay / f)
    (lay / "README.txt").write_text("s*_gt.npy are zero placeholders (shape check only)"
                                    + (f"; standardisation + PCA fitted on the unique images of {lay.name}" if feats.fit_ids is not None else "")
                                    + "; betas/coords/support order symlink to clip_b16/\n")
    print(f"[layout] wrote {lay} (dims {np.load(lay / 'common_img_feats_normed.npy', mmap_mode='r').shape[1]})")


def add_gt(lay: Path, root: Path, subjects=(1, 2, 5, 7), chunk=4096):
    """Replace the zero gt of an existing layout by ridge weights on the 9,000 unique images (as preprocess_nsd
    --stage gt) and add the _val files (each subject's shared images) that BraInCoRL stage 3 validates on."""
    import os
    from src.utils import ridge_cv_brainwise
    fc = np.load(lay / "common_img_feats_normed.npy")
    for s in subjects:
        X = torch.from_numpy(np.load(lay / f"s{s}_unique_img_feats_normed.npy")).cuda()
        y = torch.from_numpy(np.load(root / f"s{s}_betas_unique.npy").T.copy()).cuda()   # (V, n_unique)
        gt = torch.cat([ridge_cv_brainwise(X, y[i:i + chunk]) for i in range(0, y.shape[0], chunk)]).cpu().numpy()
        np.save(lay / f"s{s}_gt.npy", gt.astype(np.float32))
        yc = np.load(root / f"s{s}_betas_common.npy")
        ok = ~np.isnan(yc[:, 0])
        pred = np.c_[fc, np.ones(len(fc), np.float32)] @ gt.T
        ev = 1 - np.var(yc[ok] - pred[ok], 0) / np.var(yc[ok], 0)
        good = np.load(root / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2
        print(f"[gt] s{s}: {gt.shape}, EV on common images: all {ev.mean():.4f}, ncsnr>0.2 {ev[good].mean():.4f}")
        np.save(lay / f"s{s}_unique_img_feats_normed_val.npy", fc)
        if not os.path.lexists(lay / f"s{s}_gt_val.npy"):
            os.symlink(f"s{s}_gt.npy", lay / f"s{s}_gt_val.npy")
        for f in [f"s{s}_unique_nrn_feats_msk_val.npy", f"s{s}_coords_val.npy"]:
            if not os.path.lexists(lay / f):
                os.symlink(f"../clip_b16/{f}", lay / f)
    with open(lay / "README.txt", "a") as f:
        f.write("s*_gt.npy replaced by ridge weights on the 9,000 unique images; _val files = each subject's shared images\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/braincorl_nsdgeneral")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--out", default="outputs/clip_layers/ceiling")
    ap.add_argument("--save_voxel_ev", action="store_true")
    ap.add_argument("--layout", default=None, help="only build the npy_multisubj layout for this variant and exit")
    ap.add_argument("--gt", action="store_true", help="with --layout: add real ridge gt + _val files to the layout (BraInCoRL training)")
    ap.add_argument("--fit_subjects", type=int, nargs="+", default=None,
                    help="with --layout: fit standardisation + PCA on these subjects' unique images; layout clip_<variant>_fit<subjects>")
    ap.add_argument("--holdout", type=int, default=None,
                    help="with --layout --fit_subjects: assert that the fit images share nothing with this subject's unique images or the common images")
    ap.add_argument("--fit_train", action="store_true",
                    help="ceiling mode: for each subject, fit standardisation + PCA on the unique images of the other --subjects only")
    a = ap.parse_args()
    if a.layout:
        name = f"clip_{a.layout}_fit{''.join(map(str, a.fit_subjects))}" if a.fit_subjects else None
        lay = Path(a.root) / (name or f"clip_{a.layout}")
        if not (lay / "common_img_feats_normed.npy").exists():   # never re-fit an existing layout (PCA signs may differ)
            feats = Feats(a.root, a.fit_subjects)
            if a.holdout is not None:
                test_ids = np.concatenate([np.load(feats.root / f"s{a.holdout}_nsdid_{t}.npy") for t in ("unique", "common")])
                assert feats.fit_ids is not None and len(np.intersect1d(feats.fit_ids, test_ids)) == 0, "fit images overlap the hold-out's images"
            build_layout(feats, a.layout, name=name)
        if a.gt:
            add_gt(lay, Path(a.root))
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    variants = a.variants or (["pooled512"] + [f"cls_L{l}" for l in range(1, 13)] + [f"mean_L{l}" for l in range(13)]   # CLS at layer 0 is image-independent
                              + ["cls_all", "mean_all", "clsmean_all", "clsmean_3_6_9_12", "clsmean_6_9_12",
                                 "pca512_clsmean_all", "pca1024_clsmean_all"])
    feats = Feats(a.root)
    dev = torch.device("cuda")
    rows = []
    for s in a.subjects:
        if a.fit_train:
            feats = Feats(a.root, [t for t in a.subjects if t != s])
            assert len(np.intersect1d(feats.fit_ids, np.concatenate([np.load(feats.root / f"s{s}_nsdid_{t}.npy") for t in ("unique", "common")]))) == 0
        ids_u = np.load(feats.root / f"s{s}_nsdid_unique.npy"); ids_c = np.load(feats.root / f"s{s}_nsdid_common.npy")
        yu = torch.from_numpy(np.load(feats.root / f"s{s}_betas_unique.npy")).to(dev)
        yc = np.load(feats.root / f"s{s}_betas_common.npy")
        ok = ~np.isnan(yc[:, 0])
        g = groups(np.load(feats.root / f"s{s}_voxel_meta.npz"))
        for v in variants:
            X = torch.from_numpy(feats.build(v, ids_u)).to(dev)
            Xq = torch.from_numpy(feats.build(v, ids_c)).to(dev)
            pred, lam = kernel_ridge_cv_predict(X, yu, Xq)
            ev = explained_variance(pred[ok].T, torch.from_numpy(yc[ok]).to(dev).T).cpu().numpy()
            row = dict(subject=s, variant=v, dims=int(X.shape[1]), **{f"ev_{k}": float(ev[m].mean()) for k, m in g.items()},
                       lam_median=float(lam.median()))
            rows.append(row)
            print(json.dumps(row))
            if a.save_voxel_ev:
                np.save(out / f"s{s}_{v}_ev.npy", ev)
            pd.DataFrame(rows).to_csv(out / "ceiling.csv", index=False)
            del X, Xq, pred; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index="variant", columns="subject", values="ev_all").round(4))
    print(df.pivot_table(index="variant", columns="subject", values="ev_ncsnr>0.2").round(4))


if __name__ == "__main__":
    main()
