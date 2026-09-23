"""Single-trial noise statistics for H2 / H3 of docs/few-shot-calibrated-encoding-problem.md (plan step 2).

    python scripts/calib_data_stats.py --subject 1

Noise samples. For an image with trials y1, y2, y3 the orthonormal contrasts
    c1 = (y1 - y2) / sqrt(2),   c2 = (y1 + y2 - 2 y3) / sqrt(6)
contain no image-driven signal, have covariance equal to the single-trial noise covariance and are
independent of each other if the noise is Gaussian and iid over repeats. If the noise has excess
kurtosis K and skewness G, both contrasts have excess kurtosis K / 2; c1 has skewness 0 and c2 has
skewness -0.408 G. Images are split 70 / 15 / 15 into fit / validation / test.

Model residuals. r = y_trial - yhat(x) on the common images, yhat from a ridge fit (per-voxel lambda,
5-fold CV) on the repeat means of the 9,000 unique images. These contain noise plus unexplained signal.

H2  (a) excess kurtosis, skewness per voxel
    (b) held-out NLL of a Student-t (nu, scale by ML on the fit split) minus Gaussian, per voxel
    (c) noise variance by quintile of yhat (common images), relative to the voxel's mean noise variance
Repeat effect: mean beta of repeat 1 / 2 / 3 per voxel; contrasts are centred with the fit-split mean
before (b) of H2 and (b) of H3.

H3  (a) noise correlation between voxels by distance, same / different `streams` ROI
    (b) held-out joint Gaussian NLL: (1 - a) S + a diag(S) (a by validation) minus diag(S); same marginals
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ridge_cv_brainwise, unit_norm  # noqa: E402

DEV = torch.device("cuda")
NC_BINS = [0.1, 0.2, 0.4, 0.7]
NC_LABELS = ["<0.1", "0.1-0.2", "0.2-0.4", "0.4-0.7", ">0.7"]
STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal", 5: "ventral", 6: "lateral", 7: "parietal"}
NUS = [2.5, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
DIST_BINS = [0, 2, 4, 8, 16, 32, 1e9]


def moments(x):
    """x (N, V) -> skewness, excess kurtosis per voxel."""
    z = x - x.mean(0)
    s = z.std(0, unbiased=False)
    return ((z / s) ** 3).mean(0), ((z / s) ** 4).mean(0) - 3


def t_minus_gauss(fit, test):
    """Per-voxel held-out NLL(t) - NLL(Gaussian), nats per sample; location = fit-split mean for both.
    Returns (diff, nu)."""
    fit, test = fit.double(), test.double()
    fit, test = fit - fit.mean(0), test - fit.mean(0)
    s2g = (fit ** 2).mean(0)
    nll_g = 0.5 * math.log(2 * math.pi) + 0.5 * s2g.log() + 0.5 * (test ** 2).mean(0) / s2g
    best_ll, best_nll, best_nu = None, None, None
    for nu in NUS:
        s2 = s2g.clone()
        for _ in range(30):                                   # EM for the scale of a t with fixed nu
            w = (nu + 1) / (nu + fit ** 2 / s2)
            s2 = (w * fit ** 2).mean(0)
        c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
        ll = lambda x: c - 0.5 * s2.log() - (nu + 1) / 2 * torch.log1p(x ** 2 / (nu * s2)).mean(0)
        l_fit, l_test = ll(fit), ll(test)
        if best_ll is None:
            best_ll, best_nll, best_nu = l_fit, -l_test, torch.full_like(s2, nu)
        else:
            m = l_fit > best_ll
            best_ll, best_nll, best_nu = torch.where(m, l_fit, best_ll), torch.where(m, -l_test, best_nll), \
                torch.where(m, torch.full_like(s2, nu), best_nu)
    gauss_better = -(0.5 * math.log(2 * math.pi) + 0.5 * s2g.log() + 0.5) > best_ll    # Gaussian ll on the fit split
    diff = torch.where(gauss_better, torch.zeros_like(nll_g), best_nll - nll_g)
    nu = torch.where(gauss_better, torch.full_like(best_nu, float("inf")), best_nu)
    return diff.float(), nu.float()


def joint_gain(fit, val, test):
    """(N, v) noise samples -> held-out [NLL(full shrunk) - NLL(diag)] in nats per voxel per sample, alpha."""
    fit, val, test = fit.double(), val.double(), test.double()
    mu = fit.mean(0)                                          # contrasts have a non-zero mean (repeat effect)
    fit, val, test = fit - mu, val - mu, test - mu
    S = fit.T @ fit / len(fit)
    d = torch.diagonal(S)

    def nll(x, C):
        L = torch.linalg.cholesky(C)
        q = (torch.linalg.solve_triangular(L, x.T, upper=False) ** 2).sum(0).mean()
        return 0.5 * (2 * torch.log(torch.diagonal(L)).sum() + q + x.shape[1] * math.log(2 * math.pi)) / x.shape[1]

    alphas = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]
    vals = [nll(val, (1 - a) * S + a * torch.diag(d)).item() for a in alphas]
    a = alphas[int(np.argmin(vals))]
    return (nll(test, (1 - a) * S + a * torch.diag(d)) - nll(test, torch.diag(d))).item(), a


def by_bin(x, lab, n):
    return [float(x[lab == b].mean()) for b in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/braincorl_nsdgeneral")
    ap.add_argument("--feats", default="feats_clip_b16_73k.npy")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n_corr_vox", type=int, default=4000)
    ap.add_argument("--n_joint_vox", type=int, default=1500)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    root, s = Path(a.root), a.subject
    out_dir = Path(a.out_dir or f"outputs/calib/stats_s{s}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    trials = np.load(root / f"s{s}_trials.npy", mmap_mode="r")
    meta = np.load(root / f"s{s}_trial_meta.npz")
    vm = np.load(root / f"s{s}_voxel_meta.npz")
    ncsnr, xyz, streams = vm["ncsnr"], vm["xyz"], vm["streams"]
    lab = np.digitize(ncsnr, NC_BINS)
    V = trials.shape[1]

    # trial index of rep 0 / 1 / 2 for every image, images ordered by nsdid
    o = np.lexsort((meta["rep"], meta["nsdid"])).reshape(-1, 3)             # (10000, 3)
    ids = meta["nsdid"][o[:, 0]]
    common = meta["is_common"][o[:, 0]]
    Y = torch.from_numpy(np.asarray(trials))                                # (30000, V) on CPU
    y1, y2, y3 = Y[o[:, 0]], Y[o[:, 1]], Y[o[:, 2]]
    c1 = (y1 - y2) / math.sqrt(2)
    c2 = (y1 + y2 - 2 * y3) / math.sqrt(6)
    perm = rng.permutation(len(ids))
    n_fit, n_val = int(0.7 * len(ids)), int(0.15 * len(ids))
    split = {"fit": perm[:n_fit], "val": perm[n_fit:n_fit + n_val], "test": perm[n_fit + n_val:]}
    noise = lambda k: torch.cat([c1[split[k]], c2[split[k]]]).to(DEV)      # (2 n_k, V)
    res = {"subject": s, "V": int(V), "voxels_per_ncsnr_bin": np.bincount(lab, minlength=5).tolist()}

    # repeat effect: mean over images of each repeat, per voxel (betas are z-scored within session)
    rep_mean = torch.stack([y1.mean(0), y2.mean(0), y3.mean(0)]).numpy()    # (3, V)
    res["repeat_mean_over_voxels"] = rep_mean.mean(1).tolist()
    res["repeat_mean_by_ncsnr_bin"] = [by_bin(rep_mean[r], lab, 5) for r in range(3)]
    res["repeat_mean_abs_rep3_minus_rep1"] = float(np.abs(rep_mean[2] - rep_mean[0]).mean())
    res["repeat_mean_null_sd"] = 1 / math.sqrt(len(ids))

    # ------------------------------------------------------------------ H2 (a, b) on noise contrasts
    sk1, ku1 = moments(c1.to(DEV)); sk2, ku2 = moments(c2.to(DEV))
    ku_noise = (ku1 + ku2).cpu().numpy()                                    # K = 2 * mean(K_c1, K_c2)
    sk_noise = (sk2 / -0.408).cpu().numpy()
    res["noise_var"] = by_bin(((c1 ** 2).mean(0) + (c2 ** 2).mean(0)).numpy() / 2, lab, 5)
    res["noise_excess_kurtosis"] = by_bin(ku_noise, lab, 5)
    res["noise_excess_kurtosis_all_median_p5_p95"] = np.percentile(ku_noise, [50, 5, 95]).tolist()
    res["noise_skewness"] = by_bin(sk_noise, lab, 5)
    res["contrast_c1_skewness_abs_mean"] = float(sk1.abs().mean())          # expected 0 up to sampling error
    d_t, nu = t_minus_gauss(noise("fit"), noise("test"))
    res["noise_t_minus_gauss_nll"] = by_bin(d_t.cpu().numpy(), lab, 5)
    res["noise_t_minus_gauss_nll_all"] = float(d_t.mean())
    res["noise_t_nu_median"] = float(nu[nu.isfinite()].median()) if nu.isfinite().any() else None
    res["noise_t_frac_gauss_selected"] = float((~nu.isfinite()).float().mean())
    # Gaussian reference for the kurtosis estimate: same shape, iid N(0, 1)
    g = torch.randn(c1.shape[0], 2000, device=DEV)
    res["gauss_ref_excess_kurtosis_mean_sd"] = [float(moments(g)[1].mean()), float(moments(g)[1].std())]

    # ------------------------------------------------------------------ H2 (a, b, c) on model residuals, common images
    feats = np.load(root / a.feats, mmap_mode="r")
    Xc = unit_norm(np.asarray(feats[ids[common]])).astype(np.float32)
    # ridge on the repeat means of the unique images (same recipe as preprocess_nsd.stage_gt)
    Xu = torch.from_numpy(unit_norm(np.asarray(feats[ids[~common]])).astype(np.float32)).to(DEV)
    yu = ((y1 + y2 + y3) / 3)[np.flatnonzero(~common)].T.contiguous()                        # (V, 9000) on CPU
    gt = torch.cat([ridge_cv_brainwise(Xu, yu[i:i + 4096].to(DEV)) for i in range(0, V, 4096)])   # (V, E+1)
    yhat = (torch.from_numpy(np.c_[Xc, np.ones(len(Xc), np.float32)]).to(DEV) @ gt.T)      # (1000, V)
    ci = np.flatnonzero(common)
    R = torch.cat([y1[ci], y2[ci], y3[ci]]).to(DEV) - yhat.repeat(3, 1)                     # (3000, V)
    m3 = ((y1 + y2 + y3) / 3)[ci].to(DEV)                                                   # sanity: r on repeat means
    zc = lambda x: (x - x.mean(0)) / x.std(0, unbiased=False)
    res["model_r_mean3_ncsnr_gt02"] = float((zc(yhat) * zc(m3)).mean(0).cpu().numpy()[ncsnr > 0.2].mean())
    R = R - R.mean(0)
    sk_r, ku_r = moments(R)
    res["model_resid_var"] = by_bin((R ** 2).mean(0).cpu().numpy(), lab, 5)
    res["model_resid_excess_kurtosis"] = by_bin(ku_r.cpu().numpy(), lab, 5)
    res["model_resid_skewness"] = by_bin(sk_r.cpu().numpy(), lab, 5)
    half = rng.permutation(len(ci)); h1, h2 = half[:500], half[500:]
    rows = lambda h: np.concatenate([h, h + len(ci), h + 2 * len(ci)])
    d_tr, nu_r = t_minus_gauss(R[rows(h1)], R[rows(h2)])
    res["model_resid_t_minus_gauss_nll"] = by_bin(d_tr.cpu().numpy(), lab, 5)
    res["model_resid_t_minus_gauss_nll_all"] = float(d_tr.mean())
    # (c) noise variance by quintile of yhat, per voxel, relative to the voxel's mean over quintiles
    nv = ((c1[ci] ** 2 + c2[ci] ** 2) / 2).to(DEV)                                           # (1000, V), E = noise var
    q = yhat.argsort(0).argsort(0) * 5 // len(ci)                                            # quintile 0..4 per voxel
    qv = torch.stack([(nv * (q == j)).sum(0) / (q == j).sum(0) for j in range(5)])           # (5, V)
    qv = (qv / qv.mean(0)).cpu().numpy()
    res["noise_var_by_yhat_quintile"] = {NC_LABELS[b]: qv[:, lab == b].mean(1).tolist() for b in range(5)}
    # sampling reference: same computation with the image order of nv permuted
    nvp = nv[torch.randperm(len(ci), device=DEV)]
    qp = torch.stack([(nvp * (q == j)).sum(0) / (q == j).sum(0) for j in range(5)])
    res["noise_var_by_yhat_quintile_permuted_ref"] = (qp / qp.mean(0)).cpu().numpy()[:, lab == 4].mean(1).tolist()

    # ------------------------------------------------------------------ H3 (a) noise correlation by distance
    vox = np.sort(rng.permutation(V)[:a.n_corr_vox])
    N = torch.cat([c1, c2])[:, vox].to(DEV).double()
    C = torch.corrcoef(N.T)
    D = torch.cdist(*(2 * [torch.from_numpy(xyz[vox]).to(DEV).double()]))
    same = torch.from_numpy(streams[vox][:, None] == streams[vox][None]).to(DEV)
    off = ~torch.eye(len(vox), dtype=torch.bool, device=DEV)
    res["noise_corr_by_distance_mm"] = {}
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        m = (D >= lo) & (D < hi) & off
        key = f"{lo}-{hi if hi < 1e8 else 'inf'}"
        res["noise_corr_by_distance_mm"][key] = dict(
            n_pairs=int(m.sum() // 2), mean=float(C[m].mean()),
            same_stream=float(C[m & same].mean()) if (m & same).any() else None,
            diff_stream=float(C[m & ~same].mean()) if (m & ~same).any() else None)
    res["noise_corr_null_sd"] = 1 / math.sqrt(N.shape[0])
    # signal correlation (repeat means) on the same pairs, for comparison at > 32 mm and < 4 mm
    M = ((y1 + y2 + y3) / 3)[:, vox].to(DEV).double()
    Cm = torch.corrcoef(M.T)
    res["mean_beta_corr_lt4mm_gt32mm"] = [float(Cm[(D < 4) & off].mean()), float(Cm[D >= 32].mean())]
    ev = torch.linalg.eigvalsh(C).flip(0)
    res["noise_corr_top10_eig_frac"] = (ev[:10] / ev.sum()).tolist()

    # ------------------------------------------------------------------ H3 (b) held-out joint NLL gain
    res["joint_nll_gain"] = {}
    groups = {"all": np.arange(V), **{n: np.flatnonzero(streams == k) for k, n in STREAMS.items()}}
    for name, idx in groups.items():
        if len(idx) < 200:
            continue
        sel = np.sort(rng.permutation(idx)[:a.n_joint_vox])
        f, v, t = (noise(k)[:, sel] for k in ("fit", "val", "test"))
        gain, alpha = joint_gain(f, v, t)
        # reference: voxel columns permuted independently over samples -> same marginals, no correlation
        fp = torch.stack([f[torch.randperm(len(f), device=DEV), j] for j in range(f.shape[1])], 1)
        gain_ref, _ = joint_gain(fp, v, t)
        res["joint_nll_gain"][name] = dict(n_vox=int(len(sel)), full_minus_diag=gain, alpha=alpha,
                                           permuted_fit_ref=gain_ref)
        print(f"[joint] {name}: {len(sel)} voxels, full - diag = {gain:.4f} nat/voxel/sample (alpha {alpha}), "
              f"permuted ref {gain_ref:.4f}", flush=True)

    json.dump(res, open(out_dir / "stats.json", "w"), indent=1)
    np.savez(out_dir / "voxel.npz", ku_noise=ku_noise, sk_noise=sk_noise, t_minus_gauss=d_t.cpu().numpy(),
             nu=nu.cpu().numpy(), ku_resid=ku_r.cpu().numpy())
    print(json.dumps({k: v for k, v in res.items() if k != "joint_nll_gain"}, indent=1))
    print(f"[done] S{s} -> {out_dir}")


if __name__ == "__main__":
    main()
