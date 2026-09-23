"""Calibration of closed-form in-context baselines on single-trial betas (problem doc sec. 6, step 1).

    python scripts/calib_baselines.py --subject 1 --k 10 50 100 200 500 --seeds 5
    python scripts/calib_baselines.py --subject 1 --braincorl_ckpt references/BraInCoRL/checkpoints/CLIP_trained_on_s257.pth

Context: k distinct unique images of the test subject, one random trial each.
Test: common images; trial rep=2 is the held-out trial (coverage / NLL / CRPS),
the mean of the 3 trials is the target for Pearson r.

Model for one voxel: y = x^T w + e, w ~ N(0, tau2 I), e ~ N(0, sig2); x = centred, globally scaled
CLIP embedding plus a constant. With K = X X^T = U diag(s) U^T and z = U^T y, the evidence is
    log p(y) = -0.5 sum_i [ log(tau2 s_i + sig2) + z_i^2 / (tau2 s_i + sig2) ] + const
and is evaluated on a (tau2, sig2) grid.

Methods:
  null_n01       N(0, 1) for every voxel and image (betas are z-scored per voxel within session)
  ridge_homo     ridge, per-voxel lambda by 5-fold CV; sig2 = in-sample residual variance
  bayes_global   one (tau2, sig2) for all voxels, evidence summed over voxels of the context
  bayes_voxel    per-voxel (tau2, sig2) by evidence on the context; Gaussian plug-in predictive
  bayes_voxel_t  per-voxel lambda = sig2/tau2 from bayes_voxel, sig2 integrated out under p(sig2) ~ 1/sig2;
                 Student-t predictive with k degrees of freedom
  bayes_pop      per-voxel MAP of (log tau2, log sig2): evidence + Gaussian hyperprior fitted on the
                 training subjects (their per-voxel evidence estimates at n = --pop_n trials)
  braincorl_homo released BraInCoRL checkpoint on unit-norm features; sig2 = in-sample residual variance
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ridge_cv_brainwise, unit_norm  # noqa: E402

LEVELS = (0.5, 0.8, 0.9)
DEV = torch.device("cuda")


def load_subject(root: Path, s: int):
    return np.load(root / f"s{s}_trials.npy", mmap_mode="r"), np.load(root / f"s{s}_trial_meta.npz")


def draw_context(meta, k, rng):
    """k distinct unique images, one random trial each -> trial indices."""
    cand = np.flatnonzero(~meta["is_common"])
    cand = cand[rng.permutation(len(cand))]
    _, first = np.unique(meta["nsdid"][cand], return_index=True)
    return np.sort(cand[rng.permutation(first)[:k]])


def grids(E):
    tau2 = torch.logspace(-5, 0.5, 23, device=DEV, dtype=torch.float64) / E   # tau2 * E = prior signal variance
    sig2 = torch.logspace(-1.5, 0.7, 23, device=DEV, dtype=torch.float64)
    T, S = torch.meshgrid(tau2, sig2, indexing="ij")
    return T.reshape(-1), S.reshape(-1)                                       # (G,), (G,)


def evidence(s, Z, T, S, chunk=1024):
    """s (k,), Z (k, V) -> log evidence (V, G) up to a constant."""
    d = T[:, None] * s[None] + S[:, None]                                     # (G, k)
    out = []
    for i in range(0, Z.shape[1], chunk):
        z2 = Z[:, i:i + chunk].T ** 2                                         # (v, k)
        out.append(-0.5 * (d.log().sum(1)[None] + z2 @ (1.0 / d).T))
    return torch.cat(out)


def predictive(U, s, Z, Xc, Xq, tau2, sig2):
    """Per-voxel tau2, sig2 (V,) -> mean (V, Q), var (V, Q) of a new single trial."""
    lam = (sig2 / tau2)[:, None]                                              # (V, 1)
    A = (Xq @ Xc.T) @ U                                                       # (Q, k)
    inv = 1.0 / (s[None] + lam)                                               # (V, k)
    mean = (Z.T * inv) @ A.T
    quad = (Xq ** 2).sum(1)[None] - inv @ (A ** 2).T                          # x^T x - k*^T (K + lam I)^-1 k*
    return mean, sig2[:, None] + tau2[:, None] * quad, quad / lam


def gauss_scores(mean, var, y):
    sd = var.sqrt()
    z = (y - mean) / sd
    nd = torch.distributions.Normal(0.0, 1.0)
    nll = 0.5 * math.log(2 * math.pi) + sd.log() + 0.5 * z ** 2
    crps = sd * (z * (2 * nd.cdf(z) - 1) + 2 * nd.log_prob(z).exp() - 1 / math.sqrt(math.pi))
    cov = {p: z.abs() <= stats.norm.ppf(0.5 + p / 2) for p in LEVELS}
    return nll, crps, cov


def t_scores(mean, scale2, nu, y):
    sc = scale2.sqrt().cpu().numpy()
    z = ((y - mean).cpu().numpy()) / sc
    nll = -stats.t.logpdf(z, nu) + np.log(sc)
    # CRPS of a standardised Student-t (Jordan, Krueger, Lerch 2019, scoringRules), nu > 1
    c = 2 * np.sqrt(nu) / (nu - 1) * special.beta(0.5, nu - 0.5) / special.beta(0.5, nu / 2) ** 2
    crps = sc * (z * (2 * stats.t.cdf(z, nu) - 1) + 2 * stats.t.pdf(z, nu) * (nu + z ** 2) / (nu - 1) - c)
    cov = {p: torch.from_numpy(np.abs(z) <= stats.t.ppf(0.5 + p / 2, nu)) for p in LEVELS}
    return torch.from_numpy(nll), torch.from_numpy(crps), cov


def pearson_rows(a, b):
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def summarise(mean, var, scores, y_mean3, ncsnr):
    nll, crps, cov = scores
    r = pearson_rows(mean.float().cpu(), y_mean3.cpu()).numpy()       # 0 for a constant prediction
    row = dict(nll=nll.mean().item(), crps=crps.mean().item(), r=r.mean(), r_nc02=r[ncsnr > 0.2].mean(),
               # M2: 1 - NC is a decreasing function of ncsnr, so rho(var, 1 - NC) = -rho(var, ncsnr)
               rho_var_1mNC=-stats.spearmanr(var.mean(1).cpu().numpy(), ncsnr).statistic)
    for p, c in cov.items():
        c = c.float()
        row[f"cov{int(p * 100)}"] = c.mean().item()
        row[f"cov{int(p * 100)}_vox_mae"] = (c.mean(1) - p).abs().mean().item()   # mean over voxels of |coverage_v - p|
    return row, dict(nll=nll.mean(1).cpu().numpy(), cov90=cov[0.9].float().mean(1).cpu().numpy(), r=r)


def fit_hyperprior(root, subjects, feat, pop_n, pop_vox, T, S, rng):
    th = []
    for s in subjects:
        tr, meta = load_subject(root, s)
        idx = draw_context(meta, pop_n, rng)
        vox = np.sort(rng.permutation(tr.shape[1])[:pop_vox])
        X = feat(meta["nsdid"][idx])
        s_, U = torch.linalg.eigh(X @ X.T)
        Z = U.T @ torch.from_numpy(np.asarray(tr[idx])[:, vox]).to(DEV, torch.float64)
        g = evidence(s_.clamp_min(0), Z, T, S).argmax(1)
        th.append(torch.stack([T[g].log(), S[g].log()], 1))
    th = torch.cat(th)
    mu, cov = th.mean(0), torch.cov(th.T)
    print(f"[hyperprior] {len(th)} voxels from S{subjects}: mean log(tau2, sig2) = {mu.tolist()}, cov = {cov.tolist()}")
    return mu, torch.linalg.inv(cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/braincorl_nsdgeneral")
    ap.add_argument("--feats", default="feats_clip_b16_73k.npy")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--train_subjects", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50, 100, 200, 500])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pop_n", type=int, default=1000)
    ap.add_argument("--pop_vox", type=int, default=4000)
    ap.add_argument("--braincorl_ckpt", default=None)
    ap.add_argument("--braincorl_config", default="src/config/braincorl_clip_3090.yaml")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    train_subjects = a.train_subjects or [s for s in (1, 2, 5, 7) if s != a.subject]
    out_dir = Path(a.out_dir or f"outputs/calib/baselines_s{a.subject}")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(root / a.feats, mmap_mode="r")
    un = unit_norm(np.asarray(raw))
    mu_f, sd_f = un.mean(0), un.std()                                         # image statistics only (73k images)
    def feat(ids):                                                            # centred, scaled, plus a constant
        x = (un[ids] - mu_f) / sd_f
        return torch.from_numpy(np.c_[x, np.ones(len(x))]).to(DEV, torch.float64)
    E = un.shape[1]
    T, S = grids(E)

    trials, meta = load_subject(root, a.subject)
    ncsnr = np.load(root / f"s{a.subject}_voxel_meta.npz")["ncsnr"]
    # test images: common images with all 3 trials; rows ordered by nsdid
    ci = np.flatnonzero(meta["is_common"])
    ids, cnt = np.unique(meta["nsdid"][ci], return_counts=True)
    ids = ids[cnt == 3]
    rep_idx = [ci[(meta["rep"][ci] == r) & np.isin(meta["nsdid"][ci], ids)] for r in range(3)]
    rep_idx = [ix[np.argsort(meta["nsdid"][ix])] for ix in rep_idx]
    Y = [torch.from_numpy(np.asarray(trials[ix]).T.copy()).to(DEV) for ix in rep_idx]   # 3 x (V, Q)
    y_held, y_mean3 = Y[2].double(), (Y[0] + Y[1] + Y[2]) / 3
    Xq = feat(ids)
    print(f"[data] S{a.subject}: {trials.shape[1]} voxels, {len(ids)} test images, train subjects {train_subjects}")

    rng = np.random.default_rng(0)
    hp_mu, hp_prec = fit_hyperprior(root, train_subjects, feat, a.pop_n, a.pop_vox, T, S, rng)
    d = torch.stack([T.log(), S.log()], 1) - hp_mu                            # (G, 2)
    log_hp = -0.5 * ((d @ hp_prec) * d).sum(1)                                # (G,)

    model = None
    if a.braincorl_ckpt:
        from src.config import load_config
        from src.eval import predict_icl
        from src.model import build_model
        from src.model.checkpoint import load_model_weights
        model = build_model(load_config(a.braincorl_config, [])["module"]["model"])
        load_model_weights(model, a.braincorl_ckpt)
        model = model.to(DEV).eval()
        q_un = un[ids].astype(np.float32)

    rows = []
    for k in a.k:
        for seed in range(a.seeds):
            idx = draw_context(meta, k, np.random.default_rng(1000 * k + seed))
            Xc = feat(meta["nsdid"][idx])
            yc = torch.from_numpy(np.asarray(trials[idx])).to(DEV, torch.float64)      # (k, V)
            s, U = torch.linalg.eigh(Xc @ Xc.T)
            s = s.clamp_min(0)
            Z = U.T @ yc
            ev = evidence(s, Z, T, S)                                         # (V, G)
            out = {}

            m = torch.zeros_like(y_held)
            out["null_n01"] = (m, torch.ones_like(m), gauss_scores(m, torch.ones_like(m), y_held))

            g = ev.sum(0).argmax().expand(ev.shape[0])
            m, v, _ = predictive(U, s, Z, Xc, Xq, T[g], S[g])
            out["bayes_global"] = (m, v, gauss_scores(m, v, y_held))

            g = ev.argmax(1)
            m, v, q = predictive(U, s, Z, Xc, Xq, T[g], S[g])
            out["bayes_voxel"] = (m, v, gauss_scores(m, v, y_held))
            lam = (S[g] / T[g])[:, None]
            s2 = ((Z.T ** 2) * lam / (s[None] + lam)).mean(1, keepdim=True)   # (1/k) y^T (K/lam + I)^-1 y
            sc2 = s2 * (1 + q)
            out["bayes_voxel_t"] = (m, sc2 * k / (k - 2) if k > 2 else sc2, t_scores(m, sc2, k, y_held))

            g = (ev + log_hp[None]).argmax(1)
            m, v, _ = predictive(U, s, Z, Xc, Xq, T[g], S[g])
            out["bayes_pop"] = (m, v, gauss_scores(m, v, y_held))

            Xr, Xrq = Xc[:, :-1].float(), Xq[:, :-1].float()
            w = torch.cat([ridge_cv_brainwise(Xr, yc.T.float()[i:i + 4096]) for i in range(0, yc.shape[1], 4096)])
            ones = lambda x: torch.cat([x, torch.ones(len(x), 1, device=DEV)], 1)
            res = yc.T.float() - w @ ones(Xr).T
            m = (w @ ones(Xrq).T).double()
            v = res.var(1, unbiased=False).clamp_min(1e-6)[:, None].double().expand_as(m)
            out["ridge_homo"] = (m, v, gauss_scores(m, v, y_held))

            if model is not None:
                c_un = un[meta["nsdid"][idx]].astype(np.float32)
                yc32 = yc.float().cpu().numpy()
                V = yc32.shape[1]
                both = np.concatenate([q_un, c_un])                           # query + context in one pass
                p, _ = predict_icl(model, c_un, yc32, both, np.zeros((len(both), V), np.float32),
                                   np.zeros((V, E + 1), np.float32), DEV, 128)
                m = p[:, :len(q_un)].double()
                res = yc.T - p[:, len(q_un):].double()
                v = res.var(1, unbiased=False).clamp_min(1e-6)[:, None].expand_as(m)
                out["braincorl_homo"] = (m, v, gauss_scores(m, v, y_held))

            for name, (m, v, sc) in out.items():
                row, vox = summarise(m, v, sc, y_mean3, ncsnr)
                rows.append(dict(method=name, subject=a.subject, k=k, seed=seed, **row))
                if seed == 0:
                    np.savez(out_dir / f"{name}_k{k}_seed0_voxel.npz", **vox)
                print(f"[{name}] k={k} seed={seed} " + " ".join(f"{a_}={b_:.4f}" for a_, b_ in row.items()), flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.groupby(["method", "k"]).mean(numeric_only=True).drop(columns=["seed", "subject"]).round(4))


if __name__ == "__main__":
    main()
