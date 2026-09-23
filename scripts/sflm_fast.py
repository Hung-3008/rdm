"""Fast stage of SFLM on a saved latent map (outputs/sflm/h<H>/latent.pt), docs/proposal-sflm-eb-2026-09-23.md.

mode sflm   the fast stage of scripts/sflm_pilot.py with its constants exposed (lam0, tau2_ratio, prior_floor, sigma_mm);
            with the defaults it reproduces the `sflm_bma` rows of outputs/sflm/h<H>/results.csv (same latent.pt, same
            support draws).
mode eb     SFLM-EB: one generative model  B_v | m, j, C = F_m C + eps,  C ~ N(C_j^(m), tau2 v_m I),  eps ~ N(0, sig2_v I),
            p(m, j | v) = pi_v(m) w_vj (MNI kernel).  Matching and mask weights are the exact posterior responsibilities
            under the marginal likelihood (no lam0, no C_hat); tau2 (one scalar, v_m = population variance of mask m as
            unit) and sig2_v are estimated by type-II ML on the support set; pi_v(m) is a Dirichlet-smoothed kernel average
            whose concentration alpha is fitted leave-one-training-subject-out.  Sweepable: K (through latent.pt), sigma_mm.
            --tau2 mackay   MacKay / EM fixed point on (tau2, sig2_v) jointly (the pilot solver; docs/survey §2 A1/A2).
            --tau2 search   bracketed root of d loglik / d log tau2 (scipy brentq) with sig2_v profiled by the MacKay
                            fixed point at each tau2, on a random voxel subsample; then sig2_v for all voxels at tau2_hat
                            (docs/survey/sflm_constant_removal_2026-09-23.md §2 A4).
Constant inventory (docs/sflm_eb_experiments_2026-09-23.md): #6 lam_pop, #9 tau2 init (mackay), #10 sig2 init, #11 stop
rule (mackay: tol + consecutive), #12 loglik guard (mackay), #13 sig2 floor, #14 df floor, #15 bracket, #16 search_xtol,
#17 search_vox (+seed), #18 inner_tol, #19 inner_cap.  D5 counters for #12-#14 and the secant clip are written per cell.

    python scripts/sflm_fast.py --holdout 1 --mode sflm --tag sflm_default
    python scripts/sflm_fast.py --holdout 1 --mode eb --tau2 mackay --em_iters 200 --num_ic 30 --repeats 1
    python scripts/sflm_fast.py --holdout 1 --mode eb --tau2 search
Outputs: outputs/sflm_eb/h<H>/{cache_pool.pt, results.csv, <tag>.log, <tag>_n=<n>_rep=<r>_ev.npy, <tag>_info.json}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "docs/popeb_v2/code")
from lowrank_basis import popeb  # noqa: E402
from sflm_pilot import DATA, DEV, STREAMS, GridStore, LatentMap, kernel_average, mask_dictionary, ridge_fixed  # noqa: E402
from src.utils.metrics import explained_variance, pearson  # noqa: E402

TRAIN = {1: [2, 5, 7], 2: [1, 5, 7], 5: [1, 2, 7], 7: [1, 2, 5]}


# ------------------------------------------------------------------------------------------------------ pooled channels
@torch.inference_mode()
def build_cache(H: int, path: Path):
    ck = torch.load(f"outputs/sflm/h{H}/latent.pt", map_location=DEV)
    cfg = ck["cfg"]
    g = LatentMap(cfg["in_dim"], cfg["hidden"], cfg["K"], torch.zeros(cfg["in_dim"]), torch.ones(cfg["in_dim"])).to(DEV)
    g.load_state_dict(ck["state"])
    g.eval()
    A = mask_dictionary().to(DEV)
    store = GridStore()

    def pool(ids):
        return torch.cat([torch.einsum("mp,bpk->bmk", A, g(store(ids[i:i + 500]))) for i in range(0, len(ids), 500)]).cpu()

    out = {"K": cfg["K"], "U_tr": {}, "t0": time.time()}
    for s in TRAIN[H]:
        out["U_tr"][s] = pool(np.load(DATA / f"s{s}_nsdid_unique.npy"))
        print(f"[cache] s{s} {tuple(out['U_tr'][s].shape)} {time.time() - out['t0']:.0f}s", flush=True)
    out["U_u"] = pool(np.load(DATA / f"s{H}_nsdid_unique.npy"))
    out["U_q"] = pool(np.load(DATA / f"s{H}_nsdid_common.npy"))
    torch.save(out, path)
    print(f"[cache] wrote {path} in {time.time() - out['t0']:.0f}s", flush=True)
    return out


# ------------------------------------------------------------------------------------------------------ population
@torch.inference_mode()
def population(H, cache, u_mean, u_std, lam_pop: float):
    """Full-data coefficients Cpop (M, K, J), mask posteriors qpop (J, M), coordinates xyz_pop (J, 3) and per-subject
    slices, for the training subjects' ncsnr > 0.2 voxels."""
    M = cache["U_u"].shape[1]
    Cpop, qpop, xyz, slices, start = [], [], [], {}, 0
    for s in TRAIN[H]:
        good = torch.from_numpy(np.load(DATA / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2).to(DEV)
        B = torch.from_numpy(np.load(DATA / f"s{s}_betas_unique.npy")).to(DEV)
        Bc = (B - B.mean(0, keepdim=True))[:, good]
        Un = (cache["U_tr"][s].to(DEV) - u_mean) / u_std
        Cs, ll = [], []
        for m in range(M):
            C, loo = ridge_fixed(Un[:, m], Bc, lam_pop)
            Cs.append(C)
            ll.append(-0.5 * Un.shape[0] * torch.log(loo))
        Cpop.append(torch.stack(Cs))
        qpop.append(torch.softmax(torch.stack(ll, 1), 1))
        xyz.append(torch.from_numpy(np.load(DATA / f"s{s}_coords_mni.npy").astype(np.float32)).to(DEV)[good])
        slices[s] = slice(start, start + int(good.sum()))
        start += int(good.sum())
        del B, Bc, Un
    return torch.cat(Cpop, 2), torch.cat(qpop), torch.cat(xyz), slices


@torch.inference_mode()
def kernel_sums(xyz_v, xyz_p, val_p, sigma, chunk=1024):
    """Unnormalised Gaussian-kernel sums: returns (sum_j w_vj val_j (V, M), sum_j w_vj (V,))."""
    num = torch.empty(xyz_v.shape[0], val_p.shape[1], device=DEV)
    den = torch.empty(xyz_v.shape[0], device=DEV)
    pn = (xyz_p ** 2).sum(1)
    for i in range(0, xyz_v.shape[0], chunk):
        xv = xyz_v[i:i + chunk]
        w = torch.exp(-0.5 * ((xv ** 2).sum(1, keepdim=True) - 2 * xv @ xyz_p.T + pn[None]) / sigma ** 2)
        num[i:i + chunk], den[i:i + chunk] = w @ val_p, w.sum(1)
    return num, den


@torch.inference_mode()
def fit_alpha(qpop, xyz_pop, slices, sigma):
    """Dirichlet concentration of the mask prior, leave-one-training-subject-out: maximise the log probability that the
    kernel-smoothed prior (population = other subjects) assigns to each held-out voxel's own full-data mask."""
    M = qpop.shape[1]
    grid = torch.tensor([0.0] + [10 ** e for e in np.arange(-2, 4.01, 0.5)], device=DEV)
    score = torch.zeros(len(grid), device=DEV)
    for s, sl in slices.items():
        keep = torch.ones(qpop.shape[0], dtype=torch.bool, device=DEV)
        keep[sl] = False
        num, den = kernel_sums(xyz_pop[sl], xyz_pop[keep], qpop[keep], sigma)
        star = qpop[sl].argmax(1)
        for i, a in enumerate(grid):
            pi = (num + a / M) / (den + a)[:, None]
            score[i] += torch.log(pi.gather(1, star[:, None]).clamp_min(1e-12)).sum()
    best = int(score.argmax())
    return float(grid[best]), {f"{float(a):g}": float(v) for a, v in zip(grid, score)}


# ------------------------------------------------------------------------------------------------------ SFLM-EB
SIG2_FLOOR = 1e-6                                                              # #13
DF_FLOOR = 1.0                                                                 # #14


class EB:
    """Per-support-set state of the SFLM-EB fast stage: per-mask eigen-quantities, the E-step and the M-statistics.
    Fs (n, M, K) support channels, Bc (n, V) centred support betas, Fq (Q, M, K) query channels, Cpop (M, K, J),
    vscale (M,) population variance per mask (unit of tau2), log_pi (V, M) log mask prior."""

    @torch.inference_mode()
    def __init__(self, Fs, Bc, Fq, Cpop, vscale, log_pi, xyz_v, xyz_pop, sigma, chunk=4096):
        self.n, self.M, self.K = Fs.shape
        self.V, self.J = Bc.shape[1], Cpop.shape[2]
        self.bb = (Bc ** 2).sum(0)                                             # (V,)
        self.Fq, self.vscale, self.log_pi = Fq, vscale, log_pi
        self.xyz_v, self.xyz_pop, self.sigma, self.chunk = xyz_v, xyz_pop, sigma, chunk
        self.pn = (xyz_pop ** 2).sum(1)
        self.eig = []
        for m in range(self.M):
            F = Fs[:, m]
            d, Q = torch.linalg.eigh(F.T @ F)
            d = d.clamp_min(0.0)
            s_hat = Q.T @ (F.T @ Bc)                                             # (K, V)
            c_hat = Q.T @ Cpop[m]                                                # (K, J)
            Y = torch.cat([c_hat, d[:, None] * c_hat, d[:, None] ** 2 * c_hat ** 2])   # (3K, J): quadratic form in one matmul
            Y2 = torch.cat([c_hat, c_hat ** 2]).T.contiguous()                   # (J, 2K): responsibility-weighted sums
            self.eig.append((d, Q, s_hat, (d[:, None] * c_hat ** 2).sum(0), Y, Y2))
        self.n_estep = 0

    def _logw(self, vi):
        """Normalised log MNI-kernel weights (v, J) for voxels vi; independent of m, tau2, sig2."""
        xv = self.xyz_v[vi]
        logw = -0.5 * ((xv ** 2).sum(1, keepdim=True) - 2 * xv @ self.xyz_pop.T + self.pn[None]) / self.sigma ** 2
        return logw - torch.logsumexp(logw, 1, keepdim=True)

    @torch.inference_mode()
    def estep(self, tau2, sig2, vi):
        """E-step on voxels vi (LongTensor, v entries): q (v, M) mask posteriors, loglik (float, sum over vi of
        log p(B_v)), RC (M, K, v) = sum_j r_vj c_hat_j, RC2 (M, K, v) = sum_j r_vj c_hat_j^2 (Q-basis)."""
        n, K, M = self.n, self.K, self.M
        v = vi.shape[0]
        logev = torch.empty(v, M, device=DEV)
        RC, RC2 = torch.empty(M, K, v, device=DEV), torch.empty(M, K, v, device=DEV)
        for a in range(0, v, self.chunk):
            sl = vi[a:a + self.chunk]
            o = slice(a, a + sl.shape[0])
            logw = self._logw(sl)
            s2, bb = sig2[sl], self.bb[sl]
            for m in range(M):
                d, Q, s_hat, cGc, Y, Y2 = self.eig[m]
                rho = s2 / (tau2 * self.vscale[m])
                w = 1.0 / (rho[:, None] + d[None])                               # (v, K)
                sh = s_hat[:, sl].T                                              # (v, K)
                logdet = n * torch.log(s2) + torch.log1p(d[None] / rho[:, None]).sum(1)
                X = torch.cat([-2 * sh, 2 * w * sh, -w], 1)                      # (v, 3K)
                rq = X @ Y                                                       # (v, J): -2 s.c + 2 (w s).(d c) - w.(d^2 c^2)
                rq += (bb - (w * sh ** 2).sum(1))[:, None] + cGc[None]           # = ||B - F C_j||^2 - quad
                rq *= -0.5 / s2[:, None]
                rq += logw - 0.5 * logdet[:, None]                               # log p(B_v | m, j) + log w_vj
                lse = torch.logsumexp(rq, 1)
                logev[o, m] = lse
                rq -= lse[:, None]
                rq.exp_()                                                        # responsibilities r_vj
                rc = rq @ Y2                                                     # (v, 2K)
                RC[m, :, o], RC2[m, :, o] = rc[:, :K].T, rc[:, K:].T
        self.n_estep += v / self.V
        logq = self.log_pi[vi] + logev
        return torch.softmax(logq, 1), torch.logsumexp(logq, 1).sum().item(), RC, RC2

    @torch.inference_mode()
    def mstats(self, tau2, sig2, vi, q, RC, RC2):
        """Per-voxel M-statistics from an E-step at (tau2, sig2): res = sum_m q ||B - F mu||^2, qgam = sum_m q gamma,
        e_full = sum_m q E||C - C_j||^2 / v_m (with tr S), mk_num = sum_m q ||mu - C_j||^2 / v_m (without tr S)."""
        n, M = self.n, self.M
        s2, bb = sig2[vi], self.bb[vi]
        res, qgam, e_full, mk_num = (torch.zeros(vi.shape[0], device=DEV) for _ in range(4))
        for m in range(M):
            d, Q, s_hat, cGc, Y, Y2 = self.eig[m]
            rho = s2 / (tau2 * self.vscale[m])
            w = 1.0 / (rho[:, None] + d[None])
            sh = s_hat[:, vi].T
            rc, rc2 = RC[m].T, RC2[m].T                                          # (v, K)
            gamma = (d * w).sum(1)
            dev = (w ** 2 * (sh ** 2 - 2 * sh * d * rc + d ** 2 * rc2)).sum(1)  # E||mu - C_j||^2
            s_mu = (w * sh * (rho[:, None] * rc + sh)).sum(1)                    # mu = w (rho c_j + s)
            mu_G_mu = (d * w ** 2 * (rho[:, None] ** 2 * rc2 + 2 * rho[:, None] * rc * sh + sh ** 2)).sum(1)
            res += q[:, m] * (bb - 2 * s_mu + mu_G_mu)
            qgam += q[:, m] * gamma
            e_full += q[:, m] * (dev + s2 * w.sum(1)) / self.vscale[m]
            mk_num += q[:, m] * dev / self.vscale[m]
        return res, qgam, e_full, mk_num

    @torch.inference_mode()
    def predict(self, tau2, sig2, q, RC):
        """Posterior-mean prediction (V, Q) for all voxels, given an E-step on all voxels."""
        pred = torch.zeros(self.V, self.Fq.shape[0], device=DEV)
        for m in range(self.M):
            d, Q, s_hat, *_ = self.eig[m]
            rho = sig2 / (tau2 * self.vscale[m])
            w = 1.0 / (rho[:, None] + d[None])
            mu = (w * (rho[:, None] * RC[m].T + s_hat.T)) @ Q.T                  # (V, K)
            pred += q[:, m:m + 1] * (mu @ self.Fq[:, m].T)
        return pred


def _sig2_step(res, den, counters):
    """sig2 <- res / den with the two floors of the pilot (#13, #14); counts the voxels that hit them."""
    counters["n_clamp"] = int((den < DF_FLOOR).sum())
    new = res / den.clamp_min(DF_FLOOR)
    counters["n_floor"] = int((new < SIG2_FLOOR).sum())
    return new.clamp_min(SIG2_FLOOR)


def eb_fit_mackay(eb: EB, a, log):
    """The pilot solver: joint MacKay (or EM) fixed point on (tau2, sig2_v) with a secant step on log tau2.
    Stops at em_iters, or after `conv_consec` consecutive iterations with |d tau2| / tau2 < tol (#11), or when the
    log-likelihood falls by more than 1 nat (#12, counted)."""
    n, K, V = eb.n, eb.K, eb.V
    allv = torch.arange(V, device=DEV)
    sig2 = eb.bb / n * a.sig2_init_scale                                       # #10
    tau2 = a.tau2_init                                                         # #9
    counters = dict(n_floor=0, n_clamp=0, n_clip=0, n_guard=0)
    hist, sec, consec = [], [], 0
    for it in range(a.em_iters + 1):
        q, ll, RC, RC2 = eb.estep(tau2, sig2, allv)
        hist.append(dict(it=it, tau2=tau2, loglik=ll, sig2_med=sig2.median().item(), grad=sec[-1][1] if sec else np.nan))
        log(f"[eb] it {it} tau2 {tau2:.5f} sig2 median {sig2.median().item():.4f} loglik {ll:.1f}"
            + (f" grad_prev {sec[-1][1]:.1f}" if sec else ""))
        if it > 0:
            consec = consec + 1 if abs(tau2 - hist[-2]["tau2"]) < a.tol * tau2 else 0
            if hist[-1]["loglik"] < hist[-2]["loglik"] - 1.0:
                counters["n_guard"] += 1
                break
        if it == a.em_iters or consec >= a.conv_consec:
            break
        res, qgam, e_full, mk_num = eb.mstats(tau2, sig2, allv, q, RC, RC2)
        grad = e_full.sum().item() / (2 * tau2) - K * V / 2                    # d loglik / d log tau2
        x = np.log(tau2)
        tau2_new = e_full.sum().item() / (K * V) if a.update == "em" else (mk_num.sum() / qgam.sum()).item()
        if a.secant and len(sec) >= 1 and abs(grad - sec[-1][1]) > 1e-12:
            x_prev, g_prev = sec[-1]
            slope = (grad - g_prev) / (x - x_prev)
            if slope < 0:                                                        # concave region: secant on the gradient
                raw = -grad / slope
                step = float(np.clip(raw, -2.0, 2.0))
                counters["n_clip"] += int(abs(raw) > 2.0)
                tau2_new = float(np.exp(x + step))
        sec.append((x, grad))
        tau2 = tau2_new
        sig2 = _sig2_step(res + (sig2 * qgam if a.update == "em" else 0.0), torch.full_like(res, float(n)) if a.update == "em"
                          else n - qgam, counters)
    pred = eb.predict(tau2, sig2, q, RC)
    return pred, dict(tau2=tau2, hist=hist, mask_mass=q.mean(0).tolist(), counters=counters, iters=len(hist),
                      converged=bool(consec >= a.conv_consec), sig2_med=sig2.median().item(), n_estep=eb.n_estep,
                      evals=np.nan, bracket_pos=np.nan, bracket_edge=False, inner_max=np.nan, inner_cap_hit=0)


def eb_fit_search(eb: EB, a, log):
    """A4: root of d loglik / d log tau2 on the bracket (#15) by brentq (#16), with sig2_v profiled at each tau2 by the
    MacKay fixed point (#18 inner tolerance, #19 cap, counted) on a random voxel subsample (#17); then sig2_v on all
    voxels at tau2_hat and one final E-step for the prediction."""
    n, K, V = eb.n, eb.K, eb.V
    allv = torch.arange(V, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(a.search_seed)
    vi = allv if a.search_vox <= 0 or a.search_vox >= V else torch.randperm(V, generator=g)[:a.search_vox].sort().values.to(DEV)
    sig2 = eb.bb / n * a.sig2_init_scale                                       # #10 (fixed point in sig2 given tau2)
    counters = dict(n_floor=0, n_clamp=0, n_clip=0, n_guard=0)
    hist, cache, inner_used = [], {}, []

    def profile(tau2, idx):
        """sig2[idx] <- MacKay fixed point at tau2; returns (grad wrt log tau2, loglik, q, RC) of the last E-step."""
        for k in range(a.inner_cap):
            q, ll, RC, RC2 = eb.estep(tau2, sig2, idx)
            res, qgam, e_full, _ = eb.mstats(tau2, sig2, idx, q, RC, RC2)
            new = _sig2_step(res, n - qgam, counters)
            delta = (torch.log(new) - torch.log(sig2[idx])).abs().max().item()
            sig2[idx] = new
            if delta < a.inner_tol:
                break
        inner_used.append(k + 1)
        grad = e_full.sum().item() / (2 * tau2) - K * idx.shape[0] / 2
        return grad, ll, q, RC

    def f(x):
        if x not in cache:
            grad, ll, *_ = profile(float(np.exp(x)), vi)
            cache[x] = grad
            hist.append(dict(x=x, tau2=float(np.exp(x)), grad=grad, loglik=ll, sig2_med=sig2[vi].median().item(), inner=inner_used[-1]))
            log(f"[eb] eval {len(hist)} tau2 {np.exp(x):.5f} grad {grad:.2f} loglik {ll:.1f} inner {inner_used[-1]}")
        return cache[x]

    lo, hi = np.log(a.bracket[0]), np.log(a.bracket[1])
    g_lo, g_hi = f(lo), f(hi)
    edge = g_lo * g_hi > 0
    if edge:                                                                   # no sign change: criterion (i) fails
        x_hat = lo if g_lo < 0 else hi
        log(f"[eb] no interior root: grad({a.bracket[0]:g}) = {g_lo:.2f}, grad({a.bracket[1]:g}) = {g_hi:.2f}; using the edge")
    else:
        x_hat = brentq(f, lo, hi, xtol=a.search_xtol)
    tau2 = float(np.exp(x_hat))
    evals = len(hist)
    # sig2 for every voxel at tau2_hat, then the prediction
    grad_all, ll_all, q, RC = profile(tau2, allv)
    pred = eb.predict(tau2, sig2, q, RC)
    log(f"[eb] tau2_hat {tau2:.5f} pos {(x_hat - lo) / (hi - lo):.3f} evals {evals} grad_all {grad_all:.2f} loglik_all {ll_all:.1f} "
        f"inner max {max(inner_used)} estep-equiv {eb.n_estep:.1f}")
    return pred, dict(tau2=tau2, hist=hist, mask_mass=q.mean(0).tolist(), counters=counters, iters=int(sum(inner_used)),
                      converged=not edge, sig2_med=sig2.median().item(), n_estep=eb.n_estep, evals=evals,
                      bracket_pos=float((x_hat - lo) / (hi - lo)), bracket_edge=bool(edge), inner_max=int(max(inner_used)),
                      inner_cap_hit=int(sum(k == a.inner_cap for k in inner_used)), grad_all=grad_all, loglik_all=ll_all)


def eb_fit_fixed(eb: EB, a, log):
    """Diagnostic (EV-versus-tau2 curve): sig2_v profiled at the given --tau2_fixed on all voxels, then predict."""
    a2 = argparse.Namespace(**{**vars(a), "bracket": [a.tau2_fixed, a.tau2_fixed], "search_vox": 0})
    n, K, V = eb.n, eb.K, eb.V
    allv = torch.arange(V, device=DEV)
    sig2 = eb.bb / n * a.sig2_init_scale
    counters = dict(n_floor=0, n_clamp=0, n_clip=0, n_guard=0)
    tau2, used = a.tau2_fixed, 0
    for k in range(a.inner_cap):
        q, ll, RC, RC2 = eb.estep(tau2, sig2, allv)
        res, qgam, e_full, _ = eb.mstats(tau2, sig2, allv, q, RC, RC2)
        new = _sig2_step(res, n - qgam, counters)
        delta = (torch.log(new) - torch.log(sig2)).abs().max().item()
        sig2, used = new, k + 1
        log(f"[eb] fixed tau2 {tau2:.5f} inner {k} loglik {ll:.1f} max|dlog sig2| {delta:.2e}")
        if delta < a.inner_tol:
            break
    grad = e_full.sum().item() / (2 * tau2) - K * V / 2
    q, ll, RC, RC2 = eb.estep(tau2, sig2, allv)
    pred = eb.predict(tau2, sig2, q, RC)
    return pred, dict(tau2=tau2, hist=[dict(loglik=ll, grad=grad)], mask_mass=q.mean(0).tolist(), counters=counters, iters=used,
                      converged=delta < a.inner_tol, sig2_med=sig2.median().item(), n_estep=eb.n_estep, evals=1,
                      bracket_pos=np.nan, bracket_edge=False, inner_max=used, inner_cap_hit=int(used == a.inner_cap),
                      grad_all=grad, loglik_all=ll)


# ------------------------------------------------------------------------------------------------------ main
N_ALL, REPS_ALL = [30, 100, 300, 1000], 3


def support_offset(n: int, rep: int) -> int:
    """Start of the support slice in rng(100).permutation(9000): the canonical schedule of scripts/sflm_pilot.py
    (n in N_ALL, 3 consecutive repeats per n), so that any subset of cells uses the same indices as the full run."""
    return sum(REPS_ALL * m for m in N_ALL if m < n) + rep * n


def write_rows(res_path: Path, rows):
    """Replace rows with the same (tag, n, rep) under a file lock, so concurrent jobs of one fold do not clobber."""
    import fcntl
    df = pd.DataFrame(rows)
    with open(res_path.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if res_path.exists():
            old = pd.read_csv(res_path)
            key = old.set_index(["tag", "n", "rep"]).index.isin(df.set_index(["tag", "n", "rep"]).index)
            df = pd.concat([old[~key], df])
        df.to_csv(res_path, index=False)
        fcntl.flock(lk, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--mode", choices=["sflm", "eb"], default="eb")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num_ic", type=int, nargs="+", default=N_ALL)
    ap.add_argument("--reps", type=int, nargs="+", default=list(range(REPS_ALL)), help="which repeats (support draws)")
    ap.add_argument("--sigma_mm", type=float, default=10.0)
    ap.add_argument("--lam_pop", type=float, default=1.0, help="#6 ridge penalty of the population fit, both modes")
    # mode sflm constants (pilot defaults)
    ap.add_argument("--lam0", type=float, default=10.0)
    ap.add_argument("--tau2_ratio", type=float, default=0.0227)
    ap.add_argument("--prior_floor", type=float, default=0.1)
    # mode eb
    ap.add_argument("--tau2", choices=["mackay", "search", "fixed"], default="mackay", help="solver for the shared tau2 (fixed = diagnostic)")
    ap.add_argument("--tau2_fixed", type=float, default=None, help="tau2 value for --tau2 fixed (units of v_m)")
    ap.add_argument("--alpha", type=float, default=None, help="override the fitted Dirichlet concentration")
    ap.add_argument("--sig2_init_scale", type=float, default=1.0, help="#10: sig2 init = scale * ||B_v||^2 / n")
    ap.add_argument("--chunk", type=int, default=4096, help="voxels per E-step chunk (memory only)")
    #   --tau2 mackay
    ap.add_argument("--em_iters", type=int, default=30, help="#11 iteration cap of the fixed point")
    ap.add_argument("--tol", type=float, default=1e-3, help="#11 relative tau2 change below which an iteration counts as converged")
    ap.add_argument("--conv_consec", type=int, default=5, help="#11 consecutive converged iterations to stop")
    ap.add_argument("--tau2_init", type=float, default=1.0, help="#9 tau2 init of the fixed point (units of v_m)")
    ap.add_argument("--update", choices=["mackay", "em"], default="mackay")
    ap.add_argument("--no_secant", dest="secant", action="store_false", help="plain fixed point without the secant step on log tau2")
    #   --tau2 search
    ap.add_argument("--bracket", type=float, nargs=2, default=[1e-4, 1e2], help="#15 tau2 bracket (units of v_m)")
    ap.add_argument("--search_xtol", type=float, default=1e-3, help="#16 brentq tolerance on log tau2")
    ap.add_argument("--search_vox", type=int, default=2000, help="#17 random voxel subsample for the search (0 = all)")
    ap.add_argument("--search_seed", type=int, default=0, help="#17 seed of the subsample")
    ap.add_argument("--inner_tol", type=float, default=1e-3, help="#18 max |d log sig2| below which the profile stops")
    ap.add_argument("--inner_cap", type=int, default=100, help="#19 cap on profile iterations (counted when hit)")
    a = ap.parse_args()
    H = a.holdout
    out = Path(a.out or f"outputs/sflm_eb/h{H}")
    out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or (a.mode if a.mode == "sflm" else f"eb_{a.tau2}_s{a.sigma_mm:g}")
    logf = open(out / f"{tag}.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"[args] {vars(a)}")
    cache_path = out / "cache_pool.pt"
    cache = torch.load(cache_path) if cache_path.exists() else build_cache(H, cache_path)
    U_all = torch.cat([cache["U_tr"][s] for s in TRAIN[H]]).to(DEV)
    u_mean, u_std = U_all.mean(0, keepdim=True), U_all.std(0, keepdim=True).clamp_min(1e-6)
    del U_all
    M = u_mean.shape[1]
    Cpop, qpop, xyz_pop, slices = population(H, cache, u_mean, u_std, a.lam_pop)
    log(f"[pop] Cpop {tuple(Cpop.shape)} lam_pop {a.lam_pop}")

    meta = np.load(DATA / f"s{H}_voxel_meta.npz")
    good = meta["ncsnr"] > 0.2
    xyz_v = torch.from_numpy(np.load(DATA / f"s{H}_coords_mni.npy").astype(np.float32)).to(DEV)
    B_u = torch.from_numpy(np.load(DATA / f"s{H}_betas_unique.npy")).to(DEV)
    true = torch.from_numpy(np.load(DATA / f"s{H}_betas_common.npy")).to(DEV).T
    U_u = (cache["U_u"].to(DEV) - u_mean) / u_std
    U_q = (cache["U_q"].to(DEV) - u_mean) / u_std
    info = dict(mode=a.mode, args=vars(a))
    alpha = np.nan

    if a.mode == "sflm":
        prior = (1 - a.prior_floor) * kernel_average(xyz_v, xyz_pop, qpop, a.sigma_mm) + a.prior_floor / M
        tau2 = [a.tau2_ratio * Cpop[m].var(1).mean().item() for m in range(M)]
    else:
        alpha, scores = fit_alpha(qpop, xyz_pop, slices, a.sigma_mm)
        if a.alpha is not None:
            alpha = a.alpha
        num, den = kernel_sums(xyz_v, xyz_pop, qpop, a.sigma_mm)
        log_pi = torch.log((num + alpha / M) / (den + alpha)[:, None])
        vscale = torch.tensor([Cpop[m].var(1).mean().item() for m in range(M)], device=DEV)
        info.update(alpha=alpha, alpha_scores=scores)
        log(f"[eb] alpha {alpha:g}")

    order = np.random.default_rng(100).permutation(B_u.shape[0])
    rows = []
    res_path = out / "results.csv"
    for n in a.num_ic:
        for rep in a.reps:
            start = support_offset(n, rep)
            idx = torch.from_numpy(order[start:start + n]).to(DEV)
            t0 = time.time()
            Bs = B_u[idx]
            Bm = Bs.mean(0, keepdim=True)
            Bc = Bs - Bm
            with torch.inference_mode():
                if a.mode == "sflm":
                    sig2 = torch.stack([ridge_fixed(U_u[idx, m], Bc, a.lam0)[1] for m in range(M)], 1)
                    q = torch.softmax(torch.log(prior) - 0.5 * n * torch.log(sig2), 1)
                    pred = torch.zeros_like(true)
                    for m in range(M):
                        C = popeb(U_u[idx, m], Bc, Cpop[m], tau2[m], a.lam0, spatial=(xyz_v, xyz_pop, a.sigma_mm))
                        pred += q[:, m:m + 1] * (U_q[:, m] @ C).T
                    pred += Bm.T
                    diag = dict(mask_mass=q.mean(0).tolist(), counters={})
                else:
                    eb = EB(U_u[idx], Bc, U_q, Cpop, vscale, log_pi, xyz_v, xyz_pop, a.sigma_mm, chunk=a.chunk)
                    pred, diag = dict(mackay=eb_fit_mackay, search=eb_fit_search, fixed=eb_fit_fixed)[a.tau2](eb, a, log)
                    pred += Bm.T
                    del eb
            torch.cuda.synchronize()
            sec = time.time() - t0
            ev, r = explained_variance(pred, true).cpu().numpy(), pearson(pred, true).cpu().numpy()
            np.save(out / f"{tag}_n={n}_rep={rep}_ev.npy", ev.astype(np.float32))
            row = dict(tag=tag, mode=a.mode, n=n, rep=rep, ev=float(ev[good].mean()), r=float(r[good].mean()),
                       ev_all=float(ev.mean()), sec=sec, tau2=diag.get("tau2", np.nan), tau2_pos=diag.get("bracket_pos", np.nan),
                       bracket_edge=diag.get("bracket_edge", False), alpha=alpha, sig2_med=diag.get("sig2_med", np.nan),
                       iters=diag.get("iters", np.nan), evals=diag.get("evals", np.nan), n_estep=diag.get("n_estep", np.nan),
                       converged=diag.get("converged", True), inner_max=diag.get("inner_max", np.nan),
                       inner_cap_hit=diag.get("inner_cap_hit", 0), lam_pop=a.lam_pop, sigma_mm=a.sigma_mm)
            row.update({k: diag["counters"].get(k, np.nan) for k in ("n_floor", "n_clamp", "n_clip", "n_guard")})
            row.update({f"ev_{k}": float(ev[good & np.isin(meta["streams"], v)].mean()) for k, v in STREAMS.items()})
            rows.append(row)
            info[f"diag_n={n}_rep={rep}"] = diag
            log(f"[{tag}] n={n} rep={rep} ev={row['ev']:.4f} r={row['r']:.4f} early={row['ev_early']:.4f} tau2={row['tau2']:.5f} "
                f"iters={row['iters']} floor/clamp/clip/guard={row['n_floor']}/{row['n_clamp']}/{row['n_clip']}/{row['n_guard']} {sec:.0f}s")
            write_rows(res_path, [row])
    with open(out / f"{tag}_info.json", "w") as f:
        json.dump(info, f, indent=1, default=float)
    print(pd.DataFrame(rows).groupby("n")[["ev", "r", "ev_early", "sec"]].mean().round(4))


if __name__ == "__main__":
    main()
