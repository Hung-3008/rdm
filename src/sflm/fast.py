"""SFLM-EB: the closed-form fast stage of SFLM on a new subject (docs/sflm_eb_experiments_2026-09-23.md, part 2).

One generative model per voxel v of the new subject, with n support pairs and MNI coordinates only:
    B_v | m, j, C = F_m C + eps,   C ~ N(C_j^(m), tau2 v_m I),   eps ~ N(0, sig2_v I),   p(m, j | v) = pi_v(m) w_vj
F_m (n, K): pooled latent channels of mask m; C_j^(m): full-data coefficients of training voxel j; w_vj: normalised MNI
kernel (sigma_mm); v_m: population variance of mask m (unit of tau2); pi_v(m): Dirichlet-smoothed kernel average of the
training voxels' mask posteriors, concentration alpha fitted leave-one-training-subject-out (fit_alpha).
Mask and matching weights are the exact posterior responsibilities under the marginal likelihood (no per-voxel ridge);
tau2 (one scalar) and sig2_v are estimated by type-II ML on the support set with the MacKay fixed point + secant step on
log tau2 (eb_fit_mackay), started at tau2 = 1 v_m.  At n < K the likelihood profile is flat and the fixed point runs to the
boundary tau2 -> 0 (the ML-II answer; the region tau2 >~ 10 where sig2_v -> 0 is never entered).  4-fold result 2026-09-23:
above SFLM (scripts/sflm_pilot.py fast stage) in 48/48 paired cells, mean +0.0076 EV.

Pipeline (scripts/sflm_fast.py):
    cache = build_cache(H, path)                                  pooled channels of every image, from outputs/sflm/h<H>/latent.pt
    Cpop, qpop, xyz_pop, slices = population(H, cache, u_mean, u_std, lam_pop)
    alpha, _ = fit_alpha(qpop, xyz_pop, slices, sigma);  log_pi from kernel_sums
    eb = EB(Fs, Bc, Fq, Cpop, vscale, log_pi, xyz_v, xyz_pop, sigma);  pred, diag = eb_fit_mackay(eb, args, log)
Numerical constants (counted per cell in diag["counters"]): tau2 init 1 v_m, stop |d tau2|/tau2 < tol for conv_consec
iterations or em_iters cap, log-likelihood-decrease guard 1 nat, secant step clip +-2 on log tau2, sig2 floor 1e-6, df floor 1.
"""

import time
from pathlib import Path

import numpy as np
import torch

from src.sflm.latent import DATA, DEV, GridStore, LatentMap, mask_dictionary

TRAIN = {1: [2, 5, 7], 2: [1, 5, 7], 5: [1, 2, 7], 7: [1, 2, 5]}
N_ALL, REPS_ALL = [30, 100, 300, 1000], 3


@torch.inference_mode()
def ridge_fixed(F: torch.Tensor, B: torch.Tensor, lam: float):
    """Ridge with fixed lambda; returns C (K, V) and the per-voxel leave-one-out MSE (V,)."""
    evals, Q = torch.linalg.eigh(F.T @ F)
    FQ = F @ Q
    inv = 1.0 / (evals + lam)
    h = ((FQ ** 2) * inv).sum(1)
    C = Q @ (inv[:, None] * (FQ.T @ B))
    return C, (((B - F @ C) / (1.0 - h)[:, None]) ** 2).mean(0)


@torch.inference_mode()
def kernel_average(xyz_v: torch.Tensor, xyz_p: torch.Tensor, val_p: torch.Tensor, sigma: float, chunk: int = 1024):
    out = torch.empty(xyz_v.shape[0], val_p.shape[1], device=DEV)
    pn = (xyz_p ** 2).sum(1)
    for i in range(0, xyz_v.shape[0], chunk):
        xv = xyz_v[i:i + chunk]
        d2 = (xv ** 2).sum(1, keepdim=True) - 2 * xv @ xyz_p.T + pn[None]
        out[i:i + chunk] = torch.softmax(-0.5 * d2 / sigma ** 2, 1) @ val_p
    return out


@torch.inference_mode()
def build_cache(H: int, path: Path, latent: str = None, A: torch.Tensor = None):
    """Pooled channels u_m of every image of the fold, for the latent map `latent` (default outputs/sflm/h<H>/latent.pt) and
    the mask dictionary A (M, P) (default mask_dictionary())."""
    ck = torch.load(latent or f"outputs/sflm/h{H}/latent.pt", map_location=DEV)
    cfg = ck["cfg"]
    g = LatentMap(cfg["in_dim"], cfg["hidden"], cfg["K"], torch.zeros(cfg["in_dim"]), torch.ones(cfg["in_dim"])).to(DEV)
    g.load_state_dict(ck["state"])
    g.eval()
    A = (mask_dictionary() if A is None else A).to(DEV)
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

    def _pv(self, tau2, m):
        """Prior variance of mask m: tau2 (float, or (M,) tensor for per-mask tau2) in units of vscale[m]."""
        return (tau2[m] if torch.is_tensor(tau2) else tau2) * self.vscale[m]

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
                rho = s2 / self._pv(tau2, m)
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
            rho = s2 / self._pv(tau2, m)
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
            rho = sig2 / self._pv(tau2, m)
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
            consec = consec + 1 if (abs(tau2 - hist[-2]["tau2"]) < a.tol * tau2
                                    or (getattr(a, "ll_tol", None) is not None and abs(ll - hist[-2]["loglik"]) < a.ll_tol)) else 0
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


def support_offset(n: int, rep: int) -> int:
    """Start of the support slice in rng(100).permutation(9000): the canonical schedule of scripts/sflm_pilot.py
    (n in N_ALL, 3 consecutive repeats per n), so that any subset of cells uses the same indices as the full run."""
    return sum(REPS_ALL * m for m in N_ALL if m < n) + rep * n
