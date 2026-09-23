"""Option A fast stage: the population's continuous receptive fields are the candidates; no mask dictionary.

Training voxel j (gauss slow stage, latent.pt["rf"]) has its own Gaussian mask A_j (P,) and full-data coefficients C_j (K,)
in its own standardised pooled channels u_j(x) = (Z(x) A_j - mean_j) / std_j.  For a voxel v of the new subject, with n
support pairs and MNI coordinates only:
    B_v | j, C = F_j C + eps,   C ~ N(C_j, tau2 v I),   eps ~ N(0, sig2_v I),   p(j | v) = w_vj (normalised MNI kernel)
where F_j (n, K) are the support channels under mask j and v = mean over channels of the population variance of C_j.
Mask and matching are one joint latent j; the responsibilities r_vj are the exact posterior; tau2 (one scalar) and sig2_v are
estimated by type-II ML with the MacKay fixed point + secant of src/sflm/fast.py.  The prediction is
    y_v(x) = sum_j r_vj F_j^q(x) mu_vj = Z(x) . W_v - off_v,   W_v = sum_j r_vj A_j (mu_vj / std_j)^T  (P, K),
so the query cost is (Q, P K) @ (P K, V) instead of a per-(v, j) product.  Costs: the (V, J) responsibility matrix is stored
(13k x 30k floats = 1.6 GB); every E-step recomputes the per-j eigen-quantities in chunks of jchunk candidates.
"""

import time
from pathlib import Path

import numpy as np
import torch

from src.sflm.fast import DF_FLOOR, SIG2_FLOOR, TRAIN, _sig2_step
from src.sflm.latent import DATA, DEV, LatentMap, GridStore, gaussian_dictionary


@torch.inference_mode()
def build_cache_cont(H: int, path: Path, latent: str):
    """Latent maps Z (N, P, K) of every image of the fold (training subjects' unique, hold-out unique and common)."""
    ck = torch.load(latent, map_location=DEV)
    cfg = ck["cfg"]
    g = LatentMap(cfg["in_dim"], cfg["hidden"], cfg["K"], torch.zeros(cfg["in_dim"]), torch.ones(cfg["in_dim"])).to(DEV)
    g.load_state_dict(ck["state"])
    g.eval()
    store = GridStore()

    def lat(ids):
        return torch.cat([g(store(ids[i:i + 500])) for i in range(0, len(ids), 500)]).cpu()

    out = {"K": cfg["K"], "Z_tr": {s: lat(np.load(DATA / f"s{s}_nsdid_unique.npy")) for s in TRAIN[H]},
           "Z_u": lat(np.load(DATA / f"s{H}_nsdid_unique.npy")), "Z_q": lat(np.load(DATA / f"s{H}_nsdid_common.npy")),
           "rf": ck["rf"]}
    torch.save(out, path)
    return out


@torch.inference_mode()
def population_cont(H, cache, lam_pop: float, jchunk: int = 512, subjects=None):
    """Per training voxel j (ncsnr > 0.2): mask A_j (P,), channel mean/std (K,) over the subject's 9,000 images, full-data
    ridge coefficients C_j (K,) (lambda = lam_pop) and MNI xyz.  Returns dict of (J, .) tensors on DEV.
    subjects: which training subjects form the population (default TRAIN[H]; a subset for leave-one-training-subject-out)."""
    A, mean, std, C, xyz = [], [], [], [], []
    for s in (TRAIN[H] if subjects is None else subjects):
        good = np.load(DATA / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2
        mu, sig = cache["rf"][s]
        As = gaussian_dictionary(mu[good].to(DEV), sig[good].to(DEV))                       # (J_s, P)
        Z = cache["Z_tr"][s].to(DEV)                                                        # (N, P, K)
        B = torch.from_numpy(np.load(DATA / f"s{s}_betas_unique.npy")).to(DEV)[:, good]
        B = B - B.mean(0, keepdim=True)
        K = Z.shape[2]
        I = torch.eye(K, device=DEV)
        for a in range(0, As.shape[0], jchunk):
            Aj = As[a:a + jchunk]
            F = torch.einsum("npk,jp->jnk", Z, Aj)                                          # (Jc, N, K)
            m, sd = F.mean(1), F.std(1).clamp_min(1e-6)
            F = (F - m[:, None]) / sd[:, None]
            G = torch.einsum("jnk,jnl->jkl", F, F)
            rhs = torch.einsum("jnk,nj->jk", F, B[:, a:a + jchunk])
            C.append(torch.linalg.solve(G + lam_pop * I, rhs[:, :, None])[:, :, 0])
            mean.append(m)
            std.append(sd)
        A.append(As)
        xyz.append(torch.from_numpy(np.load(DATA / f"s{s}_coords_mni.npy").astype(np.float32)).to(DEV)[good])
        del Z, B
    return dict(A=torch.cat(A), mean=torch.cat(mean), std=torch.cat(std), C=torch.cat(C), xyz=torch.cat(xyz))


class EBcont:
    """Per-support-set state.  Z_s (n, P, K) support latent, Bc (n, V) centred support betas, Z_q (Q, P, K) query latent,
    pop = population_cont(...), vscale (float), xyz_v (V, 3), sigma (mm)."""

    @torch.inference_mode()
    def __init__(self, Z_s, Bc, Z_q, pop, vscale, xyz_v, sigma, jchunk=128):
        self.n, self.P, self.K = Z_s.shape
        self.V, self.J = Bc.shape[1], pop["A"].shape[0]
        self.Z_s, self.Bc, self.Z_q, self.pop, self.vscale, self.jchunk = Z_s, Bc, Z_q, pop, vscale, jchunk
        self.bb = (Bc ** 2).sum(0)
        # per-j eigen-quantities of the support design (recomputed once per support set, stored: J x K x K)
        d, Q, c = [], [], []
        for a in range(0, self.J, jchunk):
            F = self._F(a)
            G = torch.einsum("jnk,jnl->jkl", F, F)
            dj, Qj = torch.linalg.eigh(G)
            d.append(dj.clamp_min(0.0))
            Q.append(Qj)
            c.append(torch.einsum("jkl,jk->jl", Qj, pop["C"][a:a + jchunk]))                 # Q^T C_j
        self.d, self.Q, self.c = torch.cat(d), torch.cat(Q), torch.cat(c)
        self.cGc = (self.d * self.c ** 2).sum(1)
        # normalised log kernel weights: log Z_v
        pn = (pop["xyz"] ** 2).sum(1)
        self.xv2 = (xyz_v ** 2).sum(1, keepdim=True)
        self.xyz_v, self.pn, self.sigma = xyz_v, pn, sigma
        self.logZ = self._logZ(sigma)
        self.R = torch.empty(self.V, self.J, device=DEV)                 # responsibilities of the last E-step
        self.LP = torch.empty(self.V, self.J, device=DEV)                # log p(B_v | j) of the last E-step (sigma-free)
        self.n_estep = 0

    def _logZ(self, sigma):
        logZ = torch.full((self.V,), -float("inf"), device=DEV)
        for a in range(0, self.J, self.jchunk):
            logZ = torch.logaddexp(logZ, torch.logsumexp(self._logw_raw(a, sigma), 1))
        return logZ

    @torch.inference_mode()
    def set_sigma(self, sigma):
        self.sigma, self.logZ = sigma, self._logZ(sigma)

    @torch.inference_mode()
    def sigma_profile(self, grid):
        """Log evidence sum_v log sum_j w_vj(sigma) p(B_v | j) for each sigma in grid, at the (tau2, sig2) of the last E-step:
        only the kernel weights change, so no E-step is repeated.  Returns {sigma: evidence}."""
        out = {}
        for sg in grid:
            logZ = self._logZ(sg)
            ev = torch.full((self.V,), -float("inf"), device=DEV)
            for a in range(0, self.J, self.jchunk):
                sl = slice(a, a + self.jchunk)
                ev = torch.logaddexp(ev, torch.logsumexp(self.LP[:, sl] + self._logw_raw(a, sg) - logZ[:, None], 1))
            out[float(sg)] = ev.sum().item()
        return out

    def _F(self, a):
        """Standardised support channels of candidates a:a+jchunk, (Jc, n, K)."""
        sl = slice(a, a + self.jchunk)
        F = torch.einsum("npk,jp->jnk", self.Z_s, self.pop["A"][sl])
        return (F - self.pop["mean"][sl][:, None]) / self.pop["std"][sl][:, None]

    def _logw_raw(self, a, sigma=None):
        xj = self.pop["xyz"][a:a + self.jchunk]
        return -0.5 * (self.xv2 - 2 * self.xyz_v @ xj.T + self.pn[a:a + self.jchunk][None]) / (self.sigma if sigma is None else sigma) ** 2

    @torch.inference_mode()
    def estep_mstats(self, tau2, sig2):
        """Responsibilities self.R (V, J), loglik, and the M-statistics res, qgam, e_full, mk_num (V,) of src/sflm/fast.py."""
        n, K, V = self.n, self.K, self.V
        rho = sig2 / (tau2 * self.vscale)                                                      # (V,)
        for a in range(0, self.J, self.jchunk):
            sl = slice(a, a + self.jchunk)
            F = self._F(a)                                                                     # (Jc, n, K)
            T = torch.einsum("jnk,nv->jkv", F, self.Bc)                                        # F_j^T B  (Jc, K, V)
            s = torch.einsum("jkl,jkv->jlv", self.Q[sl], T)                                    # Q_j^T F_j^T B
            d = self.d[sl][:, :, None]
            c = self.c[sl][:, :, None]
            w = 1.0 / (rho[None, None] + d)                                                    # (Jc, K, V)
            logdet = n * torch.log(sig2)[None] + torch.log1p(d / rho[None, None]).sum(1)      # (Jc, V)
            quad = self.bb[None] - 2 * (s * c).sum(1) + self.cGc[sl][:, None] - (w * (s - d * c) ** 2).sum(1)
            self.LP[:, sl] = (-0.5 * (logdet + quad / sig2[None])).T
            self.R[:, sl] = self.LP[:, sl] + (self._logw_raw(a) - self.logZ[:, None])
        lse = torch.logsumexp(self.R, 1)
        self.R -= lse[:, None]
        self.R.exp_()
        res, qgam, e_full, mk_num = (torch.zeros(V, device=DEV) for _ in range(4))
        for a in range(0, self.J, self.jchunk):
            sl = slice(a, a + self.jchunk)
            F = self._F(a)
            T = torch.einsum("jnk,nv->jkv", F, self.Bc)
            s = torch.einsum("jkl,jkv->jlv", self.Q[sl], T)
            d = self.d[sl][:, :, None]
            c = self.c[sl][:, :, None]
            w = 1.0 / (rho[None, None] + d)
            r = self.R[:, sl].T                                                                # (Jc, V)
            mu = w * (rho[None, None] * c + s)
            dev = (w ** 2 * (s - d * c) ** 2).sum(1)                                           # ||mu - c_j||^2
            res += (r * (self.bb[None] - 2 * (s * mu).sum(1) + (d * mu ** 2).sum(1))).sum(0)
            qgam += (r * (d * w).sum(1)).sum(0)
            e_full += (r * (dev + sig2[None] * w.sum(1))).sum(0) / self.vscale
            mk_num += (r * dev).sum(0) / self.vscale
        self.n_estep += 1
        return lse.sum().item(), res, qgam, e_full, mk_num

    @torch.inference_mode()
    def predict(self, tau2, sig2):
        """(V, Q) posterior-mean prediction with the responsibilities of the last E-step."""
        rho = sig2 / (tau2 * self.vscale)
        W = torch.zeros(self.V, self.P, self.K, device=DEV)
        off = torch.zeros(self.V, device=DEV)
        for a in range(0, self.J, self.jchunk):
            sl = slice(a, a + self.jchunk)
            F = self._F(a)
            T = torch.einsum("jnk,nv->jkv", F, self.Bc)
            s = torch.einsum("jkl,jkv->jlv", self.Q[sl], T)
            d = self.d[sl][:, :, None]
            c = self.c[sl][:, :, None]
            mu = (rho[None, None] * c + s) / (rho[None, None] + d)
            m = torch.einsum("jkl,jlv->jkv", self.Q[sl], mu)                                   # back to channel basis
            r = self.R[:, sl].T
            ms = m / self.pop["std"][sl][:, :, None]
            W += torch.einsum("jv,jp,jkv->vpk", r, self.pop["A"][sl], ms)
            off += torch.einsum("jv,jk,jkv->v", r, self.pop["mean"][sl], ms)
        return torch.einsum("qpk,vpk->vq", self.Z_q, W) - off[:, None]


def eb_fit_mackay_cont(eb: EBcont, a, log):
    """eb_fit_mackay of src/sflm/fast.py on an EBcont (same constants: #9-#14)."""
    n, K, V = eb.n, eb.K, eb.V
    sig2 = eb.bb / n * a.sig2_init_scale
    tau2 = a.tau2_init
    counters = dict(n_floor=0, n_clamp=0, n_clip=0, n_guard=0)
    hist, sec, consec, prev, use_secant = [], [], 0, None, True
    for it in range(a.em_iters + 1):
        ll, res, qgam, e_full, mk_num = eb.estep_mstats(tau2, sig2)
        if it > 0 and ll < hist[-1]["loglik"] - 1.0 and getattr(a, "guard", "stop") == "fallback":
            # the secant step decreased the log-likelihood: return to the previous state and take a plain fixed-point step
            counters["n_guard"] += 1
            log(f"[ebc] it {it} tau2 {tau2:.5f} loglik {ll:.1f} < {hist[-1]['loglik']:.1f}: fallback to the fixed-point step")
            (tau2, sig2, ll, res, qgam, e_full, mk_num), use_secant = prev, False
            sec = sec[:-1]
        hist.append(dict(it=it, tau2=tau2, loglik=ll, sig2_med=sig2.median().item()))
        log(f"[ebc] it {it} tau2 {tau2:.5f} sig2 median {sig2.median().item():.4f} loglik {ll:.1f}")
        if it > 0:
            consec = consec + 1 if (abs(tau2 - hist[-2]["tau2"]) < a.tol * tau2
                                    or (getattr(a, "ll_tol", None) is not None and abs(ll - hist[-2]["loglik"]) < a.ll_tol)) else 0
            if hist[-1]["loglik"] < hist[-2]["loglik"] - 1.0:
                counters["n_guard"] += 1
                break
        if it == a.em_iters or consec >= a.conv_consec:
            break
        prev = (tau2, sig2, ll, res, qgam, e_full, mk_num)
        grad = e_full.sum().item() / (2 * tau2) - K * V / 2
        x = np.log(tau2)
        tau2_new = (mk_num.sum() / qgam.sum()).item()
        if use_secant and len(sec) >= 1 and abs(grad - sec[-1][1]) > 1e-12:
            x_prev, g_prev = sec[-1]
            slope = (grad - g_prev) / (x - x_prev)
            if slope < 0:
                raw = -grad / slope
                counters["n_clip"] += int(abs(raw) > 2.0)
                tau2_new = float(np.exp(x + float(np.clip(raw, -2.0, 2.0))))
        sec.append((x, grad))
        use_secant = True
        tau2 = tau2_new
        sig2 = _sig2_step(res, n - qgam, counters)
    pred = eb.predict(tau2, sig2)
    return pred, dict(tau2=tau2, hist=hist, counters=counters, iters=len(hist), converged=bool(consec >= a.conv_consec),
                      sig2_med=sig2.median().item(), n_estep=eb.n_estep, r_max_med=eb.R.max(1).values.median().item())
