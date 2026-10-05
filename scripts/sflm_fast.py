"""Fast stage of SFLM on a saved latent map (outputs/sflm/h<H>/latent.pt), docs/proposal-sflm-eb-2026-09-23.md.
Model code lives in src/sflm (latent.py, fast.py); this script keeps the CLI, the SFLM control (mode sflm) and the diagnostic
solvers (--tau2 search / fixed / loso).

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
            --tau2 loso     tau2 by leave-one-training-subject-out ML on the population (fit_loso): the held-out training
                            subject's full-data C_v^(m) ~ sum_j w_vj N(C_j^(m), tau2 v_m I), EM with the statistics pooled
                            over the three held-out subjects; the support set only enters through sig2_v (profiled at that
                            tau2).  Well defined at every n (no n < K degeneracy).  --tau2 loso_m: one tau2_m per mask.
            --sigma_grid    choose sigma_mm on this grid by the LOSO log evidence (fit_alpha score + fit_loso score); the
                            hold-out subject enters no choice.
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
sys.path.insert(0, "docs/_archive/popeb_v2/code")
from lowrank_basis import popeb  # noqa: E402
from src.sflm.fast import (DF_FLOOR, N_ALL, REPS_ALL, SIG2_FLOOR, TRAIN, EB, _sig2_step, build_cache, eb_fit_mackay,  # noqa: E402
                          fit_alpha, kernel_average, kernel_sums, population, ridge_fixed, support_offset)
from src.sflm.latent import DATA, DEV, STREAMS  # noqa: E402
from src.utils.metrics import explained_variance, pearson  # noqa: E402


# ------------------------------------------------------------------------------------------------------ pooled channels


# ------------------------------------------------------------------------------------------------------ population


@torch.inference_mode()
def fit_loso(Cpop, qpop, xyz_pop, slices, sigma, vscale, share=True, iters=100, tol=1e-4, chunk=2048, log=print):
    """tau2 by leave-one-training-subject-out type-II ML on the population.  For each held-out training subject s and
    voxel v in s, the full-data coefficients C_v^(m) (K,) are modelled with the prior of the fast stage,
        C_v^(m) ~ sum_{j not in s} w_vj N(C_j^(m), tau2_m v_m I),   w_vj = normalised MNI kernel (sigma),
    weighted by the voxel's own full-data mask posterior q_v(m).  EM on tau2 with the sufficient statistics pooled over
    the three held-out subjects: tau2_m <- sum_v q_v(m) sum_j r_vj ||C_v - C_j||^2 / (v_m K sum_v q_v(m)), r_vj the
    responsibilities; share=True pools over m as well (one scalar in units of v_m, the fast stage's convention).
    Returns (tau2 (M,) tensor, info) with info = dict(score = LOSO log evidence sum_v sum_m q_v(m) log p(C_v^(m)) at the
    estimate, per_subject = tau2 estimated from each held-out subject alone (one final E-step), hist)."""
    M, K, J = Cpop.shape
    tau2 = torch.ones(M, device=DEV)
    pn = (xyz_pop ** 2).sum(1)
    hist = []
    for it in range(iters + 1):
        stat, cnt, score = torch.zeros(M, device=DEV), torch.zeros(M, device=DEV), 0.0
        stat_s = {}
        for s, sl in slices.items():
            keep = torch.ones(J, dtype=torch.bool, device=DEV)
            keep[sl] = False
            xyz_o, pn_o = xyz_pop[keep], pn[keep]
            Cj = [Cpop[m][:, keep].T.contiguous() for m in range(M)]                   # (Jo, K) per mask
            Cj2 = [(c ** 2).sum(1) for c in Cj]
            st_s, ct_s = torch.zeros(M, device=DEV), torch.zeros(M, device=DEV)
            for a in range(sl.start, sl.stop, chunk):
                b = min(a + chunk, sl.stop)
                xv = xyz_pop[a:b]
                logw = -0.5 * ((xv ** 2).sum(1, keepdim=True) - 2 * xv @ xyz_o.T + pn_o[None]) / sigma ** 2
                logw -= torch.logsumexp(logw, 1, keepdim=True)
                q = qpop[a:b]
                for m in range(M):
                    Cv = Cpop[m][:, a:b].T
                    D = ((Cv ** 2).sum(1, keepdim=True) - 2 * Cv @ Cj[m].T + Cj2[m][None]).clamp_min(0.0)   # (v, Jo)
                    pv = tau2[m] * vscale[m]
                    lg = logw - 0.5 * D / pv - 0.5 * K * torch.log(2 * np.pi * pv)
                    lse = torch.logsumexp(lg, 1)
                    score += (q[:, m] * lse).sum().item()
                    r = torch.exp(lg - lse[:, None])
                    st_s[m] += (q[:, m] * (r * D).sum(1)).sum() / vscale[m]
                    ct_s[m] += q[:, m].sum()
            stat += st_s
            cnt += ct_s
            stat_s[s] = (st_s, ct_s)
        new = (stat.sum() / (K * cnt.sum())).expand(M).clone() if share else stat / (K * cnt)
        delta = ((new - tau2).abs() / tau2).max().item()
        hist.append(dict(it=it, score=score, tau2=float(tau2.mean()), tau2_min=float(tau2.min()), tau2_max=float(tau2.max()), delta=delta))
        log(f"[loso] sigma {sigma:g} it {it} score {score:.1f} tau2 mean {tau2.mean():.5f} min {tau2.min():.5f} max {tau2.max():.5f} max|dtau2|/tau2 {delta:.2e}")
        if it == iters or delta < tol:
            break
        tau2 = new
    per_subject = {s: float((st.sum() / (K * ct.sum())) if share else (st / (K * ct)).mean()) for s, (st, ct) in stat_s.items()}
    return tau2, dict(score=score, per_subject=per_subject, hist=hist, iters=len(hist), converged=delta < tol)


# ------------------------------------------------------------------------------------------------------ SFLM-EB


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


def eb_fit_fixed(eb: EB, a, log, tau2=None):
    """sig2_v profiled (MacKay fixed point, #18 tolerance, #19 cap) at a fixed tau2 on all voxels, then predict.
    tau2: float (--tau2 fixed, the EV-versus-tau2 diagnostic) or an (M,) tensor / float from fit_loso (--tau2 loso*)."""
    n, K, V = eb.n, eb.K, eb.V
    allv = torch.arange(V, device=DEV)
    sig2 = eb.bb / n * a.sig2_init_scale
    counters = dict(n_floor=0, n_clamp=0, n_clip=0, n_guard=0)
    tau2, used = a.tau2_fixed if tau2 is None else tau2, 0
    t2s = float(tau2.mean()) if torch.is_tensor(tau2) else float(tau2)               # scalar summary for logs / grad
    for k in range(a.inner_cap):
        q, ll, RC, RC2 = eb.estep(tau2, sig2, allv)
        res, qgam, e_full, _ = eb.mstats(tau2, sig2, allv, q, RC, RC2)
        new = _sig2_step(res, n - qgam, counters)
        delta = (torch.log(new) - torch.log(sig2)).abs().max().item()
        sig2, used = new, k + 1
        log(f"[eb] fixed tau2 {t2s:.5f} inner {k} loglik {ll:.1f} max|dlog sig2| {delta:.2e}")
        if delta < a.inner_tol:
            break
    grad = e_full.sum().item() / (2 * t2s) - K * V / 2 if not torch.is_tensor(tau2) else np.nan
    q, ll, RC, RC2 = eb.estep(tau2, sig2, allv)
    pred = eb.predict(tau2, sig2, q, RC)
    return pred, dict(tau2=t2s, tau2_min=float(tau2.min()) if torch.is_tensor(tau2) else t2s,
                      tau2_max=float(tau2.max()) if torch.is_tensor(tau2) else t2s,
                      hist=[dict(loglik=ll, grad=grad)], mask_mass=q.mean(0).tolist(), counters=counters, iters=used,
                      converged=delta < a.inner_tol, sig2_med=sig2.median().item(), n_estep=eb.n_estep, evals=1,
                      bracket_pos=np.nan, bracket_edge=False, inner_max=used, inner_cap_hit=int(used == a.inner_cap),
                      grad_all=grad, loglik_all=ll)


# ------------------------------------------------------------------------------------------------------ main


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
    ap.add_argument("--latent", default=None, help="latent.pt (default outputs/sflm/h<H>/latent.pt)")
    ap.add_argument("--dict", default=None, help=".pt with the mask dictionary A (M, 49) (default: mask_dictionary(), 70 masks)")
    ap.add_argument("--lam_pop", type=float, default=1.0, help="#6 ridge penalty of the population fit, both modes")
    # mode sflm constants (pilot defaults)
    ap.add_argument("--lam0", type=float, default=10.0)
    ap.add_argument("--tau2_ratio", type=float, default=0.0227)
    ap.add_argument("--prior_floor", type=float, default=0.1)
    # mode eb
    ap.add_argument("--tau2", choices=["mackay", "search", "fixed", "loso", "loso_m"], default="mackay",
                    help="solver for tau2: mackay / search = type-II ML on the support set; fixed = diagnostic; "
                         "loso = LOSO ML on the population, one scalar (units v_m); loso_m = per mask")
    ap.add_argument("--tau2_fixed", type=float, default=None, help="tau2 value for --tau2 fixed (units of v_m)")
    ap.add_argument("--sigma_grid", type=float, nargs="*", default=None, help="choose sigma_mm on this grid by the LOSO log evidence")
    ap.add_argument("--loso_iters", type=int, default=100, help="EM iteration cap of fit_loso")
    ap.add_argument("--loso_tol", type=float, default=1e-4, help="fit_loso stops when max |d tau2| / tau2 < tol")
    ap.add_argument("--alpha", type=float, default=None, help="override the fitted Dirichlet concentration")
    ap.add_argument("--sig2_init_scale", type=float, default=1.0, help="#10: sig2 init = scale * ||B_v||^2 / n")
    ap.add_argument("--chunk", type=int, default=4096, help="voxels per E-step chunk (memory only)")
    #   --tau2 mackay
    ap.add_argument("--em_iters", type=int, default=30, help="#11 iteration cap of the fixed point")
    ap.add_argument("--tol", type=float, default=1e-3, help="#11 relative tau2 change below which an iteration counts as converged")
    ap.add_argument("--conv_consec", type=int, default=5, help="#11 consecutive converged iterations to stop")
    ap.add_argument("--ll_tol", type=float, default=None, help="#11 alt: an iteration also counts as converged when |d loglik| < ll_tol (nat)")
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
    cache_path = out / ("cache_pool.pt" if a.dict is None else f"cache_pool_{Path(a.dict).stem}.pt")
    A_dict = None if a.dict is None else torch.load(a.dict)["A"]
    cache = torch.load(cache_path) if cache_path.exists() else build_cache(H, cache_path, a.latent, A_dict)
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
    alpha, tau2_loso, loso_score = np.nan, None, np.nan

    def loso_fit(sigma):
        """LOSO fit at one sigma, cached on disk by (sigma, lam_pop, share)."""
        share = a.tau2 == "loso"
        cp = out / f"loso_s{sigma:g}_l{a.lam_pop:g}_{'shared' if share else 'permask'}.pt"
        if cp.exists():
            d = torch.load(cp)
            log(f"[loso] sigma {sigma:g} from {cp.name}: score {d['info']['score']:.1f} tau2 mean {d['tau2'].mean():.5f}")
            return d["tau2"].to(DEV), d["info"]
        t2, inf = fit_loso(Cpop, qpop, xyz_pop, slices, sigma, vscale, share=share, iters=a.loso_iters, tol=a.loso_tol, log=log)
        torch.save(dict(tau2=t2.cpu(), info=inf), cp)
        return t2, inf

    if a.mode == "eb" and a.sigma_grid:
        vscale = torch.tensor([Cpop[m].var(1).mean().item() for m in range(M)], device=DEV)
        curve = {}
        for sg in a.sigma_grid:
            al, sc = fit_alpha(qpop, xyz_pop, slices, sg)
            s_alpha = max(sc.values())
            if a.tau2.startswith("loso"):
                t2, inf = loso_fit(sg)
                s_ev = inf["score"]
            else:
                s_ev = 0.0
            curve[f"{sg:g}"] = dict(alpha=al, score_alpha=s_alpha, score_ev=s_ev, score=s_alpha + s_ev)
            log(f"[sigma] {sg:g} mm: alpha {al:g} score_alpha {s_alpha:.1f} score_ev {s_ev:.1f} total {s_alpha + s_ev:.1f}")
        best = max(curve, key=lambda k: curve[k]["score"])
        a.sigma_mm = float(best)
        info["sigma_curve"], info["sigma_sel"] = curve, a.sigma_mm
        log(f"[sigma] selected {a.sigma_mm:g} mm by the LOSO log evidence")

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
        if a.tau2.startswith("loso"):
            tau2_loso, inf = loso_fit(a.sigma_mm)
            loso_score = inf["score"]
            info.update(loso=inf, tau2_loso=tau2_loso.tolist())
            log(f"[eb] tau2 loso mean {tau2_loso.mean():.5f} min {tau2_loso.min():.5f} max {tau2_loso.max():.5f} "
                f"per subject {inf['per_subject']} score {loso_score:.1f} iters {inf['iters']} converged {inf['converged']}")

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
                    if a.tau2.startswith("loso"):
                        pred, diag = eb_fit_fixed(eb, a, log, tau2=tau2_loso)
                    else:
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
                       inner_cap_hit=diag.get("inner_cap_hit", 0), lam_pop=a.lam_pop, sigma_mm=a.sigma_mm,
                       tau2_min=diag.get("tau2_min", np.nan), tau2_max=diag.get("tau2_max", np.nan), loso_score=loso_score)
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
