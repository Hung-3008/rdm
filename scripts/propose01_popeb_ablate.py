"""One-factor ablations of PopEB (docs/propose/propose01.md 9.6): on S1 with the same support draws as
propose01 (paired with the BraInCoRL reruns), or nested on S2/S5/S7 (leave-one-training-subject-out,
population from the other two). Self-contained (re-implements PopEB with the extra knobs).
    python scripts/propose01_popeb_ablate.py s1 [variant ...]          # default: all variants
    python scripts/propose01_popeb_ablate.py nested variant ...
Env: ROOT=dir (data root holding clip_b16/ and s*_voxel_meta.npz; default data/braincorl).
"""
import os, sys, time, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from src.model.lowrank_basis import FunctionalBasis, ridge_loo, LowRankConfig
from src.dataset import SOURCES
from src.eval import roi_masks
from src.utils import explained_variance, unit_norm

ROOT = Path(os.environ.get("ROOT", "data/braincorl"))
dev = torch.device("cuda"); D = ROOT / "clip_b16"; OUT = Path("outputs/propose01")
MODE = sys.argv[1]; REQ = sys.argv[2:]
t0 = time.time(); LAMS = LowRankConfig().lambdas
BASES = {16: OUT / "sweep/K16_a1/basis.pt", 32: OUT / "sweep/K32_a1/basis.pt", 64: OUT / "sweep/K64_a1/basis.pt"}
basis_cache = {}
def get_basis(K):
    if K not in basis_cache:
        basis_cache[K] = FunctionalBasis.from_file(BASES[K], dev)
    return basis_cache[K]

# ------------------------------------------------------------------ data
ALL = [1, 2, 5, 7]; TR = [2, 5, 7]
raw = {}
for s in ALL:
    raw[s] = dict(img=torch.from_numpy(unit_norm(np.load(D / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))).to(dev),
                  beta=np.load(D / f"s{s}_unique_nrn_feats_msk.npy", mmap_mode="r"),
                  ncsnr=np.load(ROOT / f"s{s}_voxel_meta.npz")["ncsnr"],
                  coords=torch.from_numpy(np.load(D / f"s{s}_coords.npy").astype(np.float32)).to(dev),
                  qbeta=torch.from_numpy(np.load(D / f"s{s}_common_nrn_feats_msk.npy").astype(np.float32).T).to(dev))
qimg = torch.from_numpy(unit_norm(np.load(D / "common_img_feats_normed.npy").astype(np.float32))).to(dev)
print(f"data loaded {time.time()-t0:.0f}s")

feat_cache = {}
@torch.no_grad()
def F_of(K, s):
    if (K, s) not in feat_cache:
        feat_cache[(K, s)] = get_basis(K)(raw[s]["img"])
    return feat_cache[(K, s)]

cfull_cache = {}
@torch.no_grad()
def C_full(K, s, lam=1.0):
    """full-data coefficients (K, V_s) in the per-subject standardised basis + that subject's (mean, std)."""
    key = (K, s, lam)
    if key not in cfull_cache:
        F = F_of(K, s); Fm, Fsd = F.mean(0, keepdim=True), F.std(0, keepdim=True).clamp_min(1e-6)
        Fn = (F - Fm) / Fsd
        B = torch.from_numpy(np.asarray(raw[s]["beta"], dtype=np.float32)).to(dev)
        Bc = B - B.mean(0, keepdim=True)
        C = torch.linalg.solve(Fn.T @ Fn + lam * torch.eye(K, device=dev), Fn.T @ Bc)
        R2 = 1 - (Bc - Fn @ C).var(0) / Bc.var(0)                                   # in-sample EV per voxel
        del B, Bc; torch.cuda.empty_cache()
        cfull_cache[key] = (C, Fm, Fsd, R2)
    return cfull_cache[key]


popsm_cache = {}
def population(K, subjects, snr=0.2, lam=1.0, smooth=0.0):
    """smooth (mm): each training subject's coefficient map smoothed over its own voxels before masking (E39)."""
    Cs, XYZ, R2s = [], [], []
    for s in subjects:
        C, _, _, R2 = C_full(K, s, lam); m = torch.from_numpy(raw[s]["ncsnr"] > snr).to(dev)
        if smooth > 0:
            if (K, s, lam, smooth) not in popsm_cache:
                popsm_cache[(K, s, lam, smooth)] = smooth_over_voxels(C, raw[s]["coords"], smooth)
            C = popsm_cache[(K, s, lam, smooth)]
        Cs.append(C[:, m]); XYZ.append(raw[s]["coords"][m]); R2s.append(R2[m])
    return torch.cat(Cs, 1), torch.cat(XYZ, 0), torch.cat(R2s, 0)


@torch.no_grad()
def anat_average(xyz_v, xyz_p, vals, sigma, chunk=2048):
    """Gaussian (MNI distance, sigma mm) weighted mean of vals (J,) over the population for each new voxel."""
    out = torch.empty(xyz_v.shape[0], device=dev); pn = (xyz_p ** 2).sum(1)
    for a in range(0, xyz_v.shape[0], chunk):
        xv = xyz_v[a:a + chunk]; d2 = (xv ** 2).sum(1, keepdim=True) - 2 * xv @ xyz_p.T + pn[None, :]
        out[a:a + chunk] = torch.softmax(-0.5 * d2 / sigma ** 2, 1) @ vals
    return out


@torch.no_grad()
def smooth_over_voxels(C, xyz, sigma, chunk=1024):
    """Gaussian (sigma mm) weighted average of C (K, V) over the voxels of the same subject (self included)."""
    out = torch.empty_like(C); pn = (xyz ** 2).sum(1)
    for a in range(0, xyz.shape[0], chunk):
        xv = xyz[a:a + chunk]; d2 = (xv ** 2).sum(1, keepdim=True) - 2 * xv @ xyz.T + pn[None, :]
        out[:, a:a + chunk] = (torch.softmax(-0.5 * d2 / sigma ** 2, 1) @ C.T).T
    return out


# ------------------------------------------------------------------ parcel response fingerprint (E38)
cube_cache = {}
def cube_ids(cube):
    """MNI grid parcels (cube mm side) shared by all subjects: {s: (V_s,) parcel id}, number of parcels."""
    if cube not in cube_cache:
        cells = {s: torch.floor(raw[s]["coords"] / cube).long() for s in ALL}
        inv = torch.unique(torch.cat([cells[s] for s in ALL], 0), dim=0, return_inverse=True)[1]
        sizes = [raw[s]["coords"].shape[0] for s in ALL]
        cube_cache[cube] = (dict(zip(ALL, torch.split(inv, sizes))), int(inv.max()) + 1)
    return cube_cache[cube]


@torch.no_grad()
def fingerprint(Bc, ids, P):
    """Bc (n, V) centred responses -> (V, P) Pearson r between each voxel and the mean z-scored response of
    each parcel of the same subject (over the n images), plus the parcel voxel counts (P,)."""
    Z = Bc / Bc.std(0, keepdim=True).clamp_min(1e-6)
    cnt = torch.bincount(ids, minlength=P)
    Pm = torch.zeros(Z.shape[0], P, device=dev).index_add_(1, ids, Z) / cnt.clamp_min(1)[None, :]
    Pz = (Pm - Pm.mean(0, keepdim=True)) / Pm.std(0, keepdim=True).clamp_min(1e-6)
    return Z.T @ Pz / (Z.shape[0] - 1), cnt


fc_cache = {}
@torch.no_grad()
def fc_full(s, cube, K=None):
    """full-data fingerprint of subject s; K: on the residual of the full-data fit in basis K."""
    if (s, cube, K) not in fc_cache:
        ids, P = cube_ids(cube)
        B = torch.from_numpy(np.asarray(raw[s]["beta"], dtype=np.float32)).to(dev); B = B - B.mean(0, keepdim=True)
        if K is not None:
            C, Fm, Fsd, _ = C_full(K, s); B = B - ((F_of(K, s) - Fm) / Fsd) @ C
        fc_cache[(s, cube, K)] = fingerprint(B, ids[s], P)
        del B; torch.cuda.empty_cache()
    return fc_cache[(s, cube, K)]


# ------------------------------------------------------------------ generalised PopEB
@torch.no_grad()
def popeb_g(Fn, Bc, Cpop, tau2, lam0=1.0, kappa=1.0, kdiag=None, iters=1, spatial=None, chunk=512,
            temp=1.0, sig2_mix=None, local_cov=0.0, prior_only=False, anat_only=False, chat_match=None, match_s2_scale=1.0,
            extra=None):
    """spatial = (xyz_v (V,3), xyz_pop (J,3), sigma_mm): weights multiplied by exp(-d^2/2 sigma^2).
    temp: multiplies the population logits (softmax temperature 1/temp), shrinkage unchanged.
    sig2_mix = (w, s2_alt (V,)): sig2 <- (1-w) sig2_loo + w s2_alt (alternative noise estimate).
    local_cov = alpha: prior covariance  alpha Cov_w(C_j) + tau2 I  (weighted local population covariance)
    instead of tau2 I; weights unchanged.  prior_only: return Cbar (no likelihood term in the posterior).
    anat_only: population weights from the spatial term only (Cbar = anatomical prior mean).
    chat_match (K, V): coefficients used only inside the population weights (oracle-matching bound).
    extra = (D_v (V,P), D_pop (J,P), sigma): weights multiplied by exp(-|D_v - D_j|^2 / 2 sigma^2) (fingerprint)."""
    n, K = Fn.shape
    G = Fn.T @ Fn; e, Q = torch.linalg.eigh(G); FQ = Fn @ Q
    inv = 1.0 / (e + lam0); h = ((FQ ** 2) * inv).sum(1)
    Chat = Q @ (inv[:, None] * (FQ.T @ Bc))
    sig2 = kappa * (((Bc - Fn @ Chat) / (1 - h)[:, None]) ** 2).mean(0)
    if sig2_mix is not None:
        w_alt, s2_alt = sig2_mix; sig2 = (1 - w_alt) * sig2 + w_alt * s2_alt.clamp_min(1e-4)
    I = torch.eye(K, device=dev)
    Tau = tau2 * (torch.eye(K, device=dev) if kdiag is None else torch.diag(kdiag))
    Tinv = torch.linalg.inv(Tau)
    CC = (Cpop[:, None, :] * Cpop[None, :, :]).reshape(K * K, -1)
    if spatial is not None:
        xyz_v, xyz_p, sg = spatial; pn = (xyz_p ** 2).sum(1)
    if extra is not None:
        D_v, D_p, sd = extra; dn = (D_p ** 2).sum(1)
    cur = Chat if chat_match is None else chat_match
    for it in range(iters):
        out = torch.empty_like(Chat)
        for a in range(0, Chat.shape[1], chunk):
            s2 = sig2[a:a + chunk]; c = cur[:, a:a + chunk].T
            M = torch.linalg.solve((match_s2_scale * s2)[:, None, None] * I[None] + (G @ Tau)[None], G[None].expand(len(s2), K, K))   # (sig2 G^-1 + Tau)^-1 without inverting G
            cM = torch.einsum("vi,vij->vj", c, M)
            logits = -0.5 * ((cM * c).sum(1, keepdim=True) - 2 * cM @ Cpop + M.reshape(len(s2), -1) @ CC)
            if anat_only:
                logits = torch.zeros_like(logits)
            logits = logits * temp
            if spatial is not None:
                xv = xyz_v[a:a + chunk]; d2 = (xv ** 2).sum(1, keepdim=True) - 2 * xv @ xyz_p.T + pn[None, :]
                logits = logits - 0.5 * d2 / sg ** 2
            if extra is not None:
                dv = D_v[a:a + chunk]; logits = logits - 0.5 * ((dv ** 2).sum(1, keepdim=True) - 2 * dv @ D_p.T + dn[None, :]) / sd ** 2
            W = torch.softmax(logits, 1)
            Cbar = W @ Cpop.T
            if prior_only:
                out[:, a:a + chunk] = Cbar.T; continue
            if local_cov > 0:
                S = torch.einsum("vj,ij,kj->vik", W, Cpop, Cpop) - Cbar[:, :, None] * Cbar[:, None, :]   # weighted covariance
                Sinv = torch.linalg.inv(local_cov * S + Tau[None])
                A = G[None] / s2[:, None, None] + Sinv
                rhs = (G @ Chat[:, a:a + chunk]).T / s2[:, None] + torch.einsum("vi,vij->vj", Cbar, Sinv)
            else:
                A = G[None] / s2[:, None, None] + Tinv[None]
                rhs = (G @ Chat[:, a:a + chunk]).T / s2[:, None] + Cbar @ Tinv
            out[:, a:a + chunk] = torch.linalg.solve(A, rhs[:, :, None])[:, :, 0].T
        cur = out
    return cur


# ------------------------------------------------------------------ PopEB in the eigenbasis of G (E39)
# G = Q diag(e) Q' is shared by all voxels, so the matching precision (sig2 G^-1 + tau2 I)^-1 = Q diag(e/(sig2 + tau2 e)) Q'
# and the posterior precision G/sig2 + I/tau2 are diagonal in Q: no per-voxel solves, any K (K > n included).
def _space(Fn, Bc, Cpop, lam0, ols=False):
    e, Q = torch.linalg.eigh(Fn.T @ Fn); e = e.clamp_min(0); FQ = Fn @ Q
    inv = 1.0 / (e + lam0); h = ((FQ ** 2) * inv).sum(1); FtB = FQ.T @ Bc
    Ct = inv[:, None] * FtB                                                     # ridge estimate, eigen-coordinates (K, V)
    on = e > 1e-6 * e.max()
    Cm = torch.where(on[:, None], FtB / e.clamp_min(1e-12)[:, None], torch.zeros_like(FtB)) if ols else Ct
    P = Q.T @ Cpop
    return dict(e=e, Q=Q, Ct=Ct, Cm=Cm, on=on, P=P, P2=P ** 2, FQ=FQ, h=h)


def _quad(sp, c, s2, t2, w=1.0):
    """c (v, K) eigen-coordinates, s2 (v,), t2 (v,) -> Mahalanobis (v, J) to every population voxel, 0.5 log det (v,)."""
    d = sp["e"][None, :] / (s2[:, None] + t2[:, None] * sp["e"][None, :])
    dc = d * c
    q = (dc * c).sum(1, keepdim=True) - 2 * w * (dc @ sp["P"]) + w * w * (d @ sp["P2"])
    return q, 0.5 * torch.log(d[:, sp["on"]].clamp_min(1e-30)).sum(1)


@torch.no_grad()
def popeb_e(Fn, Bc, Cpop, tau2, lam0, spatial=None, smooth=None, alt=None, npmle=0, student=None,
            tau_grid=None, tau_mode="global", chunk=512, scales=None, proj=None):
    """smooth = (xyz, sigma): matching coefficients smoothed over the new subject's voxels (as match_smooth).
    alt = dict(Fn, Cpop, tau2, mix): second matching space (exact likelihood, OLS sufficient statistic); logits
    (1-mix) main + mix alt.  npmle = EM iterations on the mixing weights pi_j of the population atoms, estimated on the
    new subject's voxels (Kiefer-Wolfowitz NPMLE with fixed atoms).  student = nu: multivariate-t kernel in the weights.
    scales = [w_l]: atoms w_l C_j for every l (mash-style scale grid; matching invariant to the voxel gain on that grid).
    proj = U (K, d): matching on the statistic U'C_hat only (top-d principal directions of the population), marginal
    covariance sig2 U'(G + lam0 I)^-1 U + tau2 I; replaces the main-space logits.
    tau_grid: multipliers of tau2, selected by the marginal likelihood of the new subject ('global') or per voxel ('voxel')."""
    V = Bc.shape[1]; J = Cpop.shape[1]
    sp = _space(Fn, Bc, Cpop, lam0)
    sig2 = (((Bc - sp["FQ"] @ sp["Ct"]) / (1 - sp["h"])[:, None]) ** 2).mean(0)
    cm = sp["Cm"] if smooth is None else smooth_over_voxels(sp["Cm"], smooth[0], smooth[1])
    if alt is not None:
        spa = _space(alt["Fn"], Bc, alt["Cpop"], lam0, ols=True)
        cma = spa["Cm"] if smooth is None else smooth_over_voxels(spa["Cm"], smooth[0], smooth[1])
    if proj is not None:
        a_, R_ = torch.linalg.eigh(proj.T @ (sp["Q"] * (1.0 / (sp["e"] + lam0))[None, :]) @ sp["Q"].T @ proj)
        T_ = R_.T @ proj.T; Pp = T_ @ Cpop
        spp = dict(e=1.0 / a_.clamp_min(1e-12), on=torch.ones_like(a_, dtype=torch.bool), P=Pp, P2=Pp ** 2)
        cmp_ = T_ @ (sp["Q"] @ cm)
    if spatial is not None:
        xyz_v, xyz_p, sg = spatial; pn = (xyz_p ** 2).sum(1)
    Kn = int(sp["on"].sum()); sc = [1.0] if scales is None else scales; scw = torch.tensor(sc, device=dev)
    kern = (lambda q: -0.5 * q) if student is None else (lambda q: -0.5 * (student + Kn) * torch.log1p(q / student))

    def logits(a, b, t2, logpi):
        s2 = sig2[a:b]
        if proj is not None:
            q, ld = _quad(spp, cmp_[:, a:b].T, s2, t2[a:b]); lg = kern(q)
        elif scales is None:
            q, ld = _quad(sp, cm[:, a:b].T, s2, t2[a:b]); lg = kern(q)
        else:
            qs = [_quad(sp, cm[:, a:b].T, s2, t2[a:b], w) for w in sc]; ld = qs[0][1]; lg = torch.stack([kern(q) for q, _ in qs], 2)   # (v, J, L)
        if alt is not None:
            qa, _ = _quad(spa, cma[:, a:b].T, s2, torch.full_like(s2, alt["tau2"])); lg = (1 - alt["mix"]) * lg + alt["mix"] * kern(qa)
        if spatial is not None:
            sp_t = -0.5 * ((xyz_v[a:b] ** 2).sum(1, keepdim=True) - 2 * xyz_v[a:b] @ xyz_p.T + pn[None, :]) / sg ** 2
            lg = lg + (sp_t if lg.dim() == 2 else sp_t[:, :, None])
        return lg + (logpi[None, :] if lg.dim() == 2 else logpi[None, :, None]), ld

    t2 = torch.full((V,), tau2, device=dev); logpi = torch.zeros(J, device=dev)
    if tau_grid is not None:
        ev = torch.stack([torch.cat([(lambda lg, ld: torch.logsumexp(lg, 1) + ld)(*logits(a, min(a + chunk, V), t2 * g, logpi))
                                     for a in range(0, V, chunk)]) for g in tau_grid])                    # (grid, V)
        gsel = ev.argmax(0) if tau_mode == "voxel" else ev.sum(1).argmax().expand(V)
        t2 = t2 * torch.tensor(tau_grid, device=dev)[gsel]
    for _ in range(npmle):
        acc = torch.zeros(J, device=dev)
        for a in range(0, V, chunk):
            acc += torch.softmax(logits(a, min(a + chunk, V), t2, logpi)[0], 1).sum(0)
        logpi = torch.log(acc / V + 1e-12)
    out = torch.empty_like(sp["Ct"]); e = sp["e"]
    for a in range(0, V, chunk):
        b = min(a + chunk, V)
        lg = logits(a, b, t2, logpi)[0]
        W = torch.softmax(lg, 1) if lg.dim() == 2 else (torch.softmax(lg.reshape(lg.shape[0], -1), 1).reshape(lg.shape) * scw[None, None, :]).sum(2)
        Cbar = W @ sp["P"].T                                                                            # (v, K) eigen-coordinates
        s2 = sig2[a:b, None]; tv = t2[a:b, None]
        out[:, a:b] = ((e[None, :] * sp["Ct"][:, a:b].T / s2 + Cbar / tv) / (e[None, :] / s2 + 1.0 / tv)).T
    return sp["Q"] @ out


ALT_DIRS = {"clip": "clip_b16", "ml": "clip_pca512_clsmean_all"}
altx_cache = {}
def alt_X(space, s):
    """unit-norm image features of subject s's unique images (s = 'q': the common query images)."""
    if (space, s) not in altx_cache:
        f = ROOT / ALT_DIRS[space] / ("common_img_feats_normed.npy" if s == "q" else f"s{s}_unique_img_feats_normed.npy")
        altx_cache[(space, s)] = torch.from_numpy(unit_norm(np.load(f).astype(np.float32))).to(dev)
    return altx_cache[(space, s)]


altw_cache = {}
@torch.no_grad()
def alt_population(space, subs, snr):
    """full-data ridge weights (m, J) of the training voxels in a raw feature space (global LOO lambda per subject)
    and the map X -> design used for them (pooled mean, scale sqrt(m))."""
    key = (space, tuple(subs))
    if key not in altw_cache:
        mu = torch.cat([alt_X(space, s) for s in subs], 0).mean(0, keepdim=True); sc = math.sqrt(mu.shape[1])
        Ws = {}
        for s in subs:
            B = torch.from_numpy(np.asarray(raw[s]["beta"], dtype=np.float32)).to(dev); B = B - B.mean(0, keepdim=True)
            Ws[s], lam = ridge_loo((alt_X(space, s) - mu) * sc, B, LAMS); print(f"  alt {space} S{s}: lambda {lam}")
            del B; torch.cuda.empty_cache()
        altw_cache[key] = (Ws, mu, sc)
    Ws, mu, sc = altw_cache[key]
    return torch.cat([Ws[s][:, torch.from_numpy(raw[s]["ncsnr"] > snr).to(dev)] for s in subs], 1), (lambda X: (X - mu) * sc)


# ------------------------------------------------------------------ variants (one factor each)
BASE = dict(K=16, snr=0.2, subjects=None, cfull_lam=1.0, tau2=1e-3, lam0=1.0, kappa=1.0, kdiag=None, iters=1,
            spatial=None, std="support", temp=1.0, sig2=None, sig2_w=0.5, local_cov=0.0, prior_only=False, anat_only=False)
VARIANTS = {
    "popeb":        {},
    "A":            {"A": True},
    "pop_all":      dict(snr=-1),
    "pop_snr0.3":   dict(snr=0.3),
    "pop_snr0.4":   dict(snr=0.4),
    "pop_1subj":    dict(subjects=1),        # first available training subject only
    "pop_2subj":    dict(subjects=2),
    "cfull_lam10":  dict(cfull_lam=10.0),
    "cfull_lam100": dict(cfull_lam=100.0),
    "lam0_0.1":     dict(lam0=0.1),
    "lam0_10":      dict(lam0=10.0),
    "kappa0.5":     dict(kappa=0.5),
    "kappa2":       dict(kappa=2.0),
    "tau2_0.0005":  dict(tau2=5e-4),
    "tau2_0.002":   dict(tau2=2e-3),
    "aniso":        dict(kdiag="popvar"),
    "iter2":        dict(iters=2),
    "std_global":   dict(std="global"),
    "spatial5":     dict(spatial=5.0),
    "spatial10":    dict(spatial=10.0),
    "spatial20":    dict(spatial=20.0),
    "K32":          dict(K=32, tau2="ratio"),
    "K64":          dict(K=64, tau2="ratio"),
    "lam0_100":     dict(lam0=100.0),
    "K32_l10":      dict(K=32, tau2="ratio", lam0=10.0),
    "K32_l100":     dict(K=32, tau2="ratio", lam0=100.0),
    "K64_l10":      dict(K=64, tau2="ratio", lam0=10.0),
    "K64_l100":     dict(K=64, tau2="ratio", lam0=100.0),
    "sp10_std":     dict(spatial=10.0, std="global"),
    "sp10_K32_l100":     dict(spatial=10.0, K=32, tau2="ratio", lam0=100.0),
    "sp10_std_K32_l100": dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=100.0),
    "sp10_std_K64_l100": dict(spatial=10.0, std="global", K=64, tau2="ratio", lam0=100.0),
    "sp10_std_l10":      dict(spatial=10.0, std="global", lam0=10.0),
    "sp10_std_K32":      dict(spatial=10.0, std="global", K=32, tau2="ratio"),
    "sp10_std_K64":      dict(spatial=10.0, std="global", K=64, tau2="ratio"),
    "sp10_std_K32_l10":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0),
    "sp10_std_K64_l10":  dict(spatial=10.0, std="global", K=64, tau2="ratio", lam0=10.0),
    "v2_shuffle_xyz":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, shuffle_xyz=True),   # E16 control
    "v2_nospatial":      dict(std="global", K=32, tau2="ratio", lam0=10.0),
    # --- round 2 (2026-09-12): one factor each on top of PopEB-v2 ---
    "v2":                dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0),
    "v2_temp0.5":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, temp=0.5),
    "v2_temp2":          dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, temp=2.0),
    "v2_sig2pool":       dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, sig2="pool"),      # 0.5 LOO + 0.5 median
    "v2_sig2anat":       dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, sig2="anat"),      # 0.5 LOO + 0.5 var(B)(1-R2_anat)
    "v2_sig2anat1":      dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, sig2="anat", sig2_w=1.0),
    "v2_sig2oracle":     dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, sig2="oracle", sig2_w=1.0),   # bound, S1 only
    "v2_localcov":       dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, local_cov=1.0),
    "v2_localcov0.5":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, local_cov=0.5),
    "v2_localcov0.25":   dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, local_cov=0.25),
    "v2_prioronly":      dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, prior_only=True),
    "v2_anatonly":       dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, anat_only=True),
    "v2_anatprioronly":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, anat_only=True, prior_only=True),
    "v2_oraclematch":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, oracle_match=True),   # bound, S1 only
    "v2_ensK":           dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, ens=[16, 32, 64]),
    "v2_ensK2":          dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, ens=[32, 64]),
    # matching with the support estimate smoothed over the new subject's neighbouring voxels (sigma mm); posterior unchanged
    "v2_msm1.5":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.5),
    "v2_msm2":           dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=2.0),
    "v2_msm3":           dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=3.0),
    "v2_msm2_s0.5":      dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=2.0, match_s2=0.5),
    "v2_msm2_s0.25":     dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=2.0, match_s2=0.25),
    "v2_msm3_s0.5":      dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=3.0, match_s2=0.5),
    "v2_msm3_s0.25":     dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=3.0, match_s2=0.25),
    "v2_msm5_s0.25":     dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=5.0, match_s2=0.25),
    "v2_msm1":           dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.0),
    "v2_msm1.2":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2),
    # adaptive mix: C_match = (1-a) C_hat + a smooth(C_hat), a_v = sig2_v / (sig2_v + beta n)  (noise ~ sig2/n vs fixed neighbour bias)
    "v2_msm1.5_b0.001":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.5, match_beta=0.001),
    "v2_msm1.5_b0.003":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.5, match_beta=0.003),
    "v2_msm1.5_b0.01":   dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.5, match_beta=0.01),
    "v2_msm2_b0.003":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=2.0, match_beta=0.003),
    "v2_msm2_b0.01":     dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=2.0, match_beta=0.01),
    "v2_ensK_msm1.5":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.5, ens=[16, 32, 64]),
    "v2_ensK_msm1.2":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, ens=[16, 32, 64]),
    # --- E38 (2026-09-18): matching descriptors from the new subject's own response covariance, on top of v2_msm1.2 ---
    # match_pca = d: C_hat for the weights from the support responses projected on their top-d trial-space PCs
    "v2m_pca5":          dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, match_pca=5),
    "v2m_pca10":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, match_pca=10),
    "v2m_pca20":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, match_pca=20),
    "v2m_pca50":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, match_pca=50),
    # fc = sigma: extra weight term on the parcel fingerprint distance (fc_cube mm grid; fc_resid: residual of the basis fit)
    "v2m_fc0.25":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.25),
    "v2m_fc0.5":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.5),
    "v2m_fc1.0":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=1.0),
    "v2m_fc2.0":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=2.0),
    "v2m_fc0.5_c20":   dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.5, fc_cube=20.0),
    "v2m_fcres0.5":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.5, fc_resid=True),
    "v2m_fcres1.0":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=1.0, fc_resid=True),
    "v2m_fc0.5_nosp":  dict(std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.5),      # fingerprint instead of MNI distance
    "v2m_fcres0.1_q10n":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.1, fc_resid=True, fc_q=10, fc_noise=True),
    "v2m_fcres0.2_q10n":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.2, fc_resid=True, fc_q=10, fc_noise=True),
    "v2m_fcres0.4_q10n":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.4, fc_resid=True, fc_q=10, fc_noise=True),
    "v2m_fcres0.1_q30n":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.1, fc_resid=True, fc_q=30, fc_noise=True),
    "v2m_fcres0.2_q30n":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.2, fc_resid=True, fc_q=30, fc_noise=True),
    "v2m_fcres0.4_q30n":  dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.4, fc_resid=True, fc_q=30, fc_noise=True),
    "v2m_fcres0.5_n":   dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.5, fc_resid=True, fc_noise=True),
    "v2m_fcres1.0_n":   dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=1.0, fc_resid=True, fc_noise=True),
    "v2m_fc0.2_q10n":   dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.2, fc_q=10, fc_noise=True),
    "v2m_fc0.4_q10n":   dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.4, fc_q=10, fc_noise=True),
    "v2m_fcres0.2_orc": dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.2, fc_resid=True, fc_oracle=True),      # bound: held-out full-data fingerprint
    "v2m_fcres0.5_orc": dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=0.5, fc_resid=True, fc_oracle=True),      # bound: held-out full-data fingerprint
    "v2m_fcres1.0_orc": dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=1.0, fc_resid=True, fc_oracle=True),      # bound: held-out full-data fingerprint
    "v2m_fcres2.0":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, fc=2.0, fc_resid=True),
    # --- E39 (2026-09-18): component replacements on top of v2_msm1.2, eigenbasis implementation (popeb_e) ---
    "e_v2":            dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True),                                   # must equal v2_msm1.2
    "e_npmle1":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, npmle=1),
    "e_npmle3":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, npmle=3),
    "e_npmle10":       dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, npmle=10),
    "e_student4":      dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, student=4.0),
    "e_student1":      dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, student=1.0),
    "e_tauG":          dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, tau_grid=[0.25, 0.5, 1.0, 2.0, 4.0]),
    "e_tauV":          dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, tau_grid=[0.25, 0.5, 1.0, 2.0, 4.0], tau_mode="voxel"),
    "e_popsm1.2":      dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, pop_smooth=1.2),
    "e_popsm2":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, pop_smooth=2.0),
    "e_xK64":          dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, alt="K64", alt_mix=1.0),
    "e_xK64_h":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, alt="K64", alt_mix=0.5),
    "e_xclip":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, alt="clip", alt_mix=1.0),
    "e_xclip_h":       dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, alt="clip", alt_mix=0.5),
    "e_xclip_r0.2":    dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, alt="clip", alt_mix=1.0, alt_ratio=0.2),
    "e_xml":           dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, alt="ml", alt_mix=1.0),
    "e_xml_h":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, alt="ml", alt_mix=0.5),
    "e_scale3":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, scales=[0.7, 1.0, 1.4]),
    "e_scale5":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, scales=[0.5, 0.7, 1.0, 1.4, 2.0]),
    "e_scale7":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, scales=[0.25, 0.5, 0.7, 1.0, 1.4, 2.0, 4.0]),
    "e_proj8":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, proj=8),
    "e_proj12":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, proj=12),
    "e_proj16":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, proj=16),
    "e_proj24":        dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, proj=24),
    "e_rr":            dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, resid_ridge="clip"),
    "e_rr_ml":         dict(spatial=10.0, std="global", K=32, tau2="ratio", lam0=10.0, match_smooth=1.2, eig=True, resid_ridge="ml"),                 # adds multi-layer features: not the same feature budget
}
names = REQ or list(VARIANTS)


@torch.no_grad()
def run_variant(name, held, train_subjects, idx, qF_getter):
    """returns per-voxel EV (V,) for held-out subject `held` with support indices idx."""
    v = dict(BASE, **VARIANTS[name])
    if v.get("ens"):     # average the predictions of the same variant over several bases
        preds = [predict_variant(dict(v, K=K, ens=None), held, train_subjects, idx, qF_getter) for K in v["ens"]]
        return explained_variance(sum(preds) / len(preds), raw[held]["qbeta"]).cpu().numpy()
    return explained_variance(predict_variant(v, held, train_subjects, idx, qF_getter), raw[held]["qbeta"]).cpu().numpy()


@torch.no_grad()
def predict_variant(v, held, train_subjects, idx, qF_getter):
    K = v["K"]
    subs = train_subjects if v["subjects"] is None else train_subjects[:v["subjects"]]
    basis = get_basis(K)
    B = torch.from_numpy(np.asarray(raw[held]["beta"][idx], dtype=np.float32)).to(dev)
    Fs = F_of(K, held)[idx]; Fq = qF_getter(K)
    Bm = B.mean(0, keepdim=True); Bc = B - Bm
    if v["std"] == "support":
        Fm, Fsd = Fs.mean(0, keepdim=True), Fs.std(0, keepdim=True).clamp_min(1e-6)
    else:   # global: pooled statistics of the training subjects' full feature sets
        Fall = torch.cat([F_of(K, s) for s in subs], 0); Fm, Fsd = Fall.mean(0, keepdim=True), Fall.std(0, keepdim=True)
    Fn, Fqn = (Fs - Fm) / Fsd, (Fq - Fm) / Fsd
    if v.get("A"):
        C = ridge_loo(Fn, Bc, LAMS)[0]
    else:
        Cpop, xyz_p, R2pop = population(K, subs, v["snr"], v["cfull_lam"], v.get("pop_smooth", 0.0))
        tau2 = v["tau2"]
        if tau2 == "ratio":
            tau2 = 0.0227 * Cpop.var(1).mean().item()      # same tau2 / population-variance ratio as K=16
        kdiag = None
        if v["kdiag"] == "popvar":
            pv = Cpop.var(1); kdiag = pv / pv.mean()
        spatial = None if v["spatial"] is None else (raw[held]["coords"], xyz_p, v["spatial"])
        if v.get("shuffle_xyz") and spatial is not None:      # E16 control: permute population coordinates (seeded)
            g = torch.Generator(device="cpu").manual_seed(0)
            spatial = (spatial[0], xyz_p[torch.randperm(xyz_p.shape[0], generator=g).to(dev)], spatial[2])
        sig2_mix = None
        if v["sig2"] == "pool":
            e_, Q_ = torch.linalg.eigh(Fn.T @ Fn); FQ_ = Fn @ Q_; inv_ = 1 / (e_ + v["lam0"]); h_ = ((FQ_ ** 2) * inv_).sum(1)
            s2 = (((Bc - Fn @ (Q_ @ (inv_[:, None] * (FQ_.T @ Bc)))) / (1 - h_)[:, None]) ** 2).mean(0)
            sig2_mix = (v["sig2_w"], torch.full_like(s2, s2.median().item()))
        elif v["sig2"] == "anat":
            r2 = anat_average(raw[held]["coords"], xyz_p, R2pop, 10.0)
            sig2_mix = (v["sig2_w"], Bc.var(0) * (1 - r2).clamp_min(0.02))
        elif v["sig2"] == "oracle":
            Ch, _, _, _ = C_full(K, held); Fh = (F_of(K, held) - Fm) / Fsd
            Bh = torch.from_numpy(np.asarray(raw[held]["beta"], dtype=np.float32)).to(dev); Bh = Bh - Bh.mean(0, keepdim=True)
            Ch = torch.linalg.solve(Fh.T @ Fh + torch.eye(K, device=dev), Fh.T @ Bh)
            sig2_mix = (v["sig2_w"], (Bh - Fh @ Ch).var(0)); del Bh
        chat_match = None; Bmatch = Bc
        if v.get("match_pca"):
            Zs = Bc / Bc.std(0, keepdim=True).clamp_min(1e-6)
            U = torch.linalg.eigh(Zs @ Zs.T)[1][:, -min(v["match_pca"], Bc.shape[0] - 1):]
            Bmatch = U @ (U.T @ Bc)
        if (v.get("match_smooth") or v.get("match_pca")) and not v.get("eig"):
            e_, Q_ = torch.linalg.eigh(Fn.T @ Fn); FQ_ = Fn @ Q_; inv_ = 1 / (e_ + v["lam0"]); h_ = ((FQ_ ** 2) * inv_).sum(1)
            Ch_ = Q_ @ (inv_[:, None] * (FQ_.T @ Bmatch))
            chat_match = smooth_over_voxels(Ch_, raw[held]["coords"], v["match_smooth"]) if v.get("match_smooth") else Ch_
            if v.get("match_beta"):
                s2_ = (((Bc - Fn @ Ch_) / (1 - h_)[:, None]) ** 2).mean(0)
                a_ = s2_ / (s2_ + v["match_beta"] * Fn.shape[0])
                chat_match = (1 - a_)[None, :] * Ch_ + a_[None, :] * chat_match
        extra = None
        if v.get("fc"):
            cube = v.get("fc_cube", 10.0); ids, P = cube_ids(cube); Kr = K if v.get("fc_resid") else None
            Bfc = Bc
            if Kr is not None:      # leave-one-out residual of the support ridge
                e_, Q_ = torch.linalg.eigh(Fn.T @ Fn); FQ_ = Fn @ Q_; inv_ = 1 / (e_ + v["lam0"]); h_ = ((FQ_ ** 2) * inv_).sum(1)
                Bfc = (Bc - Fn @ (Q_ @ (inv_[:, None] * (FQ_.T @ Bc)))) / (1 - h_)[:, None]; Bfc = Bfc - Bfc.mean(0, keepdim=True)
            D_v, cnt = fc_full(held, cube, Kr) if v.get("fc_oracle") else fingerprint(Bfc, ids[held], P); ok = cnt >= 20
            for s in subs:
                ok &= fc_full(s, cube, Kr)[1] >= 20
            D_p = torch.cat([fc_full(s, cube, Kr)[0][torch.from_numpy(raw[s]["ncsnr"] > v["snr"]).to(dev)] for s in subs], 0)
            D_v, D_p = D_v[:, ok], D_p[:, ok]
            if v.get("fc_q"):       # top-q principal axes of the population fingerprints
                mu = D_p.mean(0, keepdim=True); Vq = torch.linalg.eigh((D_p - mu).T @ (D_p - mu))[1][:, -v["fc_q"]:]
                D_v, D_p = (D_v - mu) @ Vq, (D_p - mu) @ Vq
            # fc_noise: sampling variance 1/n of a correlation estimated on n images added to sigma^2
            extra = (D_v, D_p, math.sqrt(v["fc"] ** 2 + 1.0 / Bc.shape[0]) if v.get("fc_noise") else v["fc"])
        if v.get("oracle_match"):      # bound: weights from the held-out subject's own full-data coefficients
            Fh = (F_of(K, held) - Fm) / Fsd
            Bh = torch.from_numpy(np.asarray(raw[held]["beta"], dtype=np.float32)).to(dev); Bh = Bh - Bh.mean(0, keepdim=True)
            chat_match = torch.linalg.solve(Fh.T @ Fh + torch.eye(K, device=dev), Fh.T @ Bh); del Bh
        if v.get("eig"):
            alt = None
            if v.get("alt") == "K64":
                Fa = torch.cat([F_of(64, s) for s in subs], 0); Ca = population(64, subs, v["snr"], v["cfull_lam"])[0]
                alt = dict(Fn=(F_of(64, held)[idx] - Fa.mean(0, keepdim=True)) / Fa.std(0, keepdim=True), Cpop=Ca)
            elif v.get("alt"):
                Ca, fmap = alt_population(v["alt"], subs, v["snr"]); alt = dict(Fn=fmap(alt_X(v["alt"], held)[idx]), Cpop=Ca)
            if alt is not None:
                alt.update(tau2=v.get("alt_ratio", 0.0227) * alt["Cpop"].var(1).mean().item(), mix=v["alt_mix"])
            C = popeb_e(Fn, Bc, Cpop, tau2, v["lam0"], spatial, (raw[held]["coords"], v["match_smooth"]), alt,
                        v.get("npmle", 0), v.get("student"), v.get("tau_grid"), v.get("tau_mode", "global"), scales=v.get("scales"),
                        proj=None if not v.get("proj") else torch.linalg.eigh(torch.cov(Cpop))[1][:, -v["proj"]:])
            pred = Bm + Fqn @ C
            if v.get("resid_ridge"):      # ridge (global LOO lambda) on raw features fitted to the support residual of PopEB
                Xs = alt_X(v["resid_ridge"], held)[idx]; xm = Xs.mean(0, keepdim=True)
                pred = pred + (alt_X(v["resid_ridge"], "q") - xm) @ ridge_loo(Xs - xm, Bc - Fn @ C, LAMS)[0]
            return pred.T
        C = popeb_g(Fn, Bc, Cpop, tau2, v["lam0"], v["kappa"], kdiag, v["iters"], spatial,
                    temp=v["temp"], sig2_mix=sig2_mix, local_cov=v["local_cov"], prior_only=v["prior_only"], anat_only=v["anat_only"],
                    chat_match=chat_match, match_s2_scale=v.get("match_s2", 1.0), extra=extra)
    return (Bm + Fqn @ C).T


rng = np.random.default_rng(0)
def boot(d, nb=1000):
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nb)]); return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


if MODE == "s1":
    bundle = SOURCES.build(dict(name="npy_multisubj", data_dir=str(D), train_subjects=TR, val_subjects=[2])).load_eval(1)
    rois = roi_masks(str(ROOT / "s1_voxel_meta.npz")); good = rois["ncsnr>0.2"]; ncsnr = raw[1]["ncsnr"]
    bins = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 10]; bm = [good & (ncsnr >= a) & (ncsnr < b) for a, b in zip(bins[:-1], bins[1:])]
    draws = []
    start = 0
    for rep in range(5):
        draws.append((100, rep, "braincorl_s1_ic100", bundle.icl_order[start:start + 100])); start += 100
    start = 0
    for n in [30, 300, 1000]:
        for rep in range(3):
            draws.append((n, rep, "braincorl_s1_scaling", bundle.icl_order[start:start + n])); start += n
    qF = lambda K: get_basis(K)(qimg)
    res = {}
    for name in names:
        for n, rep, bdir, idx in draws:
            res.setdefault((name, n), []).append(run_variant(name, 1, TR, idx, qF))
        print(f"{name:13s} " + " ".join(f"n={n}:{np.mean([e[good].mean() for e in res[(name, n)]]):.4f}" for n in [30, 100, 300, 1000]) + f"  ({time.time()-t0:.0f}s)")
    bc = {n: np.mean([np.load(OUT / bdir / f"icl_num_ic={n}_rep={rep}_expl_var.npy") for nn, rep, bdir, _ in draws if nn == n], 0) for n in [30, 100, 300, 1000]}
    print("\n## S1, EV ncsnr>0.2: variant | per n: EV (diff vs popeb [CI95])")
    print(f"{'BraInCoRL':13s} " + " ".join(f"n={n}: {bc[n][good].mean():.4f}" for n in [30, 100, 300, 1000]))
    for name in names:
        line = f"{name:13s} "
        for n in [30, 100, 300, 1000]:
            M = np.mean(res[(name, n)], 0); P = np.mean(res[(names[0], n)], 0)
            if name == names[0]:
                line += f"n={n}: {M[good].mean():.4f}             "
            else:
                d, lo, hi = boot((M - P)[good]); line += f"n={n}: {M[good].mean():.4f} ({d:+.4f} [{lo:+.4f},{hi:+.4f}]) "
        print(line)
    print("\n## S1 n=100 by ncsnr bin [0.2,.3) [.3,.4) [.4,.5) [.5,.7) [.7,1) >=1")
    for name in names:
        M = np.mean(res[(name, 100)], 0); print(f"{name:13s} " + " ".join(f"{M[m].mean():.3f}" for m in bm))
    print(f"{'BraInCoRL':13s} " + " ".join(f"{bc[100][m].mean():.3f}" for m in bm))

else:   # nested
    NS = [30, 100, 300, 1000]; REPS = 2
    res = {}
    for h in TR:
        others = [s for s in TR if s != h]; g = torch.from_numpy(raw[h]["ncsnr"] > 0.2).to(dev).cpu().numpy()
        r = np.random.default_rng(11 + h); qF = lambda K: get_basis(K)(qimg)
        for n in NS:
            for _ in range(REPS):
                idx = np.sort(r.permutation(len(raw[h]["beta"]))[:n])
                for name in names:
                    res.setdefault((name, n), []).append(run_variant(name, h, others, idx, qF)[g].mean())
        print(f"held-out S{h} done ({time.time()-t0:.0f}s)")
    print("\n## nested (mean EV on held-out training subject, ncsnr>0.2, 3 subjects x 2 draws); diff vs popeb")
    for name in names:
        print(f"{name:13s} " + " ".join(f"n={n}: {np.mean(res[(name, n)]):.4f} ({np.mean(res[(name, n)]) - np.mean(res[(names[0], n)]):+.4f})" for n in NS))
print(f"done {time.time()-t0:.0f}s")
