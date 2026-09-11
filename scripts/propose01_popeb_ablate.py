"""One-factor ablations of PopEB (docs/propose/propose01.md 9.6): on S1 with the same support draws as
propose01 (paired with the BraInCoRL reruns), or nested on S2/S5/S7 (leave-one-training-subject-out,
population from the other two). Self-contained (re-implements PopEB with the extra knobs).
    python scripts/propose01_popeb_ablate.py s1 [variant ...]          # default: all variants
    python scripts/propose01_popeb_ablate.py nested variant ...
"""
import sys, time, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from src.model.lowrank_basis import FunctionalBasis, ridge_loo, LowRankConfig
from src.dataset import SOURCES
from src.eval import roi_masks
from src.utils import explained_variance, unit_norm

dev = torch.device("cuda"); D = Path("data/braincorl/clip_b16"); OUT = Path("outputs/propose01")
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
                  ncsnr=np.load(f"data/braincorl/s{s}_voxel_meta.npz")["ncsnr"],
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
        C = torch.linalg.solve(Fn.T @ Fn + lam * torch.eye(K, device=dev), Fn.T @ (B - B.mean(0, keepdim=True)))
        del B; torch.cuda.empty_cache()
        cfull_cache[key] = (C, Fm, Fsd)
    return cfull_cache[key]


def population(K, subjects, snr=0.2, lam=1.0):
    Cs, XYZ = [], []
    for s in subjects:
        C, _, _ = C_full(K, s, lam); m = torch.from_numpy(raw[s]["ncsnr"] > snr).to(dev)
        Cs.append(C[:, m]); XYZ.append(raw[s]["coords"][m])
    return torch.cat(Cs, 1), torch.cat(XYZ, 0)


# ------------------------------------------------------------------ generalised PopEB
@torch.no_grad()
def popeb_g(Fn, Bc, Cpop, tau2, lam0=1.0, kappa=1.0, kdiag=None, iters=1, spatial=None, chunk=512):
    """spatial = (xyz_v (V,3), xyz_pop (J,3), sigma_mm): weights multiplied by exp(-d^2/2 sigma^2)."""
    n, K = Fn.shape
    G = Fn.T @ Fn; e, Q = torch.linalg.eigh(G); FQ = Fn @ Q
    inv = 1.0 / (e + lam0); h = ((FQ ** 2) * inv).sum(1)
    Chat = Q @ (inv[:, None] * (FQ.T @ Bc))
    sig2 = kappa * (((Bc - Fn @ Chat) / (1 - h)[:, None]) ** 2).mean(0)
    I = torch.eye(K, device=dev)
    Tau = tau2 * (torch.eye(K, device=dev) if kdiag is None else torch.diag(kdiag))
    Tinv = torch.linalg.inv(Tau)
    CC = (Cpop[:, None, :] * Cpop[None, :, :]).reshape(K * K, -1)
    if spatial is not None:
        xyz_v, xyz_p, sg = spatial; pn = (xyz_p ** 2).sum(1)
    cur = Chat
    for it in range(iters):
        out = torch.empty_like(Chat)
        for a in range(0, Chat.shape[1], chunk):
            s2 = sig2[a:a + chunk]; c = cur[:, a:a + chunk].T
            M = torch.linalg.solve(s2[:, None, None] * I[None] + (G @ Tau)[None], G[None].expand(len(s2), K, K))   # (sig2 G^-1 + Tau)^-1 without inverting G
            cM = torch.einsum("vi,vij->vj", c, M)
            logits = -0.5 * ((cM * c).sum(1, keepdim=True) - 2 * cM @ Cpop + M.reshape(len(s2), -1) @ CC)
            if spatial is not None:
                xv = xyz_v[a:a + chunk]; d2 = (xv ** 2).sum(1, keepdim=True) - 2 * xv @ xyz_p.T + pn[None, :]
                logits = logits - 0.5 * d2 / sg ** 2
            Cbar = torch.softmax(logits, 1) @ Cpop.T
            A = G[None] / s2[:, None, None] + Tinv[None]
            rhs = (G @ Chat[:, a:a + chunk]).T / s2[:, None] + Cbar @ Tinv
            out[:, a:a + chunk] = torch.linalg.solve(A, rhs[:, :, None])[:, :, 0].T
        cur = out
    return cur


# ------------------------------------------------------------------ variants (one factor each)
BASE = dict(K=16, snr=0.2, subjects=None, cfull_lam=1.0, tau2=1e-3, lam0=1.0, kappa=1.0, kdiag=None, iters=1,
            spatial=None, std="support")
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
}
names = REQ or list(VARIANTS)


@torch.no_grad()
def run_variant(name, held, train_subjects, idx, qF_getter):
    """returns per-voxel EV (V,) for held-out subject `held` with support indices idx."""
    v = dict(BASE, **VARIANTS[name]); K = v["K"]
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
        Cpop, xyz_p = population(K, subs, v["snr"], v["cfull_lam"])
        tau2 = v["tau2"]
        if tau2 == "ratio":
            tau2 = 0.0227 * Cpop.var(1).mean().item()      # same tau2 / population-variance ratio as K=16
        kdiag = None
        if v["kdiag"] == "popvar":
            pv = Cpop.var(1); kdiag = pv / pv.mean()
        spatial = None if v["spatial"] is None else (raw[held]["coords"], xyz_p, v["spatial"])
        C = popeb_g(Fn, Bc, Cpop, tau2, v["lam0"], v["kappa"], kdiag, v["iters"], spatial)
    return explained_variance((Bm + Fqn @ C).T, raw[held]["qbeta"]).cpu().numpy()


rng = np.random.default_rng(0)
def boot(d, nb=1000):
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nb)]); return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


if MODE == "s1":
    bundle = SOURCES.build(dict(name="npy_multisubj", data_dir=str(D), train_subjects=TR, val_subjects=[2])).load_eval(1)
    rois = roi_masks("data/braincorl/s1_voxel_meta.npz"); good = rois["ncsnr>0.2"]; ncsnr = raw[1]["ncsnr"]
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
            M = np.mean(res[(name, n)], 0); P = np.mean(res[("popeb", n)], 0) if ("popeb", n) in res else None
            if P is None or name == "popeb":
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
        print(f"{name:13s} " + " ".join(f"n={n}: {np.mean(res[(name, n)]):.4f} ({np.mean(res[(name, n)]) - np.mean(res[('popeb', n)]):+.4f})" for n in NS))
print(f"done {time.time()-t0:.0f}s")
