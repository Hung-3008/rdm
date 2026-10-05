"""Shared functional latent map (SFLM) pilot, docs/proposal-shared-functional-latent-2026-09-20.md step 2.

Slow stage (training subjects, unique images only, 8,400 train / 600 val per subject, seed 0 as in
docs/_archive/popeb_v2/code/train_lowrank.py):
    latent map      z_p(x) = W2 SiLU(W1 phi_p(x) + b1) in R^K for each of the 49 cells of the 7x7 CLIP grid
    source readout  beta_v(x) = c_v' sum_m softmax(l_v)_m u_m(x) + b_v,   u_m(x) = sum_p A[m, p] z_p(x)
                    A = dictionary of M isotropic Gaussian masks over the grid (+ one uniform mask)
Fast stage (held-out subject, closed form, n support pairs + MNI coordinates only):
    for every mask m: PopEB (docs/_archive/popeb_v2/code/lowrank_basis.py, unchanged) in the K pooled channels u_m
    mask posterior  q_vm ∝ pi_vm * sig2_vm^(-n/2)   sig2 = leave-one-out MSE of the lam0 ridge,
                    pi = MNI-kernel average of the training voxels' full-data mask posteriors
    prediction      sum_m q_vm pred_vm
Baselines on the same support draws: ridge and PopEB-v2 on the PCA-512 multi-layer CLIP feature.
Model code lives in src/sflm (latent map, mask dictionary, readout, training loop); this script keeps the pilot protocol.

    python scripts/sflm_pilot.py --holdout 1 --out outputs/sflm/h1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_

sys.path.insert(0, ".")
sys.path.insert(0, "docs/_archive/popeb_v2/code")
from lowrank_basis import FunctionalBasis, SubjectHeads, feature_stats, full_coefficients, popeb  # noqa: E402
from src.sflm.fast import kernel_average, ridge_fixed  # noqa: E402
from src.sflm.latent import DATA, DEV, STREAMS, GridStore, mask_dictionary, split_8400_600, train_latent  # noqa: E402
from src.utils.metrics import explained_variance, pearson  # noqa: E402
from src.utils.predict import unit_norm  # noqa: E402
from src.utils.ridge import ridge_cv_brainwise  # noqa: E402

PCA = DATA / "clip_pca512_clsmean_all"


def train_pca_basis(train_subj, a, out: Path) -> FunctionalBasis:
    """PopEB-v2 slow stage on the PCA-512 feature (train_lowrank.py with the 8,400 / 600 split)."""
    imgs = [torch.from_numpy(unit_norm(np.load(PCA / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))).to(DEV) for s in train_subj]
    betas = [torch.from_numpy(np.load(DATA / f"s{s}_betas_unique.npy")).to(DEV) for s in train_subj]
    split = split_8400_600(len(train_subj))
    basis = FunctionalBasis(imgs[0].shape[1], 512, a.K).to(DEV)
    heads = SubjectHeads(a.K, [b.shape[1] for b in betas]).to(DEV)
    opt = torch.optim.AdamW([{"params": basis.parameters(), "weight_decay": 0.0},
                             {"params": heads.parameters(), "weight_decay": 1e-4}], lr=1e-3)
    best, best_ep = -1e9, -1
    for ep in range(200):
        perms = [torch.from_numpy(np.random.permutation(tr)).to(DEV) for tr, _ in split]
        for i in range(len(perms[0]) // 512):
            loss = sum(torch.mean((heads(basis(imgs[k][perms[k][i * 512:(i + 1) * 512]]), k) - betas[k][perms[k][i * 512:(i + 1) * 512]]) ** 2)
                       for k in range(len(train_subj)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            ev = float(np.mean([explained_variance(heads(basis(imgs[k][va]), k).T, betas[k][va].T).mean().item()
                                for k, (_, va) in enumerate(split)]))
        if ev > best:
            best, best_ep = ev, ep
            torch.save(dict(cfg=basis.cfg, state_dict=basis.state_dict()), out / "pca_basis.pt")
        elif ep - best_ep >= 10:
            break
    print(f"[pca basis] best val_ev {best:.4f} at epoch {best_ep}", flush=True)
    return FunctionalBasis.from_file(out / "pca_basis.pt", DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--out", default="outputs/sflm/h1")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_ic", type=int, nargs="+", default=[30, 100, 300, 1000])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--lam0", type=float, default=10.0)
    ap.add_argument("--tau2_ratio", type=float, default=0.0227)
    ap.add_argument("--sigma_mm", type=float, default=10.0)
    ap.add_argument("--prior_floor", type=float, default=0.1, help="weight of the uniform component in the mask prior")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    train_subj = [s for s in (1, 2, 5, 7) if s != a.holdout]
    A = mask_dictionary().to(DEV)
    M = A.shape[0]
    store = GridStore()

    # ---- slow stage -------------------------------------------------------------------------------------------
    g, U_tr, betas_tr, info = train_latent(train_subj, store, A, a, out)
    U_all = torch.cat(U_tr)
    u_mean, u_std = U_all.mean(0, keepdim=True), U_all.std(0, keepdim=True).clamp_min(1e-6)     # (1, M, K) pooled statistics
    del U_all

    # population: full-data coefficients and mask posterior of the training subjects' ncsnr > 0.2 voxels
    Cpop, qpop, xyz_pop = [], [], []
    for s, U, B in zip(train_subj, U_tr, betas_tr):
        good = torch.from_numpy(np.load(DATA / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2).to(DEV)
        Bc = (B - B.mean(0, keepdim=True))[:, good]
        Un = (U - u_mean) / u_std
        Cs, ll = [], []
        for m in range(M):
            C, loo = ridge_fixed(Un[:, m], Bc, 1.0)
            Cs.append(C)
            ll.append(-0.5 * Un.shape[0] * torch.log(loo))
        Cpop.append(torch.stack(Cs))                                                               # (M, K, J_s)
        qpop.append(torch.softmax(torch.stack(ll, 1), 1))                                          # (J_s, M)
        xyz_pop.append(torch.from_numpy(np.load(DATA / f"s{s}_coords_mni.npy").astype(np.float32)).to(DEV)[good])
    Cpop, qpop, xyz_pop = torch.cat(Cpop, 2), torch.cat(qpop), torch.cat(xyz_pop)
    info["pop_mask_mass"] = dict(zip(["s0.7", "s1.4", "s2.5", "uniform"],
                                     [qpop[:, :49].sum(1).mean().item(), qpop[:, 49:65].sum(1).mean().item(),
                                      qpop[:, 65:69].sum(1).mean().item(), qpop[:, 69].mean().item()]))
    print("[population] mean posterior mass per mask family:", info["pop_mask_mass"], flush=True)
    del U_tr, betas_tr
    torch.cuda.empty_cache()

    # ---- held-out subject -------------------------------------------------------------------------------------
    s = a.holdout
    meta = np.load(DATA / f"s{s}_voxel_meta.npz")
    good = meta["ncsnr"] > 0.2
    xyz_v = torch.from_numpy(np.load(DATA / f"s{s}_coords_mni.npy").astype(np.float32)).to(DEV)
    B_u = torch.from_numpy(np.load(DATA / f"s{s}_betas_unique.npy")).to(DEV)
    true = torch.from_numpy(np.load(DATA / f"s{s}_betas_common.npy")).to(DEV).T                     # (V, Q)
    with torch.inference_mode():
        pool = lambda ids: torch.cat([(torch.einsum("mp,bpk->bmk", A, g(store(ids[i:i + 500]))) - u_mean) / u_std
                                      for i in range(0, len(ids), 500)])
        U_u, U_q = pool(np.load(DATA / f"s{s}_nsdid_unique.npy")), pool(np.load(DATA / f"s{s}_nsdid_common.npy"))
    prior = (1 - a.prior_floor) * kernel_average(xyz_v, xyz_pop, qpop, a.sigma_mm) + a.prior_floor / M   # (V, M)
    tau2 = [a.tau2_ratio * Cpop[m].var(1).mean().item() for m in range(M)]
    flat = list(range(49, 65))                                                                      # the 16 sigma-1.4 masks: 16 x K flat control

    # PCA-512 baselines
    basis = train_pca_basis(train_subj, a, out)
    pca_tr = [unit_norm(np.load(PCA / f"s{t}_unique_img_feats_normed.npy").astype(np.float32)) for t in train_subj]
    F_stats = feature_stats(basis, pca_tr, DEV)
    pop_pca, tmp_xyz = [], []
    for t, img in zip(train_subj, pca_tr):
        gd = np.load(DATA / f"s{t}_voxel_meta.npz")["ncsnr"] > 0.2
        pop_pca.append(full_coefficients(basis, img, np.load(DATA / f"s{t}_betas_unique.npy"), DEV)[:, gd])
    pop_pca = torch.from_numpy(np.concatenate(pop_pca, 1)).to(DEV)
    tau2_pca = a.tau2_ratio * pop_pca.var(1).mean().item()
    P_u = unit_norm(np.load(PCA / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
    P_q = unit_norm(np.load(PCA / "common_img_feats_normed.npy").astype(np.float32))
    with torch.inference_mode():
        Fp_u = (basis(torch.from_numpy(P_u).to(DEV)) - F_stats[0]) / F_stats[1]
        Fp_q = (basis(torch.from_numpy(P_q).to(DEV)) - F_stats[0]) / F_stats[1]

    order = np.random.default_rng(100).permutation(B_u.shape[0])
    rows, start = [], 0
    for n in a.num_ic:
        for rep in range(a.repeats):
            idx = torch.from_numpy(order[start:start + n]).to(DEV)
            start += n
            t0 = time.time()
            Bs = B_u[idx]
            Bm = Bs.mean(0, keepdim=True)
            Bc = Bs - Bm
            preds = {}
            with torch.inference_mode():
                sig2 = torch.stack([ridge_fixed(U_u[idx, m], Bc, a.lam0)[1] for m in range(M)], 1)   # (V, M)
                ll = -0.5 * n * torch.log(sig2)
                q = torch.softmax(torch.log(prior) + ll, 1)
                q_flat = torch.softmax(ll, 1)
                hard = F_.one_hot(q.argmax(1), M).float()
                acc = {k: torch.zeros_like(true) for k in ("sflm_bma", "sflm_bma_noprior", "sflm_argmax", "sflm_prior_only")}
                for m in range(M):
                    C = popeb(U_u[idx, m], Bc, Cpop[m], tau2[m], a.lam0, spatial=(xyz_v, xyz_pop, a.sigma_mm))
                    pm = (Bm + U_q[:, m] @ C).T                                                      # (V, Q)
                    for k, w in (("sflm_bma", q), ("sflm_bma_noprior", q_flat), ("sflm_argmax", hard), ("sflm_prior_only", prior)):
                        acc[k] += w[:, m:m + 1] * pm
                    if m == M - 1:
                        preds["sflm_uniform_mask"] = pm
                preds.update(acc)
                C = popeb(Fp_u[idx], Bc, pop_pca, tau2_pca, a.lam0, spatial=(xyz_v, xyz_pop, a.sigma_mm))
                preds["popeb_v2_pca512"] = (Bm + Fp_q @ C).T
            Xf, Xfq = U_u[idx][:, flat].reshape(n, -1), U_q[:, flat].reshape(U_q.shape[0], -1)
            w = ridge_cv_brainwise(Xf, Bs.T)
            preds["ridge_flat_latent"] = (torch.cat([Xfq, torch.ones(Xfq.shape[0], 1, device=DEV)], 1) @ w.T).T
            w = ridge_cv_brainwise(torch.from_numpy(P_u).to(DEV)[idx], Bs.T)
            preds["ridge_pca512"] = (torch.cat([torch.from_numpy(P_q).to(DEV), torch.ones(P_q.shape[0], 1, device=DEV)], 1) @ w.T).T
            for k, p in preds.items():
                ev, r = explained_variance(p, true).cpu().numpy(), pearson(p, true).cpu().numpy()
                np.save(out / f"{k}_n={n}_rep={rep}_ev.npy", ev.astype(np.float32))
                row = dict(method=k, n=n, rep=rep, ev=float(ev[good].mean()), r=float(r[good].mean()), ev_all=float(ev.mean()))
                row.update({f"ev_{name}": float(ev[good & np.isin(meta["streams"], v)].mean()) for name, v in STREAMS.items()})
                rows.append(row)
            print(f"[eval] n={n} rep={rep} {time.time() - t0:.0f}s " + " ".join(f"{r_['method']}={r_['ev']:.4f}" for r_ in rows[-len(preds):]), flush=True)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.groupby(["method", "n"]).mean(numeric_only=True).drop(columns=["rep"]).round(4))
    with open(out / "summary.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
