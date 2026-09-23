"""Headroom test H0 of docs/plan.md: prediction at coordinates without responses, held-out subject.

The fast stage of scripts/sflm_pilot.py is independent per voxel (ridge, PopEB and the mask posterior of voxel v
use only the betas of v and the training population), so on any masked voxel set
    E_own = mean of the pilot's saved per-voxel EV of sflm_bma (fit with the voxel's own betas)
    E_0   = mean of the per-voxel EV of M0 (coefficients and mask weights from the population only, n = 0)
and neither depends on which other voxels are masked. This script loads <dir>/latent.pt, rebuilds the population
exactly as the pilot does, computes M0 and tabulates E_own - E_0 for the masking schemes A-C of docs/plan.md.
    m0        c_m(x) = kernel average of Cpop[m], weights pi(x) = the pilot's mask prior (plan.md section 4)
    m0_joint  kernel average of q_jm * C_jm (the population voxels' full predictors), no separate mask prior
--check refits sflm_bma for n = 100, rep = 0 and compares with the saved per-voxel EV of the pilot.

    python scripts/sflm_inpaint.py --holdout 1 --dir outputs/sflm/h1 --check
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
from sflm_pilot import DATA, DEV, STREAMS, GridStore, LatentMap, kernel_average, mask_dictionary, ridge_fixed  # noqa: E402
from lowrank_basis import popeb  # noqa: E402
from src.utils.metrics import explained_variance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--dir", default="outputs/sflm/h1")
    ap.add_argument("--num_ic", type=int, nargs="+", default=[30, 100, 300, 1000])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--lam0", type=float, default=10.0)
    ap.add_argument("--tau2_ratio", type=float, default=0.0227)
    ap.add_argument("--sigma_mm", type=float, default=10.0)
    ap.add_argument("--prior_floor", type=float, default=0.1)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = Path(a.dir)
    train_subj = [s for s in (1, 2, 5, 7) if s != a.holdout]
    A = mask_dictionary().to(DEV)
    M = A.shape[0]
    store = GridStore()
    ck = torch.load(out / "latent.pt")
    g = LatentMap(ck["cfg"]["in_dim"], ck["cfg"]["hidden"], ck["cfg"]["K"],
                  torch.zeros(ck["cfg"]["in_dim"]), torch.ones(ck["cfg"]["in_dim"])).to(DEV)
    g.load_state_dict(ck["state"])
    g.eval()
    K = ck["cfg"]["K"]

    @torch.inference_mode()
    def pool(ids):
        return torch.cat([torch.einsum("mp,bpk->bmk", A, g(store(ids[i:i + 500]))) for i in range(0, len(ids), 500)])

    # population, as in sflm_pilot.main
    U_tr = [pool(np.load(DATA / f"s{s}_nsdid_unique.npy")) for s in train_subj]
    U_all = torch.cat(U_tr)
    u_mean, u_std = U_all.mean(0, keepdim=True), U_all.std(0, keepdim=True).clamp_min(1e-6)
    del U_all
    Cpop, qpop, xyz_pop = [], [], []
    with torch.inference_mode():
        for s, U in zip(train_subj, U_tr):
            B = torch.from_numpy(np.load(DATA / f"s{s}_betas_unique.npy")).to(DEV)
            good = torch.from_numpy(np.load(DATA / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2).to(DEV)
            Bc = (B - B.mean(0, keepdim=True))[:, good]
            Un = (U - u_mean) / u_std
            Cs, ll = [], []
            for m in range(M):
                C, loo = ridge_fixed(Un[:, m], Bc, 1.0)
                Cs.append(C)
                ll.append(-0.5 * Un.shape[0] * torch.log(loo))
            Cpop.append(torch.stack(Cs))
            qpop.append(torch.softmax(torch.stack(ll, 1), 1))
            xyz_pop.append(torch.from_numpy(np.load(DATA / f"s{s}_coords_mni.npy").astype(np.float32)).to(DEV)[good])
            del B
    Cpop, qpop, xyz_pop = torch.cat(Cpop, 2), torch.cat(qpop), torch.cat(xyz_pop)      # (M, K, J), (J, M), (J, 3)
    del U_tr
    torch.cuda.empty_cache()

    s = a.holdout
    meta = np.load(DATA / f"s{s}_voxel_meta.npz")
    good = meta["ncsnr"] > 0.2
    xyz = np.load(DATA / f"s{s}_coords_mni.npy").astype(np.float32)
    xyz_v = torch.from_numpy(xyz).to(DEV)
    true = torch.from_numpy(np.load(DATA / f"s{s}_betas_common.npy")).to(DEV).T        # (V, Q)
    U_q = (pool(np.load(DATA / f"s{s}_nsdid_common.npy")) - u_mean) / u_std             # (Q, M, K)
    prior = (1 - a.prior_floor) * kernel_average(xyz_v, xyz_pop, qpop, a.sigma_mm) + a.prior_floor / M

    # ---- M0: no response of the held-out subject is used ---------------------------------------------------------
    J = xyz_pop.shape[0]
    Cflat = Cpop.permute(2, 0, 1).reshape(J, M * K)                                     # (J, M*K)
    Dflat = (qpop[:, :, None] * Cpop.permute(2, 0, 1)).reshape(J, M * K)
    ev0 = {}
    with torch.inference_mode():
        c_bar = kernel_average(xyz_v, xyz_pop, Cflat, a.sigma_mm).reshape(-1, M, K)     # (V, M, K)
        d_bar = kernel_average(xyz_v, xyz_pop, Dflat, a.sigma_mm).reshape(-1, M, K)
        for name, W in (("m0", prior[:, :, None] * c_bar), ("m0_joint", d_bar)):
            pred = torch.cat([torch.einsum("vmk,qmk->vq", W[i:i + 2048], U_q) for i in range(0, W.shape[0], 2048)])
            ev0[name] = explained_variance(pred, true).cpu().numpy()
            np.save(out / f"{name}_ev.npy", ev0[name].astype(np.float32))

    if a.check:                                                                         # refit sflm_bma, n = 100, rep = 0
        B_u = torch.from_numpy(np.load(DATA / f"s{s}_betas_unique.npy")).to(DEV)
        U_u = (pool(np.load(DATA / f"s{s}_nsdid_unique.npy")) - u_mean) / u_std
        order = np.random.default_rng(100).permutation(B_u.shape[0])
        n, start = 100, 3 * 30
        idx = torch.from_numpy(order[start:start + n]).to(DEV)
        Bs = B_u[idx]
        Bm = Bs.mean(0, keepdim=True)
        Bc = Bs - Bm
        tau2 = [a.tau2_ratio * Cpop[m].var(1).mean().item() for m in range(M)]
        with torch.inference_mode():
            sig2 = torch.stack([ridge_fixed(U_u[idx, m], Bc, a.lam0)[1] for m in range(M)], 1)
            q = torch.softmax(torch.log(prior) - 0.5 * n * torch.log(sig2), 1)
            acc = torch.zeros_like(true)
            for m in range(M):
                C = popeb(U_u[idx, m], Bc, Cpop[m], tau2[m], a.lam0, spatial=(xyz_v, xyz_pop, a.sigma_mm))
                acc += q[:, m:m + 1] * (Bm + U_q[:, m] @ C).T
        ev = explained_variance(acc, true).cpu().numpy()
        ref = np.load(out / "sflm_bma_n=100_rep=0_ev.npy")
        print(f"[check] sflm_bma n=100 rep=0: refit {ev[good].mean():.4f}  pilot {ref[good].mean():.4f}  "
              f"max |diff| per voxel {np.abs(ev - ref)[good].max():.2e}", flush=True)

    # ---- E_own - E_0 per masking scheme ---------------------------------------------------------------------------
    own = {n: np.mean([np.load(out / f"sflm_bma_n={n}_rep={r}_ev.npy") for r in range(a.repeats)], 0) for n in a.num_ic}
    sets = {"all": good, "A_random50": good & (np.random.default_rng(0).random(len(good)) < 0.5)}
    for thick in (10, 20):
        z0 = np.floor(xyz[good, 2].min())
        for lo in np.arange(z0, xyz[good, 2].max(), thick):
            sets[f"B_slab{thick}_z{int(lo)}"] = good & (xyz[:, 2] >= lo) & (xyz[:, 2] < lo + thick)
    for name, v in STREAMS.items():
        sets[f"C_{name}"] = good & np.isin(meta["streams"], v)
    rows = []
    for name, sel in sets.items():
        row = dict(mask=name, n_vox=int(sel.sum()), E0=float(ev0["m0"][sel].mean()), E0_joint=float(ev0["m0_joint"][sel].mean()))
        for n in a.num_ic:
            row[f"Eown_{n}"] = float(own[n][sel].mean())
            row[f"gap_{n}"] = row[f"Eown_{n}"] - max(row["E0"], row["E0_joint"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "h0_headroom.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
