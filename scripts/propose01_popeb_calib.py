"""Nested calibration of the PopEB bandwidth tau2 (docs/propose/propose01.md, 9.2).

For each training subject h in the config: population = full-data coefficients of the other training
subjects' ncsnr > 0.2 voxels; support = n images drawn from h's unique images (seeded); query = h's
1,000 common images; EV on h's ncsnr > 0.2 voxels. S1 (or any held-out subject) is never used.

    python scripts/propose01_popeb_calib.py --config src/config/lowrank_clip.yaml \
        --ckpt_dir outputs/propose01/sweep/K16_a1 [--tau2 0.0002 0.0005 0.001 0.002 0.005] [--num_ic 30 100 300 1000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.config import load_config  # noqa: E402
from src.model.lowrank_basis import FunctionalBasis, LowRankConfig, full_coefficients, popeb, ridge_loo  # noqa: E402
from src.utils import explained_variance, unit_norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt_dir", required=True)
ap.add_argument("--tau2", type=float, nargs="+", default=[0.0002, 0.0005, 0.001, 0.002, 0.005])
ap.add_argument("--num_ic", type=int, nargs="+", default=[30, 100, 300, 1000])
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--meta_dir", default="data/braincorl")
args = ap.parse_args()

cfg = load_config(args.config, [])
src = cfg["data"]["source"]
D = Path(src["data_dir"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
basis = FunctionalBasis.from_file(Path(args.ckpt_dir) / "basis.pt", dev)
fcfg = LowRankConfig(**cfg.get("fast", {}))
qimg = torch.from_numpy(unit_norm(np.load(D / "common_img_feats_normed.npy").astype(np.float32))).to(dev)

subs = {}
with torch.no_grad():
    for s in src["train_subjects"]:
        img = unit_norm(np.load(D / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
        beta = np.load(D / f"s{s}_unique_nrn_feats_msk.npy", mmap_mode="r")
        good = np.load(Path(args.meta_dir) / f"s{s}_voxel_meta.npz")["ncsnr"] > 0.2
        Cs = full_coefficients(basis, img, np.array(beta, dtype=np.float32), dev)
        subs[s] = dict(img=torch.from_numpy(img).to(dev), beta=beta, good=torch.from_numpy(good).to(dev),
                       Cpop=torch.from_numpy(Cs[:, good]).to(dev),
                       qbeta=torch.from_numpy(np.load(D / f"s{s}_common_nrn_feats_msk.npy").astype(np.float32).T).to(dev))
        torch.cuda.empty_cache()

res = {n: {t: [] for t in args.tau2} for n in args.num_ic}
resA = {n: [] for n in args.num_ic}
with torch.no_grad():
    for h, d in subs.items():
        Cpop = torch.cat([subs[o]["Cpop"] for o in subs if o != h], 1)
        rng = np.random.default_rng(11 + h)
        for n in args.num_ic:
            for _ in range(args.repeats):
                idx = np.sort(rng.permutation(len(d["beta"]))[:n])
                B = torch.from_numpy(np.asarray(d["beta"][idx], dtype=np.float32)).to(dev)
                Fs, Fq = basis(d["img"][idx]), basis(qimg)
                Bm = B.mean(0, keepdim=True)
                Fm, Fsd = Fs.mean(0, keepdim=True), Fs.std(0, keepdim=True).clamp_min(1e-6)
                Fn, Fqn, Bc = (Fs - Fm) / Fsd, (Fq - Fm) / Fsd, B - Bm
                ev = lambda C: explained_variance((Bm + Fqn @ C).T, d["qbeta"])[d["good"]].mean().item()
                resA[n].append(ev(ridge_loo(Fn, Bc, fcfg.lambdas)[0]))
                for t in args.tau2:
                    res[n][t].append(ev(popeb(Fn, Bc, Cpop, t, fcfg.lam0)))
        print(f"held-out S{h} done (population {Cpop.shape[1]} voxels)")

print(f"\n## PopEB tau2 calibration: EV on held-out training subject (ncsnr > 0.2), mean over {len(subs)} subjects x {args.repeats} draws")
gain = {t: 0.0 for t in args.tau2}
for n in args.num_ic:
    means = {t: float(np.mean(v)) for t, v in res[n].items()}
    best = max(means, key=means.get)
    for t in args.tau2:
        gain[t] += means[t] - float(np.mean(resA[n]))
    print(f"n={n:4d}  A={np.mean(resA[n]):.4f}  " + "  ".join(f"tau2={t:g}: {m:.4f}" for t, m in means.items()) + f"   -> best {best:g}")
print(f"single tau2 maximising the mean gain over n: {max(gain, key=gain.get):g}")
