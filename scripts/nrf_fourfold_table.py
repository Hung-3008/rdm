"""Four-fold table of the SFLM comparison with the NRF rows added, paired by support set.

    python scripts/nrf_fourfold_table.py > outputs/nrf/fourfold_table.txt

Per-voxel EV files: ridge / PopEB-v2 / SFLM from outputs/sflm/h<H>/, BraInCoRL from outputs/sflm_braincorl/{pc344/,}h<H>/eval/,
NRF from outputs/nrf/ev/ft_h<H>/ (copied from PC344). The same support sets (default_rng(100).permutation(9000), consecutive
slices) and the same voxel mask (ncsnr > 0.2 of s<H>_voxel_meta.npz) as scripts/sflm_table.py and proposal §11.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/braincorl_nsdgeneral")
NS, REPS, HOLDOUTS = (30, 100, 300, 1000), (0, 1, 2), (1, 2, 5, 7)
BRAINCORL = {1: "outputs/sflm_braincorl/pc344/h1", 2: "outputs/sflm_braincorl/pc344/h2",
             5: "outputs/sflm_braincorl/h5", 7: "outputs/sflm_braincorl/h7"}
NRF_ROWS = {"NRF ens (paper)": "nrf_ft_ens", "NRF ens allsup": "nrf_ft_ens_allsup", "NRF avg": "nrf_ft_avg"}


def files(H, n, rep):
    s, b, f = Path(f"outputs/sflm/h{H}"), Path(BRAINCORL[H]) / "eval", Path(f"outputs/nrf/ev/ft_h{H}")
    out = {"Ridge": s / f"ridge_pca512_n={n}_rep={rep}_ev.npy", "BraInCoRL": b / f"icl_num_ic={n}_rep={rep}_expl_var.npy",
           "PopEB-v2": s / f"popeb_v2_pca512_n={n}_rep={rep}_ev.npy", "SFLM": s / f"sflm_bma_n={n}_rep={rep}_ev.npy"}
    out.update({k: f / f"{v}_n{n}_rep{rep}_expl_var.npy" for k, v in NRF_ROWS.items()})
    return out


def main():
    rows = []
    for H in HOLDOUTS:
        good = np.load(ROOT / f"s{H}_voxel_meta.npz")["ncsnr"] > 0.2
        for n in NS:
            for rep in REPS:
                for m, p in files(H, n, rep).items():
                    rows.append(dict(H=H, n=n, rep=rep, method=m, ev=float(np.load(p)[good].mean())))
    df = pd.DataFrame(rows)
    order = ["Ridge", "BraInCoRL", "NRF ens (paper)", "NRF ens allsup", "NRF avg", "PopEB-v2", "SFLM"]
    pd.set_option("display.width", 200)
    t = df.groupby(["H", "method", "n"]).ev.mean().unstack("n").reindex(order, level="method")
    print("EV on ncsnr > 0.2, mean of 3 support sets\n", t.round(4).to_string())
    print("\nmean over 4 folds\n", df.groupby(["method", "n"]).ev.mean().unstack("n").reindex(order).round(4).to_string())
    w = df.pivot_table(index=["H", "n", "rep"], columns="method", values="ev")
    print("\npaired differences: mean over 48 cells (4 folds x 4 n x 3 support sets), and cells > 0")
    for a, b in [("NRF avg", "BraInCoRL"), ("NRF ens (paper)", "BraInCoRL"), ("NRF avg", "NRF ens (paper)"),
                 ("PopEB-v2", "NRF avg"), ("SFLM", "NRF avg"), ("SFLM", "NRF ens (paper)"), ("SFLM", "BraInCoRL")]:
        d = w[a] - w[b]
        per_n = d.groupby(level="n").mean()
        print(f"  {a:>16} - {b:<16}: mean {d.mean():+.4f}, {int((d > 0).sum())}/48 > 0; by n: "
              + ", ".join(f"n={n} {v:+.4f}" for n, v in per_n.items()))
    df.to_csv("outputs/nrf/fourfold_cells.csv", index=False)


if __name__ == "__main__":
    main()
