#!/bin/sh
# Nested selection of the PopEB-v2 hyperparameters (tau2_ratio, lam0, spatial_sigma) for a feature layout
# (docs/results_new.md P4). For outer hold-out H, each of its three training subjects t is the inner target, with the
# basis and the population from the other two; the config with the best mean EV (ncsnr > 0.2, n = 100 x 5) over the three
# inner targets is written to selected.json[H]. The outer hold-out subject is never used for its own selection.
#   sh scripts/clip_layers_hparam_nested.sh LAYOUT
set -e
cd "$(dirname "$0")/.."
L=$1; NG=${ROOT:-data/braincorl_nsdgeneral}; DATA=$NG/$L; CFG=src/config/popeb_clip.yaml
OUT=outputs/clip_layers/nested_hp/$L
TAU="0.0057 0.0113 0.0227 0.0454 0.0908"; LAM="1 10 100"; SIG="5 10 20"
mkdir -p $OUT
for P in "1 2" "1 5" "1 7" "2 5" "2 7" "5 7"; do
  set -- $P; PT="$1,$2"; B=$OUT/p$1$2
  [ -f $B/basis.pt ] || python -m src.train_lowrank --config $CFG --out_dir $B data.source.data_dir=$DATA "data.source.train_subjects=[$PT]" > $B.train.log 2>&1
  for T in 1 2 5 7; do
    [ $T = $1 ] || [ $T = $2 ] && continue
    mkdir -p $OUT/p$1$2_t$T
    for tau in $TAU; do for lam in $LAM; do for sig in $SIG; do
      E=$OUT/p$1$2_t$T/tau${tau}_lam${lam}_sig$sig
      [ -f $E/results.csv ] || python -m src.eval_lowrank --config $CFG --ckpt_dir $B --subject $T --num_ic 100 --repeats 5 --popeb \
          --voxel_meta $NG/s${T}_voxel_meta.npz --popeb_meta_dir $NG --out_dir $E data.source.data_dir=$DATA "data.source.train_subjects=[$PT]" \
          fast.tau2_ratio=$tau fast.lam0=$lam fast.spatial_sigma=$sig > $E.log 2>&1
    done; done; done
  done
done
python - <<PY
import glob, json, re
import pandas as pd
rows = []
for p in glob.glob("$OUT/p*_t*/tau*/results.csv"):
    pair, t, tau, lam, sig = re.search(r"p(\d\d)_t(\d)/tau([\d.]+)_lam([\d.]+)_sig([\d.]+)/", p).groups()
    df = pd.read_csv(p)
    df = df[df.method == "lowrank_popeb"]
    rows.append(dict(pair=pair, t=int(t), tau=float(tau), lam=float(lam), sig=float(sig), ev=df["ev_ncsnr>0.2"].mean(), ev_all=df.expl_var.mean()))
df = pd.DataFrame(rows)
df.to_csv("$OUT/grid.csv", index=False)
sel = {}
for H in (1, 2, 5, 7):
    # inner combos of outer H: pair and target both inside the training subjects of H
    d = df[df.apply(lambda r: str(H) not in r.pair and r.t != H, axis=1)]
    g = d.groupby(["tau", "lam", "sig"])[["ev", "ev_all"]].mean()
    assert d.groupby(["tau", "lam", "sig"]).size().min() == 3, "incomplete grid"
    best = g.ev.idxmax()
    dflt = g.loc[(0.0227, 10.0, 10.0)]
    sel[H] = dict(tau2_ratio=best[0], lam0=best[1], spatial_sigma=best[2], inner_ev=float(g.ev.max()), inner_ev_default=float(dflt.ev))
    print(f"H={H} best tau={best[0]} lam0={best[1]} sigma={best[2]} inner EV(ncsnr>0.2) {g.ev.max():.4f} vs default {dflt.ev:.4f}")
json.dump(sel, open("$OUT/selected.json", "w"), indent=1)
PY
