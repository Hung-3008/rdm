#!/bin/sh
# PopEB-v2 on one hold-out subject for a feature layout (docs/results_new.md P2-P6): basis K32 on the three other subjects
# (popeb_clip.yaml), eval on the E20 support draws (100 x 5; 30/300/1000 x 3), EV + Pearson, PopEB and ridge.
#   sh scripts/clip_layers_holdout.sh LAYOUT H [TAG]
# LAYOUT: directory name under ROOT (e.g. clip_pca512_clsmean_all). Env:
#   ROOT=dir       data root (default data/braincorl_nsdgeneral)
#   RIDGE=1        also fit within-subject ridge (method "ridge")
#   SEED=s         basis seed (basis dir K32_seed<s>)
#   BASIS_OV="k=v" train_lowrank overrides (e.g. basis.val_images=600); pair it with BTAG so the basis gets its own dir
#   FAST="k=v .."  eval overrides (e.g. fast.tau2_ratio=0.01 fast.lam0=10 fast.spatial_sigma=10), used with TAG
#   COMPARE=dir    paired comparison prefix: <dir>/eval_s<H>_{ic100,scaling} (default outputs/repro/clip_b16, if present)
# Outputs: outputs/clip_layers/holdout[_<basename ROOT>]/<LAYOUT>/s<H>/{K32[_seed<s>], <TAG>[_seed<s>]_{ic100,scaling}}
set -e
cd "$(dirname "$0")/.."
L=$1; H=$2; TAG=${3:-default}
NG=${ROOT:-data/braincorl_nsdgeneral}; DATA=$NG/$L; CFG=src/config/popeb_clip.yaml
OUT=outputs/clip_layers/holdout/$L/s$H; [ -n "$ROOT" ] && OUT=outputs/clip_layers/holdout_$(basename $NG)/$L/s$H
case $H in 1) TR="2,5,7" ;; 2) TR="1,5,7" ;; 5) TR="1,2,7" ;; 7) TR="1,2,5" ;; *) echo "bad hold-out $H"; exit 1 ;; esac
SUBJ="data.source.train_subjects=[$TR] data.source.val_subjects=[$TR]"
B=$OUT/K32$BTAG; SFX=""; [ -n "$SEED" ] && { B=${B}_seed$SEED; SFX=_seed$SEED; }
mkdir -p $OUT
[ -f $B/basis.pt ] || python -m src.train_lowrank --config $CFG --out_dir $B data.source.data_dir=$DATA $SUBJ seed=${SEED:-0} $BASIS_OV > $B.train.log 2>&1
RIDGE_FLAG=""; [ "$RIDGE" = "1" ] && RIDGE_FLAG="--ridge"
for SET in "100 5 ic100" "30_300_1000 3 scaling"; do
  set -- $SET; NIC=$(echo $1 | tr _ ' ')
  E=$OUT/$TAG${SFX}_$3
  CMP=${COMPARE:-outputs/repro/clip_b16}/eval_s${H}_$3; CMP_FLAG=""; [ -f $CMP/results.csv ] && CMP_FLAG="--compare_dir $CMP"
  [ -f $E/results.csv ] || python -m src.eval_lowrank --config $CFG --ckpt_dir $B --subject $H --num_ic $NIC --repeats $2 \
      --popeb $RIDGE_FLAG $CMP_FLAG --voxel_meta $NG/s${H}_voxel_meta.npz --popeb_meta_dir $NG --out_dir $E \
      data.source.data_dir=$DATA $SUBJ $FAST > $E.log 2>&1
done
echo "[done] $OUT $TAG$SFX"
