#!/bin/sh
# Few-shot test of a multi-layer CLIP feature variant (scripts/clip_layers_ceiling.py --layout VARIANT must exist):
# basis K32 on S2/S5/S7 (popeb_clip.yaml), PopEB-v2 on hold-out S1 on the E20 support draws (100 x 5; 30/300/1000 x 3),
# paired against BraInCoRL-scratch (outputs/repro/clip_b16/eval_s1_*). Ridge on the raw variant only when RIDGE=1
# (ridge_cv_brainwise stores fold x lambda x V x E weights; fits in 24 GB for E <= ~1500).
#   sh scripts/clip_layers_fewshot.sh VARIANT [RIDGE=1]
set -e
cd "$(dirname "$0")/.."
V=$1; NG=data/braincorl_nsdgeneral; DATA=$NG/clip_$V; OUT=outputs/clip_layers/fewshot/$V; CFG=src/config/popeb_clip.yaml
mkdir -p $OUT
[ -f $OUT/K32/basis.pt ] || python -m src.train_lowrank --config $CFG --out_dir $OUT/K32 data.source.data_dir=$DATA > $OUT/K32.train.log 2>&1
RIDGE_FLAG=""; [ "$RIDGE" = "1" ] && RIDGE_FLAG="--ridge"
for SET in "100 5 ic100" "30_300_1000 3 scaling"; do
  set -- $SET; NIC=$(echo $1 | tr _ ' ')
  [ -f $OUT/eval_$3/results.csv ] || python -m src.eval_lowrank --config $CFG --ckpt_dir $OUT/K32 --subject 1 --num_ic $NIC --repeats $2 \
      --popeb $RIDGE_FLAG --voxel_meta $NG/s1_voxel_meta.npz --popeb_meta_dir $NG --compare_dir outputs/repro/clip_b16/eval_s1_$3 \
      --out_dir $OUT/eval_$3 data.source.data_dir=$DATA > $OUT/eval_$3.log 2>&1
done
echo "[done] $OUT"
