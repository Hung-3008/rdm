#!/bin/sh
# Authors' BraInCoRL checkpoints on nsdgeneral with the paired support sets (layout: scripts/braincorl_author_paired_layout.py).
#   sh scripts/braincorl_author_paired.sh [1 2 5 7]
set -e
cd "$(dirname "$0")/.."
NG=data/braincorl_nsdgeneral; OUT=outputs/braincorl_author_ckpt_paired; mkdir -p $OUT
for H in ${@:-1 2 5 7}; do
  case $H in 1) CK=s257 ;; 2) CK=s157 ;; 5) CK=s127 ;; 7) CK=s125 ;; *) echo "bad hold-out $H"; exit 1 ;; esac
  [ -f $OUT/s$H/results.csv ] && { echo "[s$H] exists, skip"; continue; }
  python -m src.eval --config src/config/braincorl_clip_3090.yaml --ckpt references/BraInCoRL/checkpoints/CLIP_trained_on_$CK.pth \
    --original_ckpt --subject $H --num_ic 30 100 300 1000 --repeats 3 --ridge --voxel_meta $NG/s${H}_voxel_meta.npz \
    --out_dir $OUT/s$H data.source.data_dir=$NG/clip_b16_paired > $OUT/s$H.log 2>&1
  echo "PAIRED_DONE_s$H"
done
echo PAIRED_ALL_DONE
