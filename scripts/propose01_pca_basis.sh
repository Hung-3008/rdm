#!/bin/sh
# PopEB-v2 (src/config/popeb_clip.yaml, unchanged) with a PCA basis instead of the MLP basis, pooled CLIP features,
# nsdgeneral layout, same support draws as scripts/repro_braincorl.sh (outputs/repro/clip_b16/popeb_s<H>_*).
# Usage: sh scripts/propose01_pca_basis.sh [K=32] [hold-outs="1 2 5 7"]
set -e
K=${1:-32}; HS=${2:-"1 2 5 7"}
NG=data/braincorl_nsdgeneral; DATA=$NG/clip_b16; REF=outputs/repro/clip_b16; OUT=outputs/propose01/pca_basis/K$K
mkdir -p $OUT
for H in $HS; do
  case $H in 1) TR="2 5 7" ;; 2) TR="1 5 7" ;; 5) TR="1 2 7" ;; 7) TR="1 2 5" ;; esac
  SUBJ="data.source.train_subjects=[$(echo $TR | tr ' ' ',')] data.source.val_subjects=[$(echo $TR | tr ' ' ',')]"
  python scripts/propose01_pca_basis.py build --data_dir $DATA --train $TR --K $K --out $OUT/s$H/basis.pt
  for SET in "100 --repeats 5 ic100" "30 300 1000 --repeats 3 scaling"; do
    NIC=$(echo "$SET" | sed 's/ --repeats.*//'); REP=$(echo "$SET" | sed 's/.*--repeats \([0-9]*\).*/\1/'); TAG=$(echo "$SET" | awk '{print $NF}')
    python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir $OUT/s$H --subject $H --num_ic $NIC --repeats $REP --popeb \
        --voxel_meta $NG/s${H}_voxel_meta.npz --popeb_meta_dir $NG --compare_dir $REF/eval_s${H}_$TAG \
        --out_dir $OUT/popeb_s${H}_$TAG data.source.data_dir=$DATA $SUBJ > $OUT/popeb_s${H}_$TAG.log 2>&1
  done
  python scripts/propose01_pca_basis.py report --pca_dir $OUT/popeb_s$H --mlp_dir $REF/popeb_s$H --voxel_meta $NG/s${H}_voxel_meta.npz --name "S$H K$K"
done
