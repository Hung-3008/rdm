#!/bin/sh
# E09 (docs/results.md): PopEB-v2 / A / Ridge on hold-out S1 with another frozen backbone.
# Requires the layout data/braincorl/<BACKBONE>/ (src.tools.preprocess_nsd --stage feats/gt/layout --backbone <BACKBONE>).
#   sh scripts/propose01_backbone.sh dinov2_base
set -e
BB=$1
OUT=outputs/propose01/backbone_$BB
META=data/braincorl/s1_voxel_meta.npz
OVR="data.source.data_dir=data/braincorl/$BB"
mkdir -p $OUT
for K in 16 32; do
  [ -f $OUT/K$K/basis.pt ] || python -m src.train_lowrank --config src/config/lowrank_clip.yaml --out_dir $OUT/K$K basis.basis_dim=$K $OVR > $OUT/K$K.train.log 2>&1
done
for SET in "100 --repeats 5 ic100" "30 300 1000 --repeats 3 scaling"; do
  NIC=$(echo "$SET" | sed 's/ --repeats.*//'); REP=$(echo "$SET" | sed 's/.*--repeats \([0-9]*\).*/\1/'); TAG=$(echo "$SET" | awk '{print $NF}')
  python -m src.eval_lowrank --config src/config/lowrank_clip.yaml --ckpt_dir $OUT/K16 --subject 1 --num_ic $NIC --repeats $REP \
      --popeb --ridge --voxel_meta $META --out_dir $OUT/popeb_k16_$TAG $OVR > $OUT/popeb_k16_$TAG.log 2>&1
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir $OUT/K32 --subject 1 --num_ic $NIC --repeats $REP \
      --popeb --voxel_meta $META --out_dir $OUT/popeb_v2_$TAG $OVR > $OUT/popeb_v2_$TAG.log 2>&1
done
echo BACKBONE_${BB}_DONE
