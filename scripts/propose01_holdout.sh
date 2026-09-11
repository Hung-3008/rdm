#!/bin/sh
# Hold-out protocol for one subject H in {1,2,5,7} (docs/propose/propose01.md 9.7): train the K16 and K32 bases
# on the other three subjects, rerun BraInCoRL with the checkpoint trained on those three, then evaluate
# variant A, PopEB (K16) and PopEB-v2 (K32) on H with the same support draws (100 x 5; 30/300/1000 x 3).
#   sh scripts/propose01_holdout.sh 2
set -e
H=$1
case $H in
  1) TR="2,5,7"; CK=CLIP_trained_on_s257 ;;
  2) TR="1,5,7"; CK=CLIP_trained_on_s157 ;;
  5) TR="1,2,7"; CK=CLIP_trained_on_s127 ;;
  7) TR="1,2,5"; CK=CLIP_trained_on_s125 ;;
  *) echo "unknown subject $H"; exit 1 ;;
esac
V=$(echo $TR | cut -d, -f1)
OUT=outputs/propose01/holdout/s$H
META=data/braincorl/s${H}_voxel_meta.npz
mkdir -p $OUT
OVR="data.source.train_subjects=[$TR] data.source.val_subjects=[$V]"

for K in 16 32; do
  [ -f $OUT/K$K/basis.pt ] || python -m src.train_lowrank --config src/config/lowrank_clip.yaml --out_dir $OUT/K$K \
      basis.basis_dim=$K $OVR > $OUT/K$K.train.log 2>&1
done

BC=references/BraInCoRL/checkpoints/$CK.pth
[ -f $OUT/braincorl_ic100/results.csv ] || python -m src.eval --config src/config/braincorl_clip_3090.yaml --ckpt $BC --original_ckpt \
    --subject $H --num_ic 100 --repeats 5 --voxel_meta $META --out_dir $OUT/braincorl_ic100 $OVR > $OUT/braincorl_ic100.log 2>&1
[ -f $OUT/braincorl_scaling/results.csv ] || python -m src.eval --config src/config/braincorl_clip_3090.yaml --ckpt $BC --original_ckpt \
    --subject $H --num_ic 30 300 1000 --repeats 3 --voxel_meta $META --out_dir $OUT/braincorl_scaling $OVR > $OUT/braincorl_scaling.log 2>&1

# K16: variant A (lowrank) + PopEB; K32: PopEB-v2 (lowrank row there = A with K32, for reference only)
for SET in "100 --repeats 5 ic100" "30 300 1000 --repeats 3 scaling"; do
  NIC=$(echo "$SET" | sed 's/ --repeats.*//'); REP=$(echo "$SET" | sed 's/.*--repeats \([0-9]*\).*/\1/'); TAG=$(echo "$SET" | awk '{print $NF}')
  python -m src.eval_lowrank --config src/config/lowrank_clip.yaml --ckpt_dir $OUT/K16 --subject $H --num_ic $NIC --repeats $REP \
      --popeb --ridge --voxel_meta $META --compare_dir $OUT/braincorl_$TAG --out_dir $OUT/popeb_k16_$TAG $OVR > $OUT/popeb_k16_$TAG.log 2>&1
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir $OUT/K32 --subject $H --num_ic $NIC --repeats $REP \
      --popeb --voxel_meta $META --compare_dir $OUT/braincorl_$TAG --out_dir $OUT/popeb_v2_$TAG $OVR > $OUT/popeb_v2_$TAG.log 2>&1
done
echo HOLDOUT_S${H}_DONE
