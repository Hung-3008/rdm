#!/bin/sh
# docs/results.md E19: voxel set = NSD "nsdgeneral" mask (includes early visual), metrics EV and Pearson reported on
# all voxels ("all") and on ncsnr > 0.2. Hold-out S1; basis K32 trained on S2/S5/S7 of the same voxel set;
# BraInCoRL original checkpoint s257 (trained on higher visual cortex only). Same support draws as E00/E07.
#   sh scripts/propose01_nsdgeneral.sh
set -e
D=data/braincorl_nsdgeneral; L=$D/clip_b16; OUT=outputs/propose01/nsdgeneral; META=$D/s1_voxel_meta.npz
OVR="data.source.data_dir=$L"
mkdir -p $OUT

[ -f $OUT/K32/basis.pt ] || python -m src.train_lowrank --config src/config/popeb_clip.yaml --out_dir $OUT/K32 $OVR > $OUT/K32.train.log 2>&1

BC=references/BraInCoRL/checkpoints/CLIP_trained_on_s257.pth
[ -f $OUT/braincorl_ic100/results.csv ] || python -m src.eval --config src/config/braincorl_clip_3090.yaml --ckpt $BC --original_ckpt \
    --subject 1 --num_ic 100 --repeats 5 --ridge --voxel_meta $META --out_dir $OUT/braincorl_ic100 $OVR > $OUT/braincorl_ic100.log 2>&1
[ -f $OUT/braincorl_scaling/results.csv ] || python -m src.eval --config src/config/braincorl_clip_3090.yaml --ckpt $BC --original_ckpt \
    --subject 1 --num_ic 30 300 1000 --repeats 3 --ridge --voxel_meta $META --out_dir $OUT/braincorl_scaling $OVR > $OUT/braincorl_scaling.log 2>&1

for SET in "100 --repeats 5 ic100" "30 300 1000 --repeats 3 scaling"; do
  NIC=$(echo "$SET" | sed 's/ --repeats.*//'); REP=$(echo "$SET" | sed 's/.*--repeats \([0-9]*\).*/\1/'); TAG=$(echo "$SET" | awk '{print $NF}')
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir $OUT/K32 --subject 1 --num_ic $NIC --repeats $REP \
      --popeb --voxel_meta $META --popeb_meta_dir $D --compare_dir $OUT/braincorl_$TAG --out_dir $OUT/popeb_v2_$TAG $OVR > $OUT/popeb_v2_$TAG.log 2>&1
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir $OUT/K32 --subject 1 --num_ic $NIC --repeats $REP \
      --popeb --voxel_meta $META --popeb_meta_dir $D --compare_dir $OUT/braincorl_$TAG --out_dir $OUT/popeb_v2_ms12_$TAG $OVR fast.match_sigma=1.2 > $OUT/popeb_v2_ms12_$TAG.log 2>&1
done
echo NSDGENERAL_DONE
