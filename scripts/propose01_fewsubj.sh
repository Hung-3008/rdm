#!/bin/sh
# E17 (docs/results.md): PopEB-v2 with basis AND population from only 1 or 2 training subjects, hold-out S1,
# paired with BraInCoRL trained on all three (s257). Support draws identical to E00/E07.
set -e
META=data/braincorl/s1_voxel_meta.npz
for TR in "2" "2,5"; do
  TAG=$(echo $TR | tr ',' '_'); OUT=outputs/propose01/fewsubj_s$TAG; mkdir -p $OUT
  OVR="data.source.train_subjects=[$TR] data.source.val_subjects=[2]"
  [ -f $OUT/K32/basis.pt ] || python -m src.train_lowrank --config src/config/lowrank_clip.yaml --out_dir $OUT/K32 basis.basis_dim=32 $OVR > $OUT/K32.train.log 2>&1
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir $OUT/K32 --subject 1 --num_ic 100 --repeats 5 --popeb \
      --voxel_meta $META --compare_dir outputs/propose01/braincorl_s1_ic100 --out_dir $OUT/popeb_v2_ic100 $OVR > $OUT/popeb_v2_ic100.log 2>&1
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir $OUT/K32 --subject 1 --num_ic 30 300 1000 --repeats 3 --popeb \
      --voxel_meta $META --compare_dir outputs/propose01/braincorl_s1_scaling --out_dir $OUT/popeb_v2_scaling $OVR > $OUT/popeb_v2_scaling.log 2>&1
done
echo FEWSUBJ_DONE
