#!/bin/sh
# E13 (docs/results.md): population size on hold-out S1. Population from 3 (S2,S5,S7), 5 (+S3,S4) or 7
# (+S3,S4,S6,S8) subjects. (a) basis fixed = K32 trained on S2/S5/S7 (outputs/propose01/sweep/K32_a1), so only
# the population changes; (b) basis retrained on the same subjects as the population. Support draws as E07.
# Subjects 3/4/6/8 come from src.tools.preprocess_nsd (30-32 sessions; 8,481 / 7,000-ish unique images).
set -e
META=data/braincorl/s1_voxel_meta.npz
for POP in "2,5,7,3,4" "2,5,7,3,4,6,8"; do
  TAG=$(echo $POP | tr ',' '_')
  OVR="data.source.train_subjects=[$POP] data.source.val_subjects=[2]"
  # (a) fixed basis
  OUT=outputs/propose01/population_fixedbasis_$TAG; mkdir -p $OUT
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir outputs/propose01/sweep/K32_a1 --subject 1 --num_ic 100 --repeats 5 --popeb \
      --voxel_meta $META --compare_dir outputs/propose01/braincorl_s1_ic100 --out_dir $OUT/popeb_v2_ic100 $OVR > $OUT/popeb_v2_ic100.log 2>&1
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir outputs/propose01/sweep/K32_a1 --subject 1 --num_ic 30 300 1000 --repeats 3 --popeb \
      --voxel_meta $META --compare_dir outputs/propose01/braincorl_s1_scaling --out_dir $OUT/popeb_v2_scaling $OVR > $OUT/popeb_v2_scaling.log 2>&1
  # (b) basis retrained on the population subjects
  OUT=outputs/propose01/population_retrain_$TAG; mkdir -p $OUT
  [ -f $OUT/K32/basis.pt ] || python -m src.train_lowrank --config src/config/lowrank_clip.yaml --out_dir $OUT/K32 basis.basis_dim=32 $OVR > $OUT/K32.train.log 2>&1
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir $OUT/K32 --subject 1 --num_ic 100 --repeats 5 --popeb \
      --voxel_meta $META --compare_dir outputs/propose01/braincorl_s1_ic100 --out_dir $OUT/popeb_v2_ic100 $OVR > $OUT/popeb_v2_ic100.log 2>&1
  python -m src.eval_lowrank --config src/config/popeb_clip.yaml --ckpt_dir $OUT/K32 --subject 1 --num_ic 30 300 1000 --repeats 3 --popeb \
      --voxel_meta $META --compare_dir outputs/propose01/braincorl_s1_scaling --out_dir $OUT/popeb_v2_scaling $OVR > $OUT/popeb_v2_scaling.log 2>&1
done
echo POPULATION_DONE
