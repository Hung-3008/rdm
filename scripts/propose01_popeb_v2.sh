#!/bin/sh
# docs/propose/propose01.md 9.6: PopEB-v2 (src/config/popeb_clip.yaml) on hold-out S1, paired with BraInCoRL reruns.
set -e
CFG=src/config/popeb_clip.yaml
CK=outputs/propose01/sweep/K32_a1
OUT=outputs/propose01
python -m src.eval_lowrank --config $CFG --ckpt_dir $CK --subject 1 --num_ic 100 --repeats 5 --popeb \
    --voxel_meta data/braincorl/s1_voxel_meta.npz --compare_dir $OUT/braincorl_s1_ic100 \
    --out_dir $OUT/popeb_v2_s1_ic100 > $OUT/popeb_v2_s1_ic100.log 2>&1
python -m src.eval_lowrank --config $CFG --ckpt_dir $CK --subject 1 --num_ic 30 300 1000 --repeats 3 --popeb \
    --voxel_meta data/braincorl/s1_voxel_meta.npz --compare_dir $OUT/braincorl_s1_scaling \
    --out_dir $OUT/popeb_v2_s1_scaling > $OUT/popeb_v2_s1_scaling.log 2>&1
echo POPEB_V2_DONE
