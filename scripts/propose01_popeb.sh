#!/bin/sh
# docs/propose/propose01.md section 9: PopEB on hold-out S1 with the K=16 basis trained on S2/S5/S7.
# (1) nested calibration of tau2 (S1 unused); (2) S1, 100 support x 5 draws; (3) S1 scaling 30/300/1000 x 3,
# both paired with the BraInCoRL reruns in outputs/propose01/braincorl_s1_*.
set -e
CFG=src/config/lowrank_clip.yaml
CK=outputs/propose01/sweep/K16_a1
OUT=outputs/propose01
python scripts/propose01_popeb_calib.py --config $CFG --ckpt_dir $CK > $OUT/popeb_calib.log 2>&1
python -m src.eval_lowrank --config $CFG --ckpt_dir $CK --subject 1 --num_ic 100 --repeats 5 --popeb \
    --voxel_meta data/braincorl/s1_voxel_meta.npz --compare_dir $OUT/braincorl_s1_ic100 \
    --out_dir $OUT/popeb_s1_ic100 > $OUT/popeb_s1_ic100.log 2>&1
python -m src.eval_lowrank --config $CFG --ckpt_dir $CK --subject 1 --num_ic 30 300 1000 --repeats 3 --popeb \
    --voxel_meta data/braincorl/s1_voxel_meta.npz --compare_dir $OUT/braincorl_s1_scaling \
    --out_dir $OUT/popeb_s1_scaling > $OUT/popeb_s1_scaling.log 2>&1
echo POPEB_DONE
