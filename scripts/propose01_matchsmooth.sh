#!/bin/sh
# docs/results.md E18: PopEB-v2 + neighbour-smoothed matching (fast.match_sigma = 1.2 mm) on all four hold-outs,
# same support draws and BraInCoRL reruns as E07/E08.  sh scripts/propose01_matchsmooth.sh
set -e
CFG=src/config/popeb_clip.yaml; OUT=outputs/propose01; MS=fast.match_sigma=1.2
run() {  # H CK COMPARE_ic100 COMPARE_scaling OUTTAG OVR
  H=$1; CK=$2; C100=$3; CSC=$4; TAG=$5; OVR=$6; META=data/braincorl/s${H}_voxel_meta.npz
  python -m src.eval_lowrank --config $CFG --ckpt_dir $CK --subject $H --num_ic 100 --repeats 5 --popeb \
      --voxel_meta $META --compare_dir $C100 --out_dir $OUT/matchsmooth/s${H}_ic100 $MS $OVR > $OUT/matchsmooth/s${H}_ic100.log 2>&1
  python -m src.eval_lowrank --config $CFG --ckpt_dir $CK --subject $H --num_ic 30 300 1000 --repeats 3 --popeb \
      --voxel_meta $META --compare_dir $CSC --out_dir $OUT/matchsmooth/s${H}_scaling $MS $OVR > $OUT/matchsmooth/s${H}_scaling.log 2>&1
}
mkdir -p $OUT/matchsmooth
run 1 $OUT/sweep/K32_a1 $OUT/braincorl_s1_ic100 $OUT/braincorl_s1_scaling s1 ""
run 2 $OUT/holdout/s2/K32 $OUT/holdout/s2/braincorl_ic100 $OUT/holdout/s2/braincorl_scaling s2 "data.source.train_subjects=[1,5,7] data.source.val_subjects=[1]"
run 5 $OUT/holdout/s5/K32 $OUT/holdout/s5/braincorl_ic100 $OUT/holdout/s5/braincorl_scaling s5 "data.source.train_subjects=[1,2,7] data.source.val_subjects=[1]"
run 7 $OUT/holdout/s7/K32 $OUT/holdout/s7/braincorl_ic100 $OUT/holdout/s7/braincorl_scaling s7 "data.source.train_subjects=[1,2,5] data.source.val_subjects=[1]"
echo MATCHSMOOTH_DONE
