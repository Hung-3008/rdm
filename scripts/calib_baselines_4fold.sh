#!/usr/bin/env bash
# Closed-form calibration baselines on the 4 hold-out folds; each fold uses the released
# BraInCoRL checkpoint that was not trained on the test subject.
set -e
declare -A CK=([1]=s257 [2]=s157 [5]=s127 [7]=s125)
for s in 1 2 5 7; do
  python scripts/calib_baselines.py --subject $s \
    --braincorl_ckpt references/BraInCoRL/checkpoints/CLIP_trained_on_${CK[$s]}.pth 2>&1 | grep -v "it/s\]"
done
