#!/bin/sh
# docs/propose/propose01.md: selection of the basis dimension K for the low-rank encoder.
# (1) nested leave-one-training-subject-out: for each K train the slow stage on two of S2/S5/S7
#     and run the fast stage (100 support x 3 draws) on the third; S1 is never used.
# (2) S1 evaluation of every K trained on S2/S5/S7 (sensitivity table, paired with BraInCoRL).
# History: the 2026-09-11 run (outputs/propose01/sweep*, K x alpha, variants A-D) was made by an
# earlier version that also swept an NMDA alpha and Hebbian-memory variants; alpha=1 there equals
# the SiLU basis kept here, so outputs/propose01/sweep/K*_a1 are the checkpoints of this script.
set -e
CFG=src/config/lowrank_clip.yaml
OUT=outputs/propose01
for K in 16 32 64 128; do
  for H in 2 5 7; do
    TR=$(echo "2 5 7" | tr ' ' '\n' | grep -v "^$H$" | tr '\n' ',' | sed 's/,$//')
    D=$OUT/sweep_nested/K${K}_a1_h${H}
    [ -f $D/basis.pt ] || python -m src.train_lowrank --config $CFG --out_dir $D \
        basis.basis_dim=$K "data.source.train_subjects=[$TR]" > $D.log 2>&1
    python -m src.eval_lowrank --config $CFG --ckpt_dir $D --subject $H --num_ic 100 --repeats 3 \
        --voxel_meta data/braincorl/s${H}_voxel_meta.npz --out_dir $D/eval_lowrank_s$H > $D.eval_lowrank.log 2>&1
  done
  D=$OUT/sweep/K${K}_a1
  [ -f $D/basis.pt ] || python -m src.train_lowrank --config $CFG --out_dir $D basis.basis_dim=$K > $D.log 2>&1
  python -m src.eval_lowrank --config $CFG --ckpt_dir $D --subject 1 --num_ic 100 --repeats 5 --ridge \
      --voxel_meta data/braincorl/s1_voxel_meta.npz --compare_dir $OUT/braincorl_s1_ic100 \
      --out_dir $D/eval_lowrank_s1 > $D.eval_lowrank.log 2>&1
done
echo SWEEP_DONE
