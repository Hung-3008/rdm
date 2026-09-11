#!/bin/bash
# Three-stage from-scratch training (synthetic pretrain -> context extension -> real NSD).
#   scripts/train_scratch.sh braincorl        [extra overrides...]
#   scripts/train_scratch.sh braincorl_coord  [extra overrides...]
set -e
cd "$(dirname "$0")/.."
MODEL=$1; shift
COORD_ARGS=""
if [ "$MODEL" = "braincorl_coord" ]; then
  COORD_ARGS="module.model.num_freq=64 module.model.fourier_sigma=2.0 module.model.coord_scale_mm=100.0 module.model.jitter_mm=2.0 module.model.coord_token_dropout=0.1"
fi
best_ckpt() { ls "$1"/checkpoints/epoch=*-val_expl_var=*.ckpt | sort -t= -k3 -n | tail -1; }
OUT=outputs/scratch/$MODEL

python -m src.train --config src/config/scratch/stage1_synth.yaml out_dir=$OUT/stage1 module.model.name=$MODEL $COORD_ARGS "$@"
C1=$(best_ckpt $OUT/stage1); echo "[stage1 best] $C1"
python -m src.train --config src/config/scratch/stage2_synth_ctx.yaml out_dir=$OUT/stage2 module.model.name=$MODEL module.init_weights="$C1" $COORD_ARGS "$@"
C2=$(best_ckpt $OUT/stage2); echo "[stage2 best] $C2"
python -m src.train --config src/config/scratch/stage3_real.yaml out_dir=$OUT/stage3_s1 module.model.name=$MODEL module.init_weights="$C2" $COORD_ARGS "$@"
C3=$(best_ckpt $OUT/stage3_s1); echo "[stage3 best] $C3"
