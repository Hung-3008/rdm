#!/bin/sh
# BraInCoRL baseline for the SFLM pilot: same image feature (multi-layer CLIP PCA-512), betas, 8,400 / 600 split and support
# draws as scripts/sflm_pilot.py (layouts from scripts/sflm_braincorl_layout.py). Stage 3 only, from random init (NO_SYNTH
# protocol of scripts/repro_braincorl.sh, same hyperparameters: warmup 500, patience 10 on val_loss).
#   sh scripts/sflm_braincorl.sh H1 [H2 ...]        # H in {1,2,5,7}; env PY=python binary, ROOT=data root
# Outputs: outputs/sflm_braincorl/h<H>/{train.log,checkpoints,eval/results.csv}; prints SFLM_BRAINCORL_DONE_h<H> per fold.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}; NG=${ROOT:-data/braincorl_nsdgeneral}
for H in "$@"; do
  case $H in 1) TR="2,5,7" ;; 2) TR="1,5,7" ;; 5) TR="1,2,7" ;; 7) TR="1,2,5" ;; *) echo "bad hold-out $H"; exit 1 ;; esac
  DATA=$NG/clip_pca512_sflm_h$H; OUT=outputs/sflm_braincorl/h$H; mkdir -p $OUT
  SUBJ="data.source.train_subjects=[$TR] data.source.val_subjects=[$TR]"
  RESUME=""; [ -f $OUT/checkpoints/last.ckpt ] && RESUME="--resume $OUT/checkpoints/last.ckpt"
  [ -f $OUT/train.done ] || { $PY -m src.train --config src/config/repro/stage3_real.yaml $RESUME out_dir=$OUT data.source.data_dir=$DATA \
      module.init_weights=null module.model.preset=CLIP $SUBJ module.warmup_steps=500 trainer.early_stopping_patience=10 >> $OUT/train.log 2>&1 && touch $OUT/train.done; }
  CK=$(ls $OUT/checkpoints/epoch=*-val_expl_var=*.ckpt | sort -t= -k3 -n | tail -1); echo "[h$H best] $CK"
  [ -f $OUT/eval/results.csv ] || $PY -m src.eval --config src/config/braincorl_clip_3090.yaml --ckpt "$CK" --subject $H --num_ic 30 100 300 1000 --repeats 3 --ridge \
      --voxel_meta $NG/s${H}_voxel_meta.npz --out_dir $OUT/eval data.source.data_dir=$DATA $SUBJ > $OUT/eval.log 2>&1
  echo SFLM_BRAINCORL_DONE_h$H
done
echo SFLM_BRAINCORL_ALL_DONE
