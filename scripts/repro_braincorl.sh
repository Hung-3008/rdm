#!/bin/sh
# BraInCoRL reproduction on the nsdgeneral voxel set (docs/research-log-2026-09.md section 2): 3 backbones x 4 hold-out subjects.
#   sh scripts/repro_braincorl.sh synth  BB              # stages 1-2 (synthetic), once per backbone
#   sh scripts/repro_braincorl.sh stage3 BB H            # stage 3 on the 3 other subjects, then eval on hold-out H
#   sh scripts/repro_braincorl.sh queue  BB H1 [H2 ...]  # synth (if missing) + stage3 for each H
# BB in {clip_b16, dinov2_base, siglip_so400m, clip_pca512_clsmean_all}; H in {1,2,5,7}. Outputs: outputs/repro/<BB>/{stage1,stage2,stage3_s<H>,eval_s<H>_*,popeb_s<H>_*}.
# Env: SEED=s (training seed, outputs/repro/<BB>_seed<s>); ROOT=dir (data root, default data/braincorl_nsdgeneral; a
# different root writes to outputs/repro_<basename ROOT>/<BB>); SKIP_POPEB=1 (BraInCoRL eval only).
# Stage 3 eval: src.eval (BraInCoRL ckpt with best val_expl_var + ridge) and PopEB-v2 (basis K32 trained on the same
# three subjects, src.eval_lowrank --popeb --compare_dir) on the same support draws (100 x 5; 30/300/1000 x 3), EV + Pearson.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # batch 64 x context 500 peaks ~20 GiB; avoids allocator thrash near 24 GB
CMD=$1; BB=$2; shift 2
case $BB in
  clip_b16)      PRESET=CLIP;   E=512;  BS=64; ACC=1 ;;
  dinov2_base)   PRESET=DINO;   E=768;  BS=32; ACC=2 ;;   # batch 64 OOM at context 500 (22.4 GiB allocated at step 1)
  siglip_so400m) PRESET=SIGLIP; E=1152; BS=32; ACC=2 ;;   # batch 64 does not fit 24 GB at context 500
  clip_pca512_*) PRESET=CLIP;   E=512;  BS=64; ACC=1 ;;   # multi-layer CLIP PCA-512 (scripts/clip_layers_ceiling.py --layout ... --gt)
  *) echo "unknown backbone $BB"; exit 1 ;;
esac
NG=${ROOT:-data/braincorl_nsdgeneral}; DATA=$NG/$BB
OUT=outputs/repro/$BB; [ -n "$ROOT" ] && OUT=outputs/repro_$(basename $NG)/$BB
MODEL="module.model.preset=$PRESET"; [ -n "$SEED" ] && { OUT=${OUT}_seed$SEED; MODEL="$MODEL seed=$SEED"; }
mkdir -p $OUT
best_ckpt() { ls "$1"/checkpoints/epoch=*-val_expl_var=*.ckpt | sort -t= -k3 -n | tail -1; }

synth() {
  [ -f $OUT/stage2_best.ckpt ] && return
  [ -f $OUT/stage1.done ] || { python -m src.train --config src/config/repro/stage1_synth.yaml out_dir=$OUT/stage1 $MODEL data.source.embed_dim=$E \
      data.batch_size=$BS trainer.accumulate_grad_batches=$ACC > $OUT/stage1.log 2>&1 && touch $OUT/stage1.done; }
  C1=$(best_ckpt $OUT/stage1); echo "[stage1 best] $C1"
  [ -f $OUT/stage2.done ] || { python -m src.train --config src/config/repro/stage2_synth_ctx.yaml out_dir=$OUT/stage2 module.init_weights="$C1" $MODEL data.source.embed_dim=$E \
      data.batch_size=$BS trainer.accumulate_grad_batches=$ACC > $OUT/stage2.log 2>&1 && touch $OUT/stage2.done; }
  C2=$(best_ckpt $OUT/stage2); echo "[stage2 best] $C2"
  python -c "import torch; c=torch.load('$C2',map_location='cpu',weights_only=False); torch.save({'state_dict':c['state_dict']}, '$OUT/stage2_best.ckpt')"
}
stage3() {
  H=$1
  case $H in 1) TR="2,5,7" ;; 2) TR="1,5,7" ;; 5) TR="1,2,7" ;; 7) TR="1,2,5" ;; *) echo "bad hold-out $H"; exit 1 ;; esac
  SUBJ="data.source.train_subjects=[$TR] data.source.val_subjects=[$TR]"
  # PATIENCE (default 10; S1 CLIP used 5): val loss oscillates after the phase transition (~epoch 10-20), patience 5 stopped S2 at
  # epoch 22 with val EV 0.004. An interrupted run resumes from last.ckpt (Lightning restores optimizer, scheduler and early-stopping state).
  RESUME=""; [ -f $OUT/stage3_s$H/checkpoints/last.ckpt ] && RESUME="--resume $OUT/stage3_s$H/checkpoints/last.ckpt"
  [ -f $OUT/stage3_s$H.done ] || { python -m src.train --config src/config/repro/stage3_real.yaml $RESUME out_dir=$OUT/stage3_s$H data.source.data_dir=$DATA \
      module.init_weights=$INIT $MODEL $SUBJ module.warmup_steps=500 data.batch_size=$BS trainer.accumulate_grad_batches=$ACC \
      trainer.early_stopping_patience=${PATIENCE:-10} >> $OUT/stage3_s$H.log 2>&1 && touch $OUT/stage3_s$H.done; }
  C3=$(best_ckpt $OUT/stage3_s$H); echo "[stage3 s$H best] $C3"
  META=$NG/s${H}_voxel_meta.npz
  for SET in "100 --repeats 5 ic100" "30 300 1000 --repeats 3 scaling"; do
    NIC=$(echo "$SET" | sed 's/ --repeats.*//'); REP=$(echo "$SET" | sed 's/.*--repeats \([0-9]*\).*/\1/'); TAG=$(echo "$SET" | awk '{print $NF}')
    [ -f $OUT/eval_s${H}_$TAG/results.csv ] || python -m src.eval --config src/config/braincorl_clip_3090.yaml --ckpt "$C3" --subject $H --num_ic $NIC --repeats $REP --ridge \
        --voxel_meta $META --out_dir $OUT/eval_s${H}_$TAG data.source.data_dir=$DATA $SUBJ > $OUT/eval_s${H}_$TAG.log 2>&1
  done
  echo "[done s$H] $OUT/eval_s${H}_*"
}
# NO_SYNTH=1: skip the synthetic stages and train stage 3 from random init (docs/research-log-2026-09.md section 2: the synthetic
# stages did not learn in-context regression within budget). The stage-3 init file is then "null".
INIT=$OUT/stage2_best.ckpt; [ "$NO_SYNTH" = "1" ] && INIT=null
case $CMD in
  synth)  synth ;;
  stage3) stage3 "$1" ;;
  queue)  [ "$NO_SYNTH" = "1" ] || synth; for H in "$@"; do stage3 $H; done ;;
  *) echo "usage: $0 {synth|stage3|queue} BACKBONE [H ...]"; exit 1 ;;
esac
echo REPRO_DONE_${BB}_$*
