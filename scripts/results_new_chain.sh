#!/bin/bash
# Experiment chain for the plan in docs/results_new.md section 6 (P1-P7), one GPU, every step idempotent.
#   bash scripts/results_new_chain.sh gpu        # all GPU steps in order (log: outputs/results_new/chain.log)
# P1 is BraInCoRL on the author recipe (val = training subjects' shared images); P1V repeats it with the PopEB-v2 split
# (val = 500 held-out unique images per training subject), which is the row used for the method comparison.
#   bash scripts/results_new_chain.sh hvc_prep   # CPU-only part of P7 (voxel set hvc -> data/braincorl_hvc), runs beside `gpu`
# Markers: "[STEP <name> start|done] <time>" in the log; outputs/results_new/<name>.done once finished.
set -e
cd "$(dirname "$0")/.."
L=clip_pca512_clsmean_all; NG=data/braincorl_nsdgeneral; HVC=data/braincorl_hvc; R=outputs/results_new
mkdir -p $R
step() {   # step NAME CMD...
  local n=$1; shift
  [ -f $R/$n.done ] && { echo "[STEP $n skip] $(date +%F\ %T)"; return; }
  echo "[STEP $n start] $(date +%F\ %T)"
  "$@"
  touch $R/$n.done; echo "[STEP $n done] $(date +%F\ %T)"
}
p2()   { for H in 1 2 5 7; do RIDGE=1 sh scripts/clip_layers_holdout.sh $L $H; done; }
p6a()  { for S in 1 2; do for H in 1 2 5 7; do SEED=$S sh scripts/clip_layers_holdout.sh $L $H; done; done; }
p5()   {
  for H in 1 2 5 7; do
    case $H in 1) TR="2 5 7" ;; 2) TR="1 5 7" ;; 5) TR="1 2 7" ;; 7) TR="1 2 5" ;; esac
    PYTHONPATH=. python scripts/clip_layers_ceiling.py --layout pca512_clsmean_all --fit_subjects $TR
    sh scripts/clip_layers_holdout.sh ${L}_fit$(echo $TR | tr -d ' ') $H
  done
}
p4()   {
  sh scripts/clip_layers_hparam_nested.sh $L
  for H in 1 2 5 7; do
    FAST=$(python -c "import json; s=json.load(open('outputs/clip_layers/nested_hp/$L/selected.json'))['$H']; print(f\"fast.tau2_ratio={s['tau2_ratio']} fast.lam0={s['lam0']} fast.spatial_sigma={s['spatial_sigma']}\")")
    FAST="$FAST" sh scripts/clip_layers_holdout.sh $L $H nested
  done
}
p1()   { NO_SYNTH=1 sh scripts/repro_braincorl.sh queue $L 1 2 5 7; }
# P2V: PopEB-v2 with the same 8,400 / 600 image split that P1V gives BraInCoRL (basis.val_images=600)
p2v()  { for H in 1 2 5 7; do BTAG=_val600 BASIS_OV=basis.val_images=600 RIDGE=1 sh scripts/clip_layers_holdout.sh $L $H val600; done; }
# P1V: BraInCoRL retrained with the PopEB-v2 train/val split (8,500 / 500 unique images of the training subjects)
p1v()  {
  python scripts/valuniq_layout.py --root $NG --layout $L --holdout 1 2 5 7 --val_images 600
  for H in 1 2 5 7; do NO_SYNTH=1 SKIP_POPEB=1 sh scripts/repro_braincorl.sh stage3 ${L}_valuniq_h$H $H; done
}
p6b()  { NO_SYNTH=1 SKIP_POPEB=1 SEED=1 sh scripts/repro_braincorl.sh stage3 ${L}_valuniq_h1 1; }
hvc_prep() {
  for st in mask coords betas; do python -m src.tools.preprocess_nsd --out $HVC --voxel_set hvc --stage $st; done
  for f in feats_clip_b16_73k.npy feats_clip_b16_cls_73k.npy feats_clip_b16_mean_73k.npy; do
    [ -e $HVC/$f ] || ln -s "$(readlink -f $NG/$f)" $HVC/$f
  done
}
p7()   {
  while [ ! -f $R/hvc_prep.done ]; do sleep 300; done
  [ -f $HVC/s7_gt.npy ] || python -m src.tools.preprocess_nsd --out $HVC --voxel_set hvc --stage gt --backbone clip_b16
  [ -f $HVC/clip_b16/test_time_icl_idx.json ] || python -m src.tools.preprocess_nsd --out $HVC --voxel_set hvc --stage layout --backbone clip_b16
  PYTHONPATH=. python scripts/clip_layers_ceiling.py --root $HVC --layout pca512_clsmean_all --gt
  for H in 1 2 5 7; do ROOT=$HVC COMPARE=outputs/none RIDGE=1 sh scripts/clip_layers_holdout.sh $L $H; done
  python scripts/valuniq_layout.py --root $HVC --layout $L --holdout 1 2 --val_images 600
  for H in 1 2; do ROOT=$HVC NO_SYNTH=1 SKIP_POPEB=1 sh scripts/repro_braincorl.sh stage3 ${L}_valuniq_h$H $H; done
}
case $1 in
  gpu)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    step P2 p2; step P6a p6a; step P5 p5; step P4 p4; step P1 p1; step P2V p2v; step P1V p1v; step P7 p7; step P6b p6b
    echo "[CHAIN done] $(date +%F\ %T)" ;;
  hvc_prep) step hvc_prep hvc_prep ;;
  *) echo "usage: $0 {gpu|hvc_prep}"; exit 1 ;;
esac
