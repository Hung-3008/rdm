#!/bin/sh
# Progress of scripts/results_new_chain.sh: step markers, latest BraInCoRL stage-3 checkpoints, GPU.
#   sh scripts/results_new_status.sh
cd "$(dirname "$0")/.."
R=outputs/results_new
echo "== steps"; grep -h "^\[STEP\|^\[CHAIN" $R/chain.log $R/hvc_prep.log 2>/dev/null
echo "== BraInCoRL stage 3 (best checkpoints)"
for d in outputs/repro*/clip_pca512_clsmean_all*/stage3_s*; do
  [ -d $d/checkpoints ] || continue
  echo "$d $( [ -f $d.done ] && echo DONE ) $(ls $d/checkpoints | grep val_expl | sort -t= -k3 -n | tail -1)"
done
echo "== errors"; grep -l -E "Traceback|Error|Killed" $R/*.log outputs/repro*/clip_pca512_clsmean_all*/*.log outputs/clip_layers/holdout*/*/s*/*.log 2>/dev/null
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
