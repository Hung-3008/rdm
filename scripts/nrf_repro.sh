#!/bin/sh
# NRF on the 4 hold-out folds (scripts/nrf_repro.py). Rerunning resumes: finished base models and finetune runs are skipped.
#   PY=~/miniconda3/envs/rdm/bin/python nohup sh scripts/nrf_repro.sh > outputs/nrf/queue.log 2>&1 &
set -e
PY=${PY:-python}
mkdir -p outputs/nrf
for s in 1 2 5 7; do
    until [ -f data/nrf/stimuli/dl.done ] || [ -f data/nrf/feats/s${s}_higher.npy ]; do sleep 60; done
    [ -f data/nrf/feats/s${s}.done ] || { $PY scripts/nrf_repro.py feats --subjects $s && touch data/nrf/feats/s${s}.done; }
    $PY scripts/nrf_repro.py base --subjects $s --resume
done
# MICRO=8: micro-batches of 8 images (same gradient) when the GPU is shared; table on the CPU for the same reason
$PY scripts/nrf_repro.py ft --holdout 1 2 5 7 --micro ${MICRO:-0}
$PY scripts/nrf_repro.py table --holdout 1 2 5 7 --cpu
