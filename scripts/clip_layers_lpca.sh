#!/bin/sh
# Per-layer PCA features without stimulus leakage: standardisation + PCA fitted on the unique images of the three training
# subjects of each hold-out (asserted disjoint from the hold-out's unique and the common images), then PopEB-v2 few-shot.
#   sh scripts/clip_layers_lpca.sh "lpca20_clsmean_all lpcaJ40_clsmean_all" "1 2 5 7" [SEED]
set -e
cd "$(dirname "$0")/.."
VARS=$1; HS=${2:-"1 2 5 7"}; export SEED=$3
for H in $HS; do
  case $H in 1) TR="2 5 7" ;; 2) TR="1 5 7" ;; 5) TR="1 2 7" ;; 7) TR="1 2 5" ;; esac
  FIT=$(echo $TR | tr -d ' ')
  for V in $VARS; do
    PYTHONPATH=. python scripts/clip_layers_ceiling.py --layout $V --fit_subjects $TR --holdout $H
    RIDGE=1 sh scripts/clip_layers_holdout.sh clip_${V}_fit$FIT $H
  done
done
