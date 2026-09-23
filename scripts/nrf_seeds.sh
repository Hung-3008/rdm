#!/bin/sh
# NRF at two extra training seeds, for the seed-variance column of docs/re_produce/nrf/nrf.md.
#   PY=~/miniconda3/envs/rdm/bin/python nohup sh scripts/nrf_seeds.sh > outputs/nrf_seeds.log 2>&1 &
# Rerunning resumes: finished base models and finetune runs are skipped.
#
# Seed-independent and therefore shared with outputs/nrf (seed 0), so that a difference is training stochasticity only:
# the CLIP features (data/nrf/feats), the 8,400/600 base split (default_rng(0)), the support sets (default_rng(100)),
# the 1,000-image test set, and the Fourier B matrix (default_rng(42) inside references/NRF). --seed drives the weight
# init, the batch order and the per-image 2,000-voxel subsampling.
#
# MICRO=0 (full batch 32) is the default here; outputs/nrf used MICRO=8 from fold S1 n=1000 onward, which draws the
# voxel subsets from a different RNG stream. That is a second nuisance draw of the same kind as the seed, not a
# different estimator, but it does mean seed 0 is not micro-batch-matched to seeds 1 and 2 on 117 of its 144 ft runs.
set -e
PY=${PY:-python}
for seed in ${SEEDS:-1 2}; do
    OUT=outputs/nrf_seed$seed
    mkdir -p $OUT
    $PY scripts/nrf_repro.py base  --subjects 1 2 5 7 --seed $seed --out $OUT --resume
    $PY scripts/nrf_repro.py ft    --holdout  1 2 5 7 --seed $seed --out $OUT --micro ${MICRO:-0}
    $PY scripts/nrf_repro.py table --holdout  1 2 5 7 --seed $seed --out $OUT --cpu
done
