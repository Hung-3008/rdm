#!/bin/sh
# Progress of every NRF seed directory.   PY=~/miniconda3/envs/rdm/bin/python sh scripts/nrf_seed_status.sh
PY=${PY:-python}
for d in outputs/nrf outputs/nrf_seed*; do
    [ -d "$d" ] || continue
    echo "===== $d"
    $PY scripts/nrf_status.py --out "$d"
done
