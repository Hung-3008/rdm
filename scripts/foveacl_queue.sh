#!/bin/bash
# Run FoveaCL trainings one after another on this machine's GPU.
#   bash scripts/foveacl_queue.sh <queue file> [python]
# Queue file: one run per line, "<name> <train.py args>"; lines starting with # are skipped. A run whose log already holds
# the last epoch is skipped; an interrupted run resumes from last.pt. The file is re-read after every run, so lines can be appended.
Q=$1; PY=${2:-python}; DATA=${DATA:-data/images/foveacl/in100_256}; OUT=${OUT:-outputs/foveacl}
mkdir -p $OUT logs
while true; do
  next=""
  while read -r name args; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    [[ -f $OUT/$name/done ]] && continue
    next="$name"; nargs="$args"; break
  done < "$Q"
  [[ -z "$next" ]] && break
  echo "$(date +%F_%T) start $next: $nargs"
  PYTHONPATH=. $PY -m src.foveacl.train --data $DATA --out $OUT/$next $nargs >> logs/foveacl_$next.log 2>&1 && touch $OUT/$next/done || { echo "$(date +%F_%T) FAILED $next"; touch $OUT/$next/done; mv $OUT/$next/done $OUT/$next/failed; sed -i "s/^$next /#FAILED $next /" "$Q"; }
done
echo "$(date +%F_%T) queue empty"
