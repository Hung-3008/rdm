#!/bin/bash
# Poll both training queues; for every finished run: extract NSD features on PC344 (the 425 px NSD cache lives there), copy
# them here and run scripts/foveacl_ceiling.py. Exits when every run of both queue files is evaluated or failed.
#   bash scripts/foveacl_eval_watch.sh
R=/media/hung/HDD/workplaces/codes/foveacl; RPY=/home/hung/miniconda3/envs/rdm/bin/python; OUT=outputs/foveacl
LOCAL=$(grep -v '^#' scripts/foveacl_queue_local.txt | awk 'NF{print $1}' | tr '\n' ' '); REMOTE=$(grep -v '^#' scripts/foveacl_queue_pc344.txt | awk 'NF{print $1}' | tr '\n' ' ')
evaluate() {  # $1 run, $2 extra extract flag, $3 feats file
  ssh PC344 "cd $R && PYTHONPATH=. $RPY -m src.foveacl.extract_nsd --cache data/images/foveacl/nsd425 --run $OUT/$1 $2" || return 1
  mkdir -p $OUT/$1 && rsync -a PC344:$R/$OUT/$1/$3 PC344:$R/$OUT/$1/nsd_feats.json PC344:$R/$OUT/$1/args.json $OUT/$1/ || return 1
  PYTHONPATH=. python scripts/foveacl_ceiling.py --runs $OUT/$1 --feats $3 || return 1
  rm -f $OUT/$1/$3; ssh PC344 "rm -f $R/$OUT/$1/$3"
}
while true; do
  pending=0
  for r in $LOCAL $REMOTE; do
    [[ -f $OUT/$r/ceiling_done || -f $OUT/$r/failed ]] && continue
    if [[ " $LOCAL " == *" $r "* ]]; then
      [[ -f $OUT/$r/done ]] || { pending=1; continue; }
      ssh PC344 "mkdir -p $R/$OUT/$r" && rsync -a $OUT/$r/last.pt $OUT/$r/args.json PC344:$R/$OUT/$r/
    else
      ssh PC344 "test -f $R/$OUT/$r/failed" && { mkdir -p $OUT/$r; touch $OUT/$r/failed; continue; }
      ssh PC344 "test -f $R/$OUT/$r/done" || { pending=1; continue; }
      mkdir -p $OUT/$r && rsync -a PC344:$R/$OUT/$r/log.jsonl $OUT/$r/
    fi
    echo "$(date +%F_%T) evaluate $r"
    evaluate $r "" nsd_feats.npy && touch $OUT/$r/ceiling_done || { echo "$(date +%F_%T) EVAL FAILED $r"; pending=1; }
    if [[ $r == R0_s0 || $r == R2_s0 ]]; then evaluate $r --untrained nsd_feats_untrained.npy; fi
  done
  [[ $pending == 0 ]] && break
  sleep 300
done
echo "$(date +%F_%T) all runs evaluated"
