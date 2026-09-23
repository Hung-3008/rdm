#!/bin/sh
# One-shot progress probe of the seed queue (scripts/nrf_seeds.sh), for a polling monitor.
# Prints "<state key>|<human line>". The key changes only on a milestone (a base model or a block of 12 finetune runs
# finishing, the table stage, the process stopping, a new error), so a monitor comparing the key stays quiet between
# milestones instead of firing once per epoch.
cd "$(dirname "$0")/.." || exit 1
key=""; human=""
for s in 1 2; do
    b=$(ls outputs/nrf_seed$s/base_s*/done.json 2>/dev/null | wc -l)
    f=$(ls outputs/nrf_seed$s/ft_h*/n*_rep*_src*/done.json 2>/dev/null | wc -l)
    t=$([ -f outputs/nrf_seed$s/results.csv ] && echo T || echo -)
    key="$key$b.$((f / 12)).$t/"
    human="$human seed$s[base $b/4 ft $f/144 table $t]"
done
last=$(grep '^{' outputs/nrf_seeds.log 2>/dev/null | tail -1 | sed 's/.*"tag": "//;s/", "epoch": / ep/;s/, "loss".*"val_corr": / val/;s/,.*//')
err=$(grep -cE 'Traceback|CUDA out of memory|Killed|RuntimeError' outputs/nrf_seeds.log 2>/dev/null)
run=$(pgrep -f nrf_repro.py >/dev/null && echo run || echo STOPPED)
echo "$run.$key$err|$run |$human | now: ${last:-starting} | err $err"
