#!/bin/sh
# Nested selection of the basis dimension K for a multi-layer CLIP feature variant: leave-one-training-subject-out
# over S2/S5/S7 (basis on the other two, PopEB population from the same two), n = 100 x 5 on the protocol draws.
#   sh scripts/clip_layers_nested.sh VARIANT "32 64 128"
set -e
cd "$(dirname "$0")/.."
V=$1; KS=${2:-"32 64 128"}; NG=data/braincorl_nsdgeneral; DATA=$NG/clip_$V; OUT=outputs/clip_layers/nested/$V; CFG=src/config/popeb_clip.yaml
mkdir -p $OUT
for H in 2 5 7; do
  case $H in 2) TR="[5,7]" ;; 5) TR="[2,7]" ;; 7) TR="[2,5]" ;; esac
  for K in $KS; do
    R=$OUT/h${H}_K$K
    [ -f $R/basis.pt ] || python -m src.train_lowrank --config $CFG --out_dir $R data.source.data_dir=$DATA "data.source.train_subjects=$TR" basis.basis_dim=$K > $R.train.log 2>&1
    [ -f $R/eval/results.csv ] || python -m src.eval_lowrank --config $CFG --ckpt_dir $R --subject $H --num_ic 100 --repeats 5 --popeb \
        --voxel_meta $NG/s${H}_voxel_meta.npz --popeb_meta_dir $NG --out_dir $R/eval data.source.data_dir=$DATA "data.source.train_subjects=$TR" > $R.eval.log 2>&1
  done
done
python - <<PY
import glob, re
import pandas as pd
rows = []
for p in glob.glob("$OUT/h*_K*/eval/results.csv"):
    h, k = map(int, re.search(r"h(\d+)_K(\d+)", p).groups())
    df = pd.read_csv(p).groupby("method")[["expl_var", "ev_ncsnr>0.2"]].mean().reset_index()
    rows += [dict(h=h, K=k, **r) for r in df.to_dict("records")]
df = pd.DataFrame(rows)
for col in ["expl_var", "ev_ncsnr>0.2"]:
    print(col); print(df.pivot_table(index=["method", "K"], columns="h", values=col).assign(mean=lambda x: x.mean(1)).round(4))
PY
