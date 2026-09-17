#!/bin/sh
# MedARC trunk as the basis for a held-out subject (docs/results.md E34).
#   sh scripts/medarc_holdout.sh layout        # data/braincorl_nsdgeneral/concat3: CLIP+DINOv2+SigLIP, each block unit-norm, / sqrt(3)
#   sh scripts/medarc_holdout.sh nested        # embed_dim grid, leave-one-training-subject-out on S2/S5/S7 (CLIP), n = 100 x 5
#   sh scripts/medarc_holdout.sh final D       # trunk on S2/S5/S7 with embed_dim D (CLIP and concat3), eval S1 on the E20 support draws
set -e
cd "$(dirname "$0")/.."
CFG=src/config/medarc_holdout.yaml; NG=data/braincorl_nsdgeneral; OUT=outputs/medarc/holdout
EV="python -m src.eval_lowrank --config $CFG --popeb --popeb_meta_dir $NG"

layout() {
  python - <<'PY'
import json, os
import numpy as np
from src.utils import unit_norm
root = "data/braincorl_nsdgeneral"; bbs = ["clip_b16", "dinov2_base", "siglip_so400m"]; out = f"{root}/concat3"
os.makedirs(out, exist_ok=True)
cat = lambda name: np.concatenate([unit_norm(np.load(f"{root}/{bb}/{name}").astype(np.float32)) for bb in bbs], 1) / np.sqrt(len(bbs))
orders = [json.load(open(f"{root}/{bb}/test_time_icl_idx.json")) for bb in bbs]
assert all(o == orders[0] for o in orders), "support orderings differ between backbones"
np.save(f"{out}/common_img_feats_normed.npy", cat("common_img_feats_normed.npy"))
links = ["test_time_icl_idx.json"]
for s in [1, 2, 5, 7]:
    np.save(f"{out}/s{s}_unique_img_feats_normed.npy", cat(f"s{s}_unique_img_feats_normed.npy"))
    links += [f"s{s}_unique_nrn_feats_msk.npy", f"s{s}_common_nrn_feats_msk.npy", f"s{s}_coords.npy", f"s{s}_gt.npy"]  # gt: CLIP ridge, unused by eval_lowrank
for f in links:
    if not os.path.lexists(f"{out}/{f}"):
        os.symlink(f"../clip_b16/{f}", f"{out}/{f}")
print("wrote", out)
PY
}

nested() {
  for H in 2 5 7; do
    case $H in 2) TR="5,7" ;; 5) TR="2,7" ;; 7) TR="2,5" ;; esac
    for D in 16 32 64 192; do
      R=$OUT/nested/h${H}_d$D
      [ -f $R/model.pt ] || python -m src.train_medarc --config $CFG out_dir=$R "data.subjects=[$TR]" model.embed_dim=$D > $R.train.log 2>&1
      [ -f $R/eval/results.csv ] || $EV --ckpt_dir $R --subject $H --num_ic 100 --repeats 5 --voxel_meta $NG/s${H}_voxel_meta.npz \
          --out_dir $R/eval "data.source.train_subjects=[$TR]" > $R.eval.log 2>&1
    done
  done
  python - <<'PY'
import glob, re
import pandas as pd
rows = []
for p in glob.glob("outputs/medarc/holdout/nested/h*_d*/eval/results.csv"):
    h, d = map(int, re.search(r"h(\d+)_d(\d+)", p).groups())
    df = pd.read_csv(p).groupby("method")[["expl_var", "pearson"]].mean().reset_index()
    rows += [dict(h=h, d=d, **r) for r in df.to_dict("records")]
df = pd.DataFrame(rows)
print(df.pivot_table(index=["method", "d"], columns="h", values="expl_var").assign(mean=lambda x: x.mean(1)).round(4))
print(df.pivot_table(index=["method", "d"], columns="h", values="pearson").assign(mean=lambda x: x.mean(1)).round(4))
PY
}

final() {
  D=$1; mkdir -p $OUT
  for FEAT in clip_b16 concat3; do
    case $FEAT in clip_b16) BB="[clip_b16]" ;; concat3) BB="[clip_b16,dinov2_base,siglip_so400m]" ;; esac
    R=$OUT/s1_${FEAT}_d$D
    [ -f $R/model.pt ] || python -m src.train_medarc --config $CFG out_dir=$R "data.backbones=$BB" model.embed_dim=$D > $R.train.log 2>&1
    for SET in "100 5 ic100" "30_300_1000 3 scaling"; do
      set -- $SET; NIC=$(echo $1 | tr _ ' ')
      [ -f $R/eval_$3/results.csv ] || $EV --ckpt_dir $R --subject 1 --num_ic $NIC --repeats $2 --ridge --voxel_meta $NG/s1_voxel_meta.npz \
          --compare_dir outputs/repro/clip_b16/eval_s1_$3 --out_dir $R/eval_$3 data.source.data_dir=$NG/$FEAT > $R.eval_$3.log 2>&1
    done
  done
}

case $1 in
  layout) layout ;;
  nested) nested ;;
  final)  final $2 ;;
  *) echo "usage: $0 {layout|nested|final D}"; exit 1 ;;
esac
