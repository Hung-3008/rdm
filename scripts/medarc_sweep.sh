#!/bin/sh
# Validation sweep for the MedARC encoder on S1 (src/config/medarc_s1.yaml); selection by best_val_pearson
# (500 held-out unique images). Ranges follow references/algonauts2025/generate_configs.py.
set -e
cd "$(dirname "$0")/.."
OUT=outputs/medarc/sweep
run() { name=$1; shift; [ -f $OUT/$name/summary.json ] || python -m src.train_medarc --config src/config/medarc_s1.yaml out_dir=$OUT/$name eval.ridge=false "$@" > $OUT/$name.log 2>&1; }
mkdir -p $OUT
for lr in 1.0e-4 3.0e-4 1.0e-3; do for wd in 0.01 0.1 0.3; do run lr${lr}_wd${wd} train.lr=$lr train.weight_decay=$wd; done; done
for d in 128 256 512 1024; do run embed$d model.embed_dim=$d; done
run pool_linear model.global_pool=linear
run pool_attn model.global_pool=attn
run nostd data.standardize=false
run clip_only "data.backbones=[clip_b16]"
run siglip_only "data.backbones=[siglip_so400m]"
# round 2: unit-norm features (nostd had the best val_r in round 1)
run nostd_linear data.standardize=false model.global_pool=linear
run nostd_embed128 data.standardize=false model.embed_dim=128
run nostd_embed64 data.standardize=false model.embed_dim=64
run nostd_lr1e-3 data.standardize=false train.lr=1.0e-3
run nostd_linear_lr1e-3 data.standardize=false model.global_pool=linear train.lr=1.0e-3
run nostd_siglip data.standardize=false "data.backbones=[siglip_so400m]"
python - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("outputs/medarc/sweep/*/summary.json")):
    s = json.load(open(p)); t = s["test"]; t = t if "r_all" in t else t["1"]
    rows.append((os.path.basename(os.path.dirname(p)), s["best_val_pearson"], s["best_epoch"], s["train_sec"], t["r_all"], t["ev_all"], t["ev_ncsnr>0.2"]))
print(f"{'run':16s} val_r  epoch  sec   test_r_all ev_all ev_ncsnr")
for r in sorted(rows, key=lambda r: -r[1]):
    print(f"{r[0]:16s} {r[1]:.4f} {r[2]:3d} {r[3]:6.0f}  {r[4]:.4f} {r[5]:.4f} {r[6]:.4f}")
PY
