# src/ — in-context voxel encoders

Re-implementation of BraInCoRL (Yu et al., NeurIPS 2025, arXiv:2505.15813) organised
around registries so that models, data sources and losses can be swapped from YAML.
Original code: `references/BraInCoRL/` (Apache-2.0); the files below that say
"Adapted from ..." carry that license.

## Layout

```
src/
  registry.py            Registry: name -> constructor, .build(cfg)
  model/
    factory.py           MODELS registry, build_model(cfg)
    layers.py            SwiGLU FFN, log-scaled attention block, residual MLP block
    braincorl.py         @MODELS.register("braincorl")  hypernetwork -> linear readout
  dataset/
    factory.py           SOURCES registry, IclDataModule, build_datamodule(cfg)
    icl_dataset.py       SubjectArrays, IclVoxelDataset (one item = one voxel)
    sources.py           @SOURCES.register("npy_multisubj") BraInCoRL NSD layout
                         @SOURCES.register("synthetic")     linear toy voxels
  engine/
    losses.py            LOSSES registry: "pmse", "weight_cos_l2"
    pl_module.py         IclLightningModule: context-length sampling, metrics, optimiser
  utils/
    metrics.py           explained_variance, point_mse, normed_cos_similarity
    predict.py           linear_readout, unit_norm
    ridge.py             ridge_cv_brainwise (within-subject baseline)
  config/
    load.py              load_config(path, ["key.sub=value", ...])
    braincorl_clip_3090.yaml   paper architecture, single 24 GB GPU
    smoke_synthetic.yaml       3-epoch smoke test on synthetic data
  train.py               python -m src.train --config <yaml> [k=v ...]
  eval.py                python -m src.eval  --config <yaml> --ckpt <ckpt> --subject N --num_ic 30 100 [--ridge]
```

Mapping to the original scripts: `model.py` -> `model/layers.py` + `model/braincorl.py` +
`engine/pl_module.py` (+ `model/checkpoint.py` to load the released checkpoints);
`dataset.py` -> `dataset/`; `utils.py` -> `utils/`;
`train_pl_mul_subj.py` -> `train.py`; `test_det.py` -> `eval.py`.

## BraInCoRL training protocol (`scripts/repro_braincorl.sh`)

Three stages; stage 1-2 are synthetic and were skipped in the runs that produced the numbers
in `docs/research-log-2026-09.md` section 2 (`NO_SYNTH=1`, stage 3 from random init), because
the synthetic stages did not learn in-context regression within budget.

| stage | config | data | context | stop |
| --- | --- | --- | --- | --- |
| 1 | `config/repro/stage1_synth.yaml` | synthetic, 4 x 16k voxels | fixed 500 | <= 15 epochs, patience 3 |
| 2 | `config/repro/stage2_synth_ctx.yaml` | same synthetic data | U(0, 500) | <= 10 epochs, patience 3 |
| 3 | `config/repro/stage3_real.yaml` | NSD, three training subjects, one held out | U(0, 500) | patience 10 |

Deviations from the paper: single GPU, bf16; context lower bound 0 instead of 30 in stages 2-3.
Each stage initialises from the best `val_expl_var` checkpoint of the previous one
(`module.init_weights`).

`ModelCheckpoint(save_last=True, save_top_k=2)` writes `last.ckpt` only when it writes a top-k
checkpoint, so `--resume` can lose epochs when `val_expl_var` plateaus. Add a second
`ModelCheckpoint` without `monitor` if resume has to be exact.

`scripts/valuniq_layout.py` builds a layout that splits each training subject's unique images
8,400 / 600 so that a comparison method selects its model on the same held-out images
(`docs/research-log-2026-09.md` section 8).

## Adding a component

```python
from src.model.factory import MODELS

@MODELS.register("my_model")
class MyModel(nn.Module):
    def forward(self, ic_img, ic_beta, query_img): ...   # -> (pred (B,Q), weights (B,E+1))
```

then set `module.model.name: my_model` in YAML. Extra batch fields (e.g. voxel
coordinates) need a new `SubjectArrays` field, a new source, and a matching
`IclVoxelDataset.__getitem__`; the LightningModule unpacks the batch in `_step`.

## Data format expected by `npy_multisubj`

Per subject `i` under `data_dir` (float32; image rows unit-normed):

| file | shape |
| --- | --- |
| `s{i}_unique_img_feats_normed[_val].npy` | (num_img, E) |
| `s{i}_unique_nrn_feats_msk[_val].npy` | (num_img, num_vxl) |
| `s{i}_gt[_val].npy` | (num_vxl, E+1) ridge weights fit on all images |
| `common_img_feats_normed.npy` | (num_common, E) shared test images |
| `s{i}_common_nrn_feats_msk.npy` | (num_common, num_vxl) |
| `test_time_icl_idx.json` | `{"large_shuffled": [...]}` support-set ordering |

The original repository does not include the NSD preprocessing that produces these
files. `python -m src.tools.preprocess_nsd` builds them from raw NSD
(`data/NSD/data`, func1pt8mm, `betas_fithrf_GLMdenoise_RR`) into `data/braincorl/`:

| stage | output | choice made |
| --- | --- | --- |
| `mask` | `s{i}_voxel_meta.npz` (flat index, ijk, scanner xyz, streams/floc labels, floc t-values, ncsnr) | voxel set = streams labels 2–7 (non-early) ∪ floc ROIs; 22612 / 21714 / 20152 / 18231 voxels for S1/2/5/7 (the original code mentions 22961 for its test subject) |
| `betas` | `s{i}_betas_{unique,common}.npy`, `s{i}_nsdid_*.npy` | z-score per voxel within session, average the 3 repeats; hdf5 or nii.gz sessions (verified identical). The averaging is at `tools/preprocess_nsd.py:126-138`; a single-trial layout needs that step changed. |
| `feats` | `feats_clip_b16_73k.npy` (73000, 512) | CLIP ViT-B/16 `image_embeds`, unit-normed at use |
| `gt` | `s{i}_gt.npy` | `ridge_cv_brainwise` on the 9000 unique images |
| `layout` | `data/braincorl/clip_b16/` | `_val` files = the subject's 1000 shared images; `test_time_icl_idx.json` = seeded permutation |

Validation against BraInCoRL Table 1 (S1, EV on the 1000 shared images, floc t > 2,
ncsnr > 0.2; ridge rows = mean of 5 support-set draws):

| | faces | places | bodies | words |
| --- | --- | --- | --- | --- |
| ridge `gt` (ours) / Fully Trained (paper) | 0.206 / 0.19 | 0.220 / 0.20 | 0.294 / 0.28 | 0.121 / 0.11 |
| Ridge-100 (ours / paper) | 0.101 / 0.10 | 0.094 / 0.08 | 0.164 / 0.16 | 0.032 / 0.02 |
| Ridge-300 (ours / paper) | 0.140 / 0.13 | 0.142 / 0.13 | 0.221 / 0.22 | 0.063 / 0.06 |

The `gt` rows for S2/S5/S7 are 0.01–0.05 above the paper's Fully Trained column
(largest gap S7 faces 0.191 vs 0.14, n = 370). The food ROI is not in NSD and is not
evaluated. Floc t-maps and S1 ncsnr were fetched from `s3://natural-scenes-dataset`
into the shared NSD tree.

Released checkpoints (`references/BraInCoRL/checkpoints/CLIP_trained_on_s{...}.pth`,
leave-one-subject-out) run on this data with 100 in-context images, mean ± std over
5 support-set draws, versus BraInCoRL-100 in Tables S.1–S.4:

| subject (ckpt) | faces | places | bodies | words |
| --- | --- | --- | --- | --- |
| S1 (s257) | 0.168 ± 0.003 / 0.16 | 0.182 ± 0.002 / 0.16 | 0.253 ± 0.004 / 0.25 | 0.076 ± 0.007 / 0.07 |
| S2 (s157) | 0.157 ± 0.007 / 0.13 | 0.242 ± 0.010 / 0.23 | 0.235 ± 0.003 / 0.21 | 0.099 ± 0.005 / 0.08 |
| S5 (s127) | 0.227 ± 0.005 / 0.20 | 0.292 ± 0.007 / 0.29 | 0.242 ± 0.002 / 0.23 | 0.142 ± 0.007 / 0.12 |
| S7 (s125) | 0.154 ± 0.010 / 0.11 | 0.164 ± 0.004 / 0.15 | 0.196 ± 0.009 / 0.18 | 0.078 ± 0.008 / 0.08 |

Command: `python -m src.eval --config src/config/braincorl_clip_3090.yaml --ckpt <pth>
--original_ckpt --subject N --num_ic 100 --repeats 5 --voxel_meta data/braincorl/sN_voxel_meta.npz`.
Outputs in `outputs/eval_original_clip/sN/`.

## Smoke test (synthetic source, RTX 3090, bf16, context 500, full CLIP architecture)

| batch | peak GPU memory | throughput |
| --- | --- | --- |
| 16 | 6.4 GiB | 9.8 it/s |
| 32 | 9.0 GiB | 5.9 it/s |
| 64 | 16.2 GiB | 3.2 it/s |

20 epochs on 2 x 1024 synthetic voxels (2560 steps, batch 16, lr 1e-4): train EV
-0.30 -> -0.035, val EV -0.17 -> -0.030; the model had not yet learned to use the
context (held-out cos_sim to true weights ~0, EV independent of num_ic). The ridge
baseline on the same support sets reaches EV 0.006 / 0.030 / 0.117 at 30 / 100 / 300
shots (ceiling 0.5). The smoke test verifies the pipeline, not convergence.

## Image backbones (`checkpoints/backbones/`, via `python -m src.tools.download_backbones`)

| key | HF repo | output dim | basis |
| --- | --- | --- | --- |
| `clip_b16` | `openai/clip-vit-base-patch16` | 512 (`image_embeds`); NRF also uses hidden layers 3 and 6 | NRF names it in `training_config.yaml`; BraInCoRL only states E=512 |
| `dinov2_base` | `facebook/dinov2-base` | 768 | inferred from BraInCoRL E=768 |
| `siglip_so400m` | `google/siglip-so400m-patch14-384` | 1152 | inferred from BraInCoRL E=1152 |

BraInCoRL's released code does not name its CLIP/DINO/SigLIP checkpoints; the two
inferred rows are the standard models with matching embedding sizes, not confirmed by
the authors. Released BraInCoRL weights: `references/BraInCoRL/checkpoints/weights.zip`
(SharePoint, 9.38 GiB). NRF publishes no pretrained weights.

## Differences from the original training script

- single GPU, `bf16-mixed`, gradient accumulation instead of FSDP over 8 GPUs;
- context length sampled with torch RNG (deterministic under `seed_everything`);
- CSV logger instead of wandb; early stopping on `val_expl_var`;
- the synthetic-voxel pretraining stage described in the paper is available as the
  `synthetic` source (the released code only contains the real-data stage).
