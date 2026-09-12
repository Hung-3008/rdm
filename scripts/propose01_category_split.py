"""E14 (docs/results.md): in-support vs out-of-support category on hold-out S1.
Images are scored against the BraInCoRL A.3 prompts with the CLIP text encoder (mean of the prompt
embeddings per category minus the mean over the four categories, cosine with the CLIP image embedding). For category c in {faces, bodies, places,
words}: pool label = top-400 of S1's 9,000 unique images (Fig. 7 tier size); query label Q_c = top-100 of
the 1,000 common images. Support "control" = the 100 x 5 draws of E00/E07; support "excluded" = the same
slices with images labelled c removed and refilled from the following positions of the ordering (n = 100).
EV is computed on Q_c and on the other 900 queries, for voxels of ROI c (floc t > 2, ncsnr > 0.2), for
PopEB-v2 and BraInCoRL (checkpoint s257). The "reference-memory gap" = EV(control) - EV(excluded) on Q_c.
    python scripts/propose01_category_split.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.config import load_config  # noqa: E402
from src.dataset import SOURCES  # noqa: E402
from src.eval import predict_icl, roi_masks  # noqa: E402
from src.model import build_model  # noqa: E402
from src.model.checkpoint import load_original_checkpoint  # noqa: E402
from src.model.lowrank_basis import FunctionalBasis, LowRankConfig, feature_stats, full_coefficients, predict_lowrank  # noqa: E402
from src.utils import explained_variance, unit_norm  # noqa: E402

PROMPTS = {
    "faces": ["A photo of a person's face", "A portrait photo of a face", "A face facing the camera", "A photo of a face",
              "A photo of a human face", "A photo of faces", "People looking at the camera", "A portrait of a person", "A portrait photo"],
    "bodies": ["A photo of a torso", "A photo of limbs", "A photo of bodies", "A photo of a person", "A photo of people",
               "A photo of a body", "A person's arms", "A person's legs", "A photo of hands"],
    "places": ["A photo of a bedroom", "A photo of an office", "A photo of a hallway", "A photo of a doorway", "A photo of interior design",
               "A photo of a building", "A photo of a house", "A photo of nature", "A photo of a landscape"],
    "words": ["A photo of words", "A photo of glyphs", "A photo of a glyph", "A photo of text", "A photo of numbers", "A photo of a letter",
              "A photo of letters", "A photo of writing", "A photo of text on an object"],
}
POOL_TOP, QUERY_TOP = 400, 100
dev = torch.device("cuda"); D = Path("data/braincorl/clip_b16"); OUT = Path("outputs/propose01/category_split"); OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- category scores
from transformers import CLIPModel, CLIPTokenizer  # noqa: E402
root = Path("checkpoints/backbones/openai--clip-vit-base-patch16")
tok = CLIPTokenizer.from_pretrained(root); clip = CLIPModel.from_pretrained(root).to(dev).eval()
with torch.no_grad():
    def txt(p):
        o = clip.get_text_features(**tok(p, padding=True, return_tensors="pt").to(dev))
        o = o if torch.is_tensor(o) else o[0]             # newer transformers return a ModelOutput
        return o.reshape(-1, o.shape[-1])                 # (n_prompts, 512)
    text = {c: unit_norm(txt(p)).mean(0) for c, p in PROMPTS.items()}
    # contrastive category direction: remove the component shared by the four categories ("A photo of ...")
    # so that the top-400 sets do not overlap (raw prompt means: 185-282 of 400 shared between pairs)
    shared = torch.stack(list(text.values())).mean(0)
    text = {c: unit_norm(v - shared).cpu().numpy() for c, v in text.items()}
del clip; torch.cuda.empty_cache()
bundle = SOURCES.build(load_config("src/config/popeb_clip.yaml", [])["data"]["source"]).load_eval(1)
pool_lab = {c: set(np.argsort(-(bundle.ic_img @ text[c]))[:POOL_TOP].tolist()) for c in PROMPTS}
q_lab = {c: np.zeros(len(bundle.query_img), bool) for c in PROMPTS}
for c in PROMPTS:
    q_lab[c][np.argsort(-(bundle.query_img @ text[c]))[:QUERY_TOP]] = True
print("pool overlap between categories (top-400 sets):", {f"{a}&{b}": len(pool_lab[a] & pool_lab[b]) for a in PROMPTS for b in PROMPTS if a < b})

# ---------------------------------------------------------------- models
cfg2 = load_config("src/config/popeb_clip.yaml", []); f = LowRankConfig(**cfg2["fast"]); TR = cfg2["data"]["source"]["train_subjects"]
basis = FunctionalBasis.from_file("outputs/propose01/sweep/K32_a1/basis.pt", dev)
pop, xyz, imgs = [], [], []
for s in TR:
    img = unit_norm(np.load(D / f"s{s}_unique_img_feats_normed.npy").astype(np.float32))
    Cs = full_coefficients(basis, img, np.load(D / f"s{s}_unique_nrn_feats_msk.npy"), dev); torch.cuda.empty_cache()
    g = np.load(f"data/braincorl/s{s}_voxel_meta.npz")["ncsnr"] > 0.2
    pop.append(Cs[:, g]); xyz.append(np.load(D / f"s{s}_coords.npy").astype(np.float32)[g]); imgs.append(img)
Cpop = torch.from_numpy(np.concatenate(pop, 1)).to(dev); f.tau2 = float(f.tau2_ratio * Cpop.var(1).mean().item())
spatial = (torch.from_numpy(np.load(D / "s1_coords.npy").astype(np.float32)).to(dev), torch.from_numpy(np.concatenate(xyz, 0)).to(dev), f.spatial_sigma)
F_stats = feature_stats(basis, imgs, dev)
model = build_model(load_config("src/config/braincorl_clip_3090.yaml", [])["module"]["model"])
load_original_checkpoint(model, "references/BraInCoRL/checkpoints/CLIP_trained_on_s257.pth"); model = model.to(dev).eval()
rois = roi_masks("data/braincorl/s1_voxel_meta.npz")
true = torch.from_numpy(bundle.query_beta.T).to(dev); q_img = torch.from_numpy(bundle.query_img).to(dev)


def predict(idx):
    ic_img = torch.from_numpy(bundle.ic_img[idx]).to(dev); ic_beta = torch.from_numpy(bundle.ic_beta[idx]).to(dev)
    return {"popeb_v2": predict_lowrank(ic_img, ic_beta, q_img, basis, f, Cpop=Cpop, spatial=spatial, F_stats=F_stats)[0],
            "braincorl": predict_icl(model, bundle.ic_img[idx], bundle.ic_beta[idx], bundle.query_img, bundle.query_beta, bundle.gt, dev, 128)[0]}


def ev_on(pred, qmask):
    return explained_variance(pred[:, torch.from_numpy(qmask).to(dev)], true[:, torch.from_numpy(qmask).to(dev)]).cpu().numpy()


order = bundle.icl_order
res = {}          # (c, support_type, method, query_set) -> list over draws of per-voxel EV
for rep in range(5):
    ctrl_idx = order[rep * 100:(rep + 1) * 100]
    ctrl = predict(ctrl_idx)
    for c in PROMPTS:
        # excluded support: walk the ordering from the same start, skip images labelled c, take 100
        excl_idx = [i for i in order[rep * 100:] if i not in pool_lab[c]][:100]
        n_removed = sum(1 for i in ctrl_idx if i in pool_lab[c])
        excl = predict(np.array(excl_idx))
        for stype, P in [("control", ctrl), ("excluded", excl)]:
            for m, p in P.items():
                res.setdefault((c, stype, m, "Qc"), []).append(ev_on(p, q_lab[c]))
                res.setdefault((c, stype, m, "Qnot"), []).append(ev_on(p, ~q_lab[c]))
        print(f"rep {rep} {c}: {n_removed} of 100 control images were category-{c}; Q_c = {q_lab[c].sum()} queries", flush=True)

rng = np.random.default_rng(0)
def boot(d):
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]); return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
rows = []
print("\n## E14: EV on Q_c (category queries) for ROI-c voxels, mean over 5 draws")
print(f"{'category':8s} {'ROI vox':>7s} | PopEB-v2 ctrl / excl (gap) | BraInCoRL ctrl / excl (gap) | v2-BraInCoRL on excluded [CI] | Q_not: v2 ctrl/excl, BC ctrl/excl")
for c in PROMPTS:
    m = rois[c]
    M = {k: np.mean(v, 0) for k, v in res.items() if k[0] == c}
    v2c, v2e = M[(c, "control", "popeb_v2", "Qc")][m].mean(), M[(c, "excluded", "popeb_v2", "Qc")][m].mean()
    bcc, bce = M[(c, "control", "braincorl", "Qc")][m].mean(), M[(c, "excluded", "braincorl", "Qc")][m].mean()
    d = boot((M[(c, "excluded", "popeb_v2", "Qc")] - M[(c, "excluded", "braincorl", "Qc")])[m])
    qn = [M[(c, s, mm, "Qnot")][m].mean() for mm in ["popeb_v2", "braincorl"] for s in ["control", "excluded"]]
    rows.append(dict(category=c, n_vox=int(m.sum()), v2_ctrl=float(v2c), v2_excl=float(v2e), bc_ctrl=float(bcc), bc_excl=float(bce), v2_minus_bc_excl=list(map(float, d)),
                     v2_ctrl_qnot=float(qn[0]), v2_excl_qnot=float(qn[1]), bc_ctrl_qnot=float(qn[2]), bc_excl_qnot=float(qn[3])))
    print(f"{c:8s} {int(m.sum()):7d} | {v2c:.3f} / {v2e:.3f} ({v2c - v2e:+.3f}) | {bcc:.3f} / {bce:.3f} ({bcc - bce:+.3f}) | {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] | {qn[0]:.3f}/{qn[1]:.3f}, {qn[2]:.3f}/{qn[3]:.3f}")
json.dump(rows, open(OUT / "summary.json", "w"), indent=1)
