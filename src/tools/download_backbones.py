"""Download the frozen image backbones used by NRF and BraInCoRL into checkpoints/backbones/.

    python -m src.tools.download_backbones [--only clip_b16 dinov2_base siglip_so400m]

NRF names its backbone explicitly (references/NRF/training_config.yaml:
openai/clip-vit-base-patch16). BraInCoRL only reports embedding sizes
(CLIP E=512, DINO E=768, SigLIP E=1152, paper A.2); the DINO and SigLIP
checkpoints below are the standard models with those output sizes and are an
inference, not a statement from the authors.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

BACKBONES = {
    "clip_b16": ("openai/clip-vit-base-patch16", "NRF (stated); BraInCoRL CLIP E=512 (inferred)"),
    "dinov2_base": ("facebook/dinov2-base", "BraInCoRL DINO E=768 (inferred)"),
    "siglip_so400m": ("google/siglip-so400m-patch14-384", "BraInCoRL SigLIP E=1152 (inferred)"),
}

ROOT = Path("checkpoints/backbones")


def download(repo_id: str) -> Path:
    files = HfApi().list_repo_files(repo_id)
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    patterns = ["*.json", "*.txt", "*.safetensors" if has_safetensors else "*.bin"]
    local = ROOT / repo_id.replace("/", "--")
    snapshot_download(repo_id, local_dir=local, allow_patterns=patterns)
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(BACKBONES))
    args = ap.parse_args()
    for key in args.only:
        repo_id, note = BACKBONES[key]
        local = download(repo_id)
        size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file()) / 2**20
        print(f"{key:14s} {repo_id:40s} {size:8.1f} MiB  -> {local}   [{note}]")


if __name__ == "__main__":
    main()
