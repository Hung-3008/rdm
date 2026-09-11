"""Load model weights from (a) checkpoints released with the original BraInCoRL repo or
(b) Lightning checkpoints written by src.train, into src.model.* modules.

Original submodule names differ only by renaming; verified numerically equivalent
(max |pred diff| 1e-6 on the released CLIP weights, see src/README.md).
"""

import torch
import torch.nn as nn

_RENAMES = (
    ("cls_token", "reg_tokens"),
    ("transformer.", "blocks."),
    (".layer_norm_1.", ".norm1."),
    (".layer_norm_2.", ".norm2."),
)


def remap_original_key(k: str) -> str:
    for old, new in _RENAMES:
        k = k.replace(old, new)
    return k


def load_model_weights(model: nn.Module, path: str, strict: bool = True):
    """Accepts a raw state_dict, {'state_dict': ...} (original repo), or a Lightning
    checkpoint whose keys are prefixed 'model.' (src.engine.IclLightningModule)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    sd = {remap_original_key(k): v for k, v in sd.items()}
    return model.load_state_dict(sd, strict=strict)


# kept for callers that still use the old name
load_original_checkpoint = load_model_weights
