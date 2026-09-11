from typing import Any, Dict

import torch.nn as nn

from ..registry import Registry

MODELS = Registry("model")


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """cfg: {'name': <registered model>, ...kwargs}"""
    return MODELS.build(cfg)
