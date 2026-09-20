from .factory import MODELS, build_model
from . import braincorl  # noqa: F401  (registers "braincorl")

__all__ = ["MODELS", "build_model"]
