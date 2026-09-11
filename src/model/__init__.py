from .factory import MODELS, build_model
from . import braincorl, braincorl_coord  # noqa: F401  (registers "braincorl", "braincorl_coord")

__all__ = ["MODELS", "build_model"]
