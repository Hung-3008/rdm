"""Minimal name -> constructor registry used by every factory in src/.

Usage:
    MODELS = Registry("model")

    @MODELS.register("braincorl")
    class BraInCoRL(nn.Module): ...

    model = MODELS.build({"name": "braincorl", "embed_dim": 512})
"""

from typing import Any, Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable] = {}

    def register(self, name: str):
        def deco(fn: Callable):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = fn
            return fn
        return deco

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'; available: {sorted(self._items)}")
        return self._items[name]

    def build(self, cfg: Dict[str, Any], **extra):
        """cfg must contain 'name'; the remaining keys are passed as kwargs."""
        cfg = dict(cfg)
        name = cfg.pop("name")
        return self.get(name)(**cfg, **extra)

    def names(self):
        return sorted(self._items)
