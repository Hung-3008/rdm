"""YAML config loading with optional `key.sub=value` overrides from the CLI."""

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _parse_value(v: str) -> Any:
    return yaml.safe_load(v)


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def load_config(path: str, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    with open(Path(path)) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)
