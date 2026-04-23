"""Load YAML configs with caching."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@functools.lru_cache(maxsize=8)
def _load(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def field_criticality() -> dict[str, dict[str, str]]:
    return _load("field_criticality.yaml")


def doc_type_policies() -> dict[str, Any]:
    return _load("doc_type_policies.yaml")


def scoring_weights() -> dict[str, Any]:
    return _load("scoring_weights.yaml")
