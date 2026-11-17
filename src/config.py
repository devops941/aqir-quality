"""Central configuration loader.

Reads ``config.yaml`` from the repository root exactly once and exposes it to
every module. All relative paths in the config are resolved against the repo
root so the project works regardless of the current working directory.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of the ``src`` package.
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(key: str, *, create_parent: bool = False) -> Path:
    """Resolve a ``paths.<key>`` entry from config against the repo root."""
    cfg = load_config()
    rel = cfg["paths"][key]
    path = (ROOT / rel).resolve()
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs() -> None:
    """Create all directories referenced in ``paths`` (files get their parent)."""
    cfg = load_config()
    for key, rel in cfg["paths"].items():
        path = (ROOT / rel).resolve()
        target = path if key.startswith("data_") or key in {"models", "outputs"} else path.parent
        target.mkdir(parents=True, exist_ok=True)


# Eagerly-loaded singleton for convenient ``from src.config import CONFIG`` use.
CONFIG: dict[str, Any] = load_config()
