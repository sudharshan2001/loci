"""Load configs/default.yaml and resolve repo-relative paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@lru_cache(maxsize=4)
def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(rel: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()
