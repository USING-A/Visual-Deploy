from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")
    return data


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    source = Path(config_path)
    root = source.parent.parent if source.parent.name == "configs" else source.parent
    return root / path
