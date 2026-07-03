from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any
from uuid import uuid4

import yaml


class RunRecorder:
    def __init__(
        self,
        output_root: str | Path,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        name = _validate_run_name(run_name) if run_name is not None else _default_run_name()
        self.run_dir = Path(output_root) / name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for subdir in ("frames", "masks", "overlays"):
            (self.run_dir / subdir).mkdir(exist_ok=True)

        with (self.run_dir / "run_config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(config or {}, file, allow_unicode=True, sort_keys=False)

    def write_detection(self, record: dict[str, Any]) -> None:
        self._write_jsonl("detections.jsonl", record)

    def write_candidate(self, record: dict[str, Any]) -> None:
        self._write_jsonl("candidates.jsonl", record)

    def write_target(self, record: dict[str, Any]) -> None:
        self._write_jsonl("targets.jsonl", record)

    def write_error(self, record: dict[str, Any]) -> None:
        self._write_jsonl("errors.jsonl", record)

    def _write_jsonl(self, filename: str, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        with (self.run_dir / filename).open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _default_run_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _validate_run_name(run_name: str) -> str:
    windows_path = PureWindowsPath(run_name)
    if (
        run_name in {"", ".", ".."}
        or "/" in run_name
        or "\\" in run_name
        or windows_path.drive
        or windows_path.root
    ):
        raise ValueError("run_name must be a single relative directory name")
    return run_name
