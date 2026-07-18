from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
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
        (self.run_dir / "depth").mkdir(exist_ok=True)

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

    def write_timing(self, record: dict[str, Any]) -> None:
        self._write_jsonl("timings.jsonl", record)

    def write_event(self, record: dict[str, Any]) -> None:
        self._write_jsonl("events.jsonl", record)

    def save_event_artifacts(
        self,
        frame_id: int,
        *,
        color_bgr: np.ndarray | None = None,
        depth_mm: np.ndarray | None = None,
        masks: list[tuple[int, np.ndarray]] | None = None,
        overlay_bgr: np.ndarray | None = None,
    ) -> dict[str, Any]:
        stem = f"{int(frame_id):06d}"
        artifacts: dict[str, Any] = {}
        if color_bgr is not None:
            path = self.run_dir / "frames" / f"{stem}.jpg"
            _write_image(path, color_bgr)
            artifacts["frame"] = str(path.relative_to(self.run_dir))
        if depth_mm is not None:
            path = self.run_dir / "depth" / f"{stem}_mm.npy"
            np.save(path, np.asarray(depth_mm, dtype=np.float32))
            artifacts["depth"] = str(path.relative_to(self.run_dir))
        mask_paths: list[str] = []
        for track_id, mask in masks or []:
            path = self.run_dir / "masks" / f"{stem}_track_{int(track_id)}.png"
            _write_image(path, np.asarray(mask, dtype=np.uint8) * 255)
            mask_paths.append(str(path.relative_to(self.run_dir)))
        if mask_paths:
            artifacts["masks"] = mask_paths
        if overlay_bgr is not None:
            path = self.run_dir / "overlays" / f"{stem}.jpg"
            _write_image(path, overlay_bgr)
            artifacts["overlay"] = str(path.relative_to(self.run_dir))
        return artifacts

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


def _write_image(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image)
    if array.size == 0 or not cv2.imwrite(str(path), array):
        raise OSError(f"failed to write image artifact: {path}")
