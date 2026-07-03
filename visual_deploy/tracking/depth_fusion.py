from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import warnings

import numpy as np

from visual_deploy.geometry.roi import RoiTransform


@dataclass(frozen=True, eq=False)
class FusedTrackDepth:
    track_id: int
    depth_roi_mm: np.ndarray
    valid_ratio: float
    source_frame_count: int
    age_ms: float


@dataclass(frozen=True)
class _DepthEntry:
    frame_id: int
    timestamp_ms: float
    roi_transform: RoiTransform
    depth_roi_mm: np.ndarray


class DepthFusionBuffer:
    def __init__(
        self,
        window_size: int = 5,
        min_depth_mm: float = 100.0,
        max_depth_mm: float = 5000.0,
        min_valid_ratio: float = 0.3,
    ) -> None:
        if not np.isfinite(window_size) or int(window_size) != window_size or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not np.isfinite(min_depth_mm) or min_depth_mm <= 0:
            raise ValueError("min_depth_mm must be finite and positive")
        if not np.isfinite(max_depth_mm) or max_depth_mm <= 0:
            raise ValueError("max_depth_mm must be finite and positive")
        if min_depth_mm > max_depth_mm:
            raise ValueError("min_depth_mm must be less than or equal to max_depth_mm")
        if not np.isfinite(min_valid_ratio) or min_valid_ratio < 0.0 or min_valid_ratio > 1.0:
            raise ValueError("min_valid_ratio must be between 0 and 1")

        self.window_size = int(window_size)
        self.min_depth_mm = float(min_depth_mm)
        self.max_depth_mm = float(max_depth_mm)
        self.min_valid_ratio = float(min_valid_ratio)
        self._history: dict[int, deque[_DepthEntry]] = {}

    def update(
        self,
        track_id: int,
        frame_id: int,
        timestamp_ms: float,
        roi_transform: RoiTransform,
        depth_roi_mm: np.ndarray,
    ) -> FusedTrackDepth:
        timestamp = float(timestamp_ms)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_ms must be finite")

        depth = np.asarray(depth_roi_mm, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("depth_roi_mm must be a 2D array")
        expected_shape = (roi_transform.roi_size, roi_transform.roi_size)
        if depth.shape != expected_shape:
            raise ValueError("depth_roi_mm shape must match roi_transform.roi_size")

        history = self._history.setdefault(track_id, deque(maxlen=self.window_size))
        if history and timestamp < history[-1].timestamp_ms:
            raise ValueError("timestamp_ms must not go backwards for the same track")
        if history and history[-1].roi_transform.crop_xyxy != roi_transform.crop_xyxy:
            history.clear()
        if history and history[-1].depth_roi_mm.shape != depth.shape:
            raise ValueError("depth_roi_mm shape must match existing track history")

        history.append(
            _DepthEntry(
                frame_id=frame_id,
                timestamp_ms=timestamp,
                roi_transform=roi_transform,
                depth_roi_mm=depth.copy(),
            )
        )

        stack = np.stack([self._valid_depth(entry.depth_roi_mm) for entry in history], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fused = np.nanmedian(stack, axis=0).astype(np.float32)

        current_valid = self._valid_depth(depth)
        fused = np.where(np.isnan(fused), current_valid, fused)
        fused = np.where(np.isnan(fused), 0.0, fused).astype(np.float32)

        valid_ratio = float(np.count_nonzero(fused > 0.0) / fused.size)
        if valid_ratio < self.min_valid_ratio:
            fused = np.zeros_like(fused, dtype=np.float32)
            valid_ratio = 0.0

        age_ms = timestamp - history[0].timestamp_ms
        return FusedTrackDepth(
            track_id=track_id,
            depth_roi_mm=fused,
            valid_ratio=valid_ratio,
            source_frame_count=len(history),
            age_ms=age_ms,
        )

    def clear(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._history.clear()
            return
        self._history.pop(track_id, None)

    def _valid_depth(self, depth_roi_mm: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth_roi_mm)
        valid &= depth_roi_mm >= self.min_depth_mm
        valid &= depth_roi_mm <= self.max_depth_mm
        return np.where(valid, depth_roi_mm, np.nan).astype(np.float32)
