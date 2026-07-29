from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import cv2
import numpy as np

from visual_deploy.geometry.roi import RoiTransform


@dataclass(frozen=True, eq=False)
class FusedTrackDepth:
    track_id: int
    depth_roi_mm: np.ndarray
    valid_ratio: float
    source_frame_count: int
    age_ms: float
    profile: DepthFusionProfile | None = None


@dataclass(frozen=True)
class DepthFusionProfile:
    remap_ms: float
    median_ms: float
    history_frame_count: int
    remap_count: int


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
        min_roi_iou: float = 0.5,
        profiling_enabled: bool = False,
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
        if not np.isfinite(min_roi_iou) or min_roi_iou < 0.0 or min_roi_iou > 1.0:
            raise ValueError("min_roi_iou must be between 0 and 1")

        self.window_size = int(window_size)
        self.min_depth_mm = float(min_depth_mm)
        self.max_depth_mm = float(max_depth_mm)
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_roi_iou = float(min_roi_iou)
        self.profiling_enabled = bool(profiling_enabled)
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
        if history and history[-1].depth_roi_mm.shape != depth.shape:
            raise ValueError("depth_roi_mm shape must match existing track history")

        compatible = [
            entry
            for entry in history
            if _crop_iou(entry.roi_transform.crop_xyxy, roi_transform.crop_xyxy) >= self.min_roi_iou
        ]
        if len(compatible) != len(history):
            history.clear()
            history.extend(compatible)

        current_valid = self._valid_depth(depth)
        history.append(
            _DepthEntry(
                frame_id=frame_id,
                timestamp_ms=timestamp,
                roi_transform=roi_transform,
                depth_roi_mm=current_valid,
            )
        )

        if len(history) == 1:
            fused = np.nan_to_num(current_valid, nan=0.0).astype(np.float32, copy=False)
            profile = DepthFusionProfile(0.0, 0.0, 1, 0) if self.profiling_enabled else None
            return self._result(track_id, fused, history, timestamp, profile)

        remap_count = sum(entry.roi_transform.crop_xyxy != roi_transform.crop_xyxy for entry in history)
        remap_started = perf_counter() if self.profiling_enabled else 0.0
        stack = np.stack(
            [_remap_depth(entry, roi_transform) for entry in history],
            axis=0,
        )
        remap_ms = (perf_counter() - remap_started) * 1000.0 if self.profiling_enabled else 0.0
        median_started = perf_counter() if self.profiling_enabled else 0.0
        fused = _nanmedian_small(stack)
        median_ms = (perf_counter() - median_started) * 1000.0 if self.profiling_enabled else 0.0

        fused = np.where(np.isnan(fused), current_valid, fused)
        fused = np.where(np.isnan(fused), 0.0, fused).astype(np.float32)

        profile = (
            DepthFusionProfile(remap_ms, median_ms, len(history), remap_count)
            if self.profiling_enabled
            else None
        )
        return self._result(track_id, fused, history, timestamp, profile)

    def _result(
        self,
        track_id: int,
        fused: np.ndarray,
        history: deque[_DepthEntry],
        timestamp_ms: float,
        profile: DepthFusionProfile | None,
    ) -> FusedTrackDepth:
        valid_ratio = float(np.count_nonzero(fused > 0.0) / fused.size)
        if valid_ratio < self.min_valid_ratio:
            fused = np.zeros_like(fused, dtype=np.float32)
            valid_ratio = 0.0

        age_ms = timestamp_ms - history[0].timestamp_ms
        return FusedTrackDepth(
            track_id=track_id,
            depth_roi_mm=fused,
            valid_ratio=valid_ratio,
            source_frame_count=len(history),
            age_ms=age_ms,
            profile=profile,
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


def _crop_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    intersection_width = max(0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = intersection_width * intersection_height
    first_area = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    second_area = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _remap_depth(entry: _DepthEntry, target: RoiTransform) -> np.ndarray:
    if entry.roi_transform.crop_xyxy == target.crop_xyxy:
        return entry.depth_roi_mm

    source = entry.roi_transform
    size = target.roi_size
    target_x0, target_y0, _, _ = target.crop_xyxy
    source_x0, source_y0, _, _ = source.crop_xyxy
    target_u, target_v = _roi_grid(size)
    image_x = target_x0 + (target_u + 0.5) * (target.crop_width / float(size))
    image_y = target_y0 + (target_v + 0.5) * (target.crop_height / float(size))
    source_u = (image_x - source_x0) * (size / float(source.crop_width)) - 0.5
    source_v = (image_y - source_y0) * (size / float(source.crop_height)) - 0.5
    return cv2.remap(
        entry.depth_roi_mm,
        source_u,
        source_v,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    ).astype(np.float32, copy=False)


def _nanmedian_small(stack: np.ndarray) -> np.ndarray:
    """Compute a float32 NaN-aware median for a small temporal stack.

    ``stack`` is a disposable buffer containing only finite depths or NaNs. It is
    sorted in place to avoid the general-purpose masked-array path used by
    ``numpy.nanmedian``.
    """
    valid = np.isfinite(stack)
    valid_counts = valid.sum(axis=0, dtype=np.intp)
    stack[~valid] = np.inf
    stack.sort(axis=0)
    lower_indices = np.maximum(valid_counts - 1, 0) // 2
    upper_indices = valid_counts // 2
    lower = np.take_along_axis(stack, lower_indices[None], axis=0)[0]
    upper = np.take_along_axis(stack, upper_indices[None], axis=0)[0]
    median = (lower + upper) * np.float32(0.5)
    median[valid_counts == 0] = np.nan
    return median.astype(np.float32, copy=False)


@lru_cache(maxsize=8)
def _roi_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.arange(size, dtype=np.float32)
    target_u, target_v = np.meshgrid(coordinates, coordinates)
    target_u.setflags(write=False)
    target_v.setflags(write=False)
    return target_u, target_v
