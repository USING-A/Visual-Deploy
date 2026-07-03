from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from visual_deploy.types import Detection, GraspTarget


@dataclass(frozen=True)
class DebugOverlayOptions:
    mask_alpha: float = 0.28
    show_labels: bool = True
    show_status: bool = True
    max_status_lines: int = 4
    mask_color_bgr: tuple[int, int, int] = (0, 180, 255)
    bbox_color_bgr: tuple[int, int, int] = (70, 220, 70)
    target_color_bgr: tuple[int, int, int] = (0, 0, 255)
    status_lines: tuple[str, ...] = field(default_factory=tuple)


def render_debug_overlay(
    color_bgr: np.ndarray,
    *,
    detections: Iterable[Detection] = (),
    target: GraspTarget | None = None,
    masks: Iterable[np.ndarray] = (),
    options: DebugOverlayOptions | None = None,
) -> np.ndarray:
    opts = options or DebugOverlayOptions()
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("color_bgr must have shape (height, width, 3)")

    overlay = image.copy()
    _draw_masks(overlay, masks, opts)
    for detection in detections:
        _draw_detection(overlay, detection, opts)
    if target is not None and target.valid:
        _draw_target(overlay, target, opts)
    if opts.show_status and opts.status_lines:
        _draw_status(overlay, opts.status_lines[: opts.max_status_lines])
    return overlay


def depth_to_bgr(depth_mm: np.ndarray, min_mm: float | None = None, max_mm: float | None = None) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_mm must be a 2D array")

    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    lo = float(min_mm) if min_mm is not None else float(np.percentile(depth[valid], 2))
    hi = float(max_mm) if max_mm is not None else float(np.percentile(depth[valid], 98))
    if hi <= lo:
        hi = lo + 1.0

    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    preview[valid] = (normalized[valid] * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _draw_masks(image: np.ndarray, masks: Iterable[np.ndarray], opts: DebugOverlayOptions) -> None:
    color = np.array(opts.mask_color_bgr, dtype=np.uint8)
    for mask in masks:
        mask_arr = np.asarray(mask).astype(bool, copy=False)
        if mask_arr.shape != image.shape[:2] or not np.any(mask_arr):
            continue
        image[mask_arr] = (
            image[mask_arr].astype(np.float32) * (1.0 - opts.mask_alpha)
            + color.astype(np.float32) * opts.mask_alpha
        ).astype(np.uint8)


def _draw_detection(image: np.ndarray, detection: Detection, opts: DebugOverlayOptions) -> None:
    x0, y0, x1, y1 = (int(round(value)) for value in detection.bbox_xyxy)
    cv2.rectangle(image, (x0, y0), (x1, y1), opts.bbox_color_bgr, 2)
    if not opts.show_labels:
        return
    track = "-" if detection.track_id is None else str(detection.track_id)
    label = f"#{track} {detection.confidence:.2f}"
    cv2.putText(
        image,
        label,
        (x0, max(14, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        opts.bbox_color_bgr,
        1,
        cv2.LINE_AA,
    )


def _draw_target(image: np.ndarray, target: GraspTarget, opts: DebugOverlayOptions) -> None:
    if target.u_px is None or target.v_px is None:
        return
    u = int(round(target.u_px))
    v = int(round(target.v_px))
    cv2.circle(image, (u, v), 6, opts.target_color_bgr, 2)
    cv2.drawMarker(image, (u, v), opts.target_color_bgr, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    if not opts.show_labels:
        return
    label = f"z={target.z_mm:.0f}mm" if target.z_mm is not None else "target"
    cv2.putText(image, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, opts.target_color_bgr, 1, cv2.LINE_AA)


def _draw_status(image: np.ndarray, lines: tuple[str, ...]) -> None:
    if not lines:
        return
    pad = 8
    line_h = 18
    width = min(image.shape[1], 360)
    height = pad * 2 + line_h * len(lines)
    panel = image[0:height, 0:width].copy()
    panel[:] = (24, 24, 24)
    image[0:height, 0:width] = cv2.addWeighted(image[0:height, 0:width], 0.35, panel, 0.65, 0.0)
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            str(line)[:64],
            (pad, pad + 13 + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
