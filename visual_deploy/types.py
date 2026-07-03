from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float
    depth_scale: float = 0.001


@dataclass(eq=False)
class DeployFrame:
    frame_id: int
    timestamp_ms: float
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    intrinsics: CameraIntrinsics
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthStats:
    median_mm: float | None = None
    mean_mm: float | None = None
    std_mm: float | None = None
    valid_ratio: float = 0.0
    valid_count: int = 0
    state: str = "invalid"


@dataclass
class Detection:
    bbox_xyxy: BBoxXYXY
    class_id: int
    confidence: float
    label: str = ""
    track_id: int | None = None
    weighted_confidence: float | None = None
    depth_stats: DepthStats = field(default_factory=DepthStats)

    def __post_init__(self) -> None:
        if self.weighted_confidence is None:
            self.weighted_confidence = float(self.confidence)


@dataclass
class Track:
    track_id: int
    bbox_xyxy: BBoxXYXY
    class_id: int
    confidence: float
    label: str
    state: str
    hits: int
    lost: int
    depth_stats: DepthStats = field(default_factory=DepthStats)
    weighted_confidence: float | None = None


@dataclass
class GraspTarget:
    valid: bool
    frame_id: int
    reason: str | None = None
    track_id: int | None = None
    u_px: float | None = None
    v_px: float | None = None
    z_mm: float | None = None
    xyz_camera_m: tuple[float, float, float] | None = None
    normal_xyz: tuple[float, float, float] | None = None
    approach_axis: tuple[float, float, float] | None = None
    target_score: float | None = None
