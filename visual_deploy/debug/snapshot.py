from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from visual_deploy.types import Detection, GraspTarget


@dataclass(frozen=True)
class DebugSnapshot:
    frame_id: int
    timestamp_ms: float
    detections: list[Detection] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    target: GraspTarget | None = None

