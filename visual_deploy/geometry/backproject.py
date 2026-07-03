from __future__ import annotations

import numpy as np

from visual_deploy.types import CameraIntrinsics


def backproject_pixel(u: float, v: float, depth_raw: float, intr: CameraIntrinsics) -> np.ndarray:
    if intr.fx == 0 or intr.fy == 0:
        raise ValueError("camera focal lengths must be non-zero")

    z_m = depth_raw * intr.depth_scale
    x_m = ((u - intr.ppx) / intr.fx) * z_m
    y_m = ((v - intr.ppy) / intr.fy) * z_m
    return np.array([x_m, y_m, z_m], dtype=np.float32)
