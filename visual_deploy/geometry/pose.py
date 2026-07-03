from __future__ import annotations

import numpy as np


def approach_from_normal(normal_xyz: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal_xyz, dtype=np.float32)
    if normal.shape != (3,):
        raise ValueError("normal_xyz must be a 3-vector")
    if not np.all(np.isfinite(normal)):
        raise ValueError("normal_xyz must contain finite values")

    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        raise ValueError("normal_xyz is near-zero")
    return (-normal / norm).astype(np.float32)
