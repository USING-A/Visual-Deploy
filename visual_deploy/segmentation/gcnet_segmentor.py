from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from os import PathLike

import numpy as np

from visual_deploy.inference.session import InferenceSession, create_inference_session


@dataclass(frozen=True, eq=False)
class SegmentResult:
    prob_256: np.ndarray
    mask_256: np.ndarray
    mask_area: int
    largest_component_ratio: float


class GCNetSegmentor:
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: str = "cpu",
        backend: str = "auto",
        threshold: float = 0.5,
        *,
        session: InferenceSession | None = None,
    ) -> None:
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0.0, 1.0]")
        self.session = session or create_inference_session(model_path, device=device, backend=backend)
        if set(self.session.input_names) != {"bgr", "depth_mm"}:
            raise ValueError(f"GCNet inputs must be bgr and depth_mm, got {self.session.input_names}")
        if "prob" not in self.session.output_names:
            raise ValueError(f"GCNet output must include prob, got {self.session.output_names}")

    def infer(self, color_bgr_256: np.ndarray, depth_mm_256: np.ndarray) -> SegmentResult:
        color = np.asarray(color_bgr_256)
        depth = np.asarray(depth_mm_256)
        self._validate_inputs(color, depth)
        bgr = np.ascontiguousarray(color.transpose(2, 0, 1)[None], dtype=np.float32)
        depth_mm = np.ascontiguousarray(depth[None, None], dtype=np.float32)
        outputs = self.session.run({"bgr": bgr, "depth_mm": depth_mm})
        prob_256 = self._to_256_numpy(outputs["prob"], "prob").astype(np.float32, copy=False)
        mask_256 = prob_256 >= self.threshold
        mask_area = int(mask_256.sum())
        return SegmentResult(prob_256, mask_256, mask_area, _largest_component_ratio(mask_256))

    @staticmethod
    def _validate_inputs(color: np.ndarray, depth: np.ndarray) -> None:
        if color.shape != (256, 256, 3):
            raise ValueError("color_bgr_256 must have shape (256, 256, 3)")
        if depth.shape != (256, 256):
            raise ValueError("depth_mm_256 must have shape (256, 256)")
        if not np.isfinite(color.astype(np.float32, copy=False)).all():
            raise ValueError("color_bgr_256 must contain only finite values")
        if not np.isfinite(depth.astype(np.float32, copy=False)).all():
            raise ValueError("depth_mm_256 must contain only finite values")

    @staticmethod
    def _to_256_numpy(value: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.shape == (1, 1, 256, 256):
            arr = arr[0, 0]
        elif arr.shape != (256, 256):
            raise ValueError(f"model {name} output must have shape (1, 1, 256, 256) or (256, 256)")
        if not np.isfinite(arr.astype(np.float32, copy=False)).all():
            raise ValueError(f"model {name} output must contain only finite values")
        return arr


def _largest_component_ratio(mask: np.ndarray) -> float:
    total = int(mask.sum())
    if total == 0:
        return 0.0
    try:
        import cv2
    except ImportError:
        return _largest_component_ratio_bfs(mask, total)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if component_count <= 1:
        return 0.0
    return float(int(stats[1:, cv2.CC_STAT_AREA].max()) / total)


def _largest_component_ratio_bfs(mask: np.ndarray, total: int) -> float:
    visited = np.zeros(mask.shape, dtype=bool)
    largest = 0
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            size = 0
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                r, c = queue.popleft()
                size += 1
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            largest = max(largest, size)
    return float(largest / total)
