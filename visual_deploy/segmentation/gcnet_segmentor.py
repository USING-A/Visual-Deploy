from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, eq=False)
class SegmentResult:
    prob_256: np.ndarray
    mask_256: np.ndarray
    mask_area: int
    largest_component_ratio: float


class GCNetSegmentor:
    def __init__(self, model_path: str | PathLike[str], device: str | torch.device = "cpu") -> None:
        self.device = self._resolve_device(device)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def infer(self, color_bgr_256: np.ndarray, depth_mm_256: np.ndarray) -> SegmentResult:
        color = np.asarray(color_bgr_256)
        depth = np.asarray(depth_mm_256)
        self._validate_inputs(color, depth)

        bgr = np.ascontiguousarray(color.transpose(2, 0, 1), dtype=np.float32)
        depth_mm = np.ascontiguousarray(depth, dtype=np.float32)
        bgr_tensor = torch.from_numpy(bgr).unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(depth_mm).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob, mask = self.model(bgr_tensor, depth_tensor)

        prob_256 = self._to_256_numpy(prob, "prob").astype(np.float32, copy=False)
        mask_256 = self._to_256_numpy(mask, "mask") > 0
        mask_area = int(mask_256.sum())
        largest_component_ratio = _largest_component_ratio(mask_256)
        return SegmentResult(
            prob_256=prob_256,
            mask_256=mask_256,
            mask_area=mask_area,
            largest_component_ratio=largest_component_ratio,
        )

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU", RuntimeWarning, stacklevel=2)
            return torch.device("cpu")
        return requested

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
    def _to_256_numpy(tensor: Any, name: str) -> np.ndarray:
        arr = tensor.detach().cpu().numpy()
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
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(largest / total)


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
