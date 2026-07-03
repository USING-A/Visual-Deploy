from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from visual_deploy.types import CameraIntrinsics


@dataclass(frozen=True, eq=False)
class GraspPatch:
    u_px: float
    v_px: float
    z_m: float
    normal_xyz: np.ndarray
    score: float
    valid_depth_ratio: float
    valid_depth_count: int
    plane_rmse_m: float
    depth_mad_m: float
    boundary_distance_px: float
    component_area_px: int


def _validate_inputs(mask: np.ndarray, depth_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_arr = np.asarray(mask, dtype=bool)
    depth_arr = np.asarray(depth_mm, dtype=np.float32)
    if mask_arr.ndim != 2 or depth_arr.ndim != 2:
        raise ValueError("mask and depth_mm must be 2-D arrays")
    if mask_arr.shape != depth_arr.shape:
        raise ValueError("mask and depth_mm must have the same shape")
    return mask_arr, depth_arr


def _validate_intrinsics(intr: CameraIntrinsics) -> None:
    for name in ("fx", "fy", "depth_scale"):
        value = float(getattr(intr, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"camera {name} must be finite and positive")
    for name in ("ppx", "ppy"):
        value = float(getattr(intr, name))
        if not np.isfinite(value):
            raise ValueError(f"camera {name} must be finite")


def _largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        q: deque[tuple[int, int]] = deque([(int(y), int(x))])
        seen[y, x] = True
        pixels: list[tuple[int, int]] = []
        while q:
            cy, cx = q.popleft()
            pixels.append((cy, cx))
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if len(pixels) > len(best):
            best = pixels

    component = np.zeros_like(mask, dtype=bool)
    if best:
        ys, xs = zip(*best)
        component[np.asarray(ys), np.asarray(xs)] = True
    return component


def _chamfer_distance_to_background(component: np.ndarray) -> np.ndarray:
    h, w = component.shape
    inf = np.float32(h + w + 1)
    dist = np.full((h, w), inf, dtype=np.float32)
    dist[~component] = 0.0
    root2 = np.float32(np.sqrt(2.0))

    for y in range(h):
        for x in range(w):
            if not component[y, x]:
                continue
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y - 1, x - 1] + root2)
                if x + 1 < w:
                    best = min(best, dist[y - 1, x + 1] + root2)
            if x > 0:
                best = min(best, dist[y, x - 1] + 1.0)
            dist[y, x] = best

    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if not component[y, x]:
                continue
            best = dist[y, x]
            if y + 1 < h:
                best = min(best, dist[y + 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y + 1, x - 1] + root2)
                if x + 1 < w:
                    best = min(best, dist[y + 1, x + 1] + root2)
            if x + 1 < w:
                best = min(best, dist[y, x + 1] + 1.0)
            dist[y, x] = best
    return dist


def _backproject_pixels(
    u_px: np.ndarray,
    v_px: np.ndarray,
    depth_mm: np.ndarray,
    intr: CameraIntrinsics,
) -> np.ndarray:
    _validate_intrinsics(intr)
    z_m = depth_mm.astype(np.float32) * np.float32(intr.depth_scale)
    x_m = ((u_px.astype(np.float32) - np.float32(intr.ppx)) / np.float32(intr.fx)) * z_m
    y_m = ((v_px.astype(np.float32) - np.float32(intr.ppy)) / np.float32(intr.fy)) * z_m
    return np.stack([x_m, y_m, z_m], axis=1).astype(np.float32)


def _plane_stats(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1].astype(np.float32)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        raise ValueError("degenerate local patch")
    normal = normal / norm
    if normal[2] > 0.0:
        normal = -normal
    residuals = centered @ normal
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    return normal.astype(np.float32), rmse


def _candidate_score(
    valid_depth_ratio: float,
    boundary_distance_px: float,
    patch_radius_px: int,
    centroid_distance_px: float,
    max_centroid_distance_px: float,
    plane_rmse_m: float,
    max_plane_rmse_m: float,
    depth_mad_m: float,
    max_depth_mad_m: float,
    valid_depth_count: int,
    patch_area: int,
) -> float:
    boundary_score = min(boundary_distance_px / max(float(patch_radius_px) * 2.0, 1.0), 1.0)
    centroid_score = max(0.0, 1.0 - centroid_distance_px / max(max_centroid_distance_px, 1.0))
    plane_score = max(0.0, 1.0 - plane_rmse_m / max(max_plane_rmse_m, 1e-9))
    mad_score = max(0.0, 1.0 - depth_mad_m / max(max_depth_mad_m, 1e-9))
    support_score = min(valid_depth_count / max(float(patch_area), 1.0), 1.0)
    return float(
        0.25 * valid_depth_ratio
        + 0.30 * boundary_score
        + 0.20 * plane_score
        + 0.10 * mad_score
        + 0.10 * support_score
        + 0.05 * centroid_score
    )


def _iter_patch_candidates(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intr: CameraIntrinsics,
    patch_radius_px: int,
    stride_px: int,
    min_component_area_px: int,
    min_valid_depth_ratio: float,
    min_valid_depth_count: int,
    min_boundary_distance_px: float | None,
    max_plane_rmse_m: float,
    max_depth_mad_m: float,
) -> list[GraspPatch]:
    mask_arr, depth_arr = _validate_inputs(mask, depth_mm)
    if patch_radius_px <= 0 or stride_px <= 0:
        raise ValueError("patch_radius_px and stride_px must be positive")

    component = _largest_component(mask_arr)
    component_area = int(component.sum())
    if component_area < int(min_component_area_px):
        return []

    dist = _chamfer_distance_to_background(component)
    ys, xs = np.nonzero(component)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    centroid_y = float(ys.mean())
    centroid_x = float(xs.mean())
    max_centroid_distance = float(np.hypot(y1 - y0, x1 - x0))
    rr = int(patch_radius_px)
    yy, xx = np.ogrid[-rr : rr + 1, -rr : rr + 1]
    circle = (yy * yy + xx * xx) <= rr * rr
    circle_area = int(circle.sum())
    candidates: list[GraspPatch] = []

    for cy in range(y0, y1 + 1, int(stride_px)):
        for cx in range(x0, x1 + 1, int(stride_px)):
            if not component[cy, cx]:
                continue

            py0, py1 = cy - rr, cy + rr + 1
            px0, px1 = cx - rr, cx + rr + 1
            if py0 < 0 or px0 < 0 or py1 > component.shape[0] or px1 > component.shape[1]:
                continue

            local_component = component[py0:py1, px0:px1]
            patch_mask = circle & local_component
            patch_area = int(patch_mask.sum())
            if patch_area < max(min_valid_depth_count, circle_area // 3):
                continue

            local_depth = depth_arr[py0:py1, px0:px1]
            valid = patch_mask & np.isfinite(local_depth) & (local_depth > 0.0)
            valid_count = int(valid.sum())
            valid_ratio = float(valid_count / max(patch_area, 1))
            if valid_count < min_valid_depth_count or valid_ratio < min_valid_depth_ratio:
                continue

            vy_local, vx_local = np.nonzero(valid)
            vx = vx_local.astype(np.float32) + float(px0)
            vy = vy_local.astype(np.float32) + float(py0)
            depths_mm = local_depth[valid]
            points = _backproject_pixels(vx, vy, depths_mm, intr)
            try:
                normal, plane_rmse = _plane_stats(points)
            except ValueError:
                continue
            if plane_rmse > max_plane_rmse_m:
                continue

            depths_m = depths_mm.astype(np.float32) * np.float32(intr.depth_scale)
            z_m = float(np.median(depths_m))
            depth_mad = float(np.median(np.abs(depths_m - z_m)))
            if depth_mad > max_depth_mad_m:
                continue

            boundary_distance = float(dist[cy, cx])
            boundary_min = float(patch_radius_px) if min_boundary_distance_px is None else float(min_boundary_distance_px)
            if boundary_distance < boundary_min:
                continue

            centroid_distance = float(np.hypot(float(cy) - centroid_y, float(cx) - centroid_x))
            score = _candidate_score(
                valid_ratio,
                boundary_distance,
                rr,
                centroid_distance,
                max_centroid_distance,
                plane_rmse,
                max_plane_rmse_m,
                depth_mad,
                max_depth_mad_m,
                valid_count,
                patch_area,
            )
            candidates.append(
                GraspPatch(
                    u_px=float(cx),
                    v_px=float(cy),
                    z_m=z_m,
                    normal_xyz=normal,
                    score=score,
                    valid_depth_ratio=valid_ratio,
                    valid_depth_count=valid_count,
                    plane_rmse_m=plane_rmse,
                    depth_mad_m=depth_mad,
                    boundary_distance_px=boundary_distance,
                    component_area_px=component_area,
                )
            )

    return candidates


def select_grasp_patch(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intr: CameraIntrinsics,
    *,
    patch_radius_px: int = 8,
    stride_px: int = 4,
    min_component_area_px: int = 64,
    min_valid_depth_ratio: float = 0.7,
    min_valid_depth_count: int = 20,
    min_boundary_distance_px: float | None = None,
    min_score: float = 0.0,
    max_plane_rmse_m: float = 0.006,
    max_depth_mad_m: float = 0.012,
) -> GraspPatch | None:
    _validate_intrinsics(intr)
    candidates = _iter_patch_candidates(
        mask,
        depth_mm,
        intr,
        patch_radius_px,
        stride_px,
        min_component_area_px,
        min_valid_depth_ratio,
        min_valid_depth_count,
        min_boundary_distance_px,
        max_plane_rmse_m,
        max_depth_mad_m,
    )
    if not candidates:
        return None

    best = max(candidates, key=lambda candidate: candidate.score)
    if best.score < float(min_score):
        return None
    return best
