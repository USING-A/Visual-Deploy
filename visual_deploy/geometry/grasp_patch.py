from __future__ import annotations

from dataclasses import dataclass

import cv2
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
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def _chamfer_distance_to_background(component: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(
        component.astype(np.uint8),
        distanceType=cv2.DIST_L2,
        maskSize=cv2.DIST_MASK_PRECISE,
    ).astype(np.float32, copy=False)


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


def _masked_median(values: np.ndarray, valid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    flattened = np.where(valid, values, np.inf).reshape(values.shape[0], -1)
    ordered = np.sort(flattened, axis=1)
    rows = np.arange(ordered.shape[0])
    lower = (counts - 1) // 2
    upper = counts // 2
    return ((ordered[rows, lower] + ordered[rows, upper]) * 0.5).astype(np.float32, copy=False)


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
    _validate_intrinsics(intr)
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
    center_ys, center_xs = np.meshgrid(
        np.arange(y0, y1 + 1, int(stride_px), dtype=np.int32),
        np.arange(x0, x1 + 1, int(stride_px), dtype=np.int32),
        indexing="ij",
    )
    center_ys = center_ys.ravel()
    center_xs = center_xs.ravel()
    height, width = component.shape
    boundary_min = float(patch_radius_px) if min_boundary_distance_px is None else float(min_boundary_distance_px)
    eligible = (
        component[center_ys, center_xs]
        & (center_ys >= rr)
        & (center_xs >= rr)
        & (center_ys + rr < height)
        & (center_xs + rr < width)
        & (dist[center_ys, center_xs] >= boundary_min)
    )
    center_ys, center_xs = center_ys[eligible], center_xs[eligible]
    if center_ys.size == 0:
        return []

    size = 2 * rr + 1
    component_windows = np.lib.stride_tricks.sliding_window_view(component, (size, size))[
        center_ys - rr, center_xs - rr
    ]
    patch_masks = component_windows & circle[None]
    patch_areas = patch_masks.sum(axis=(1, 2), dtype=np.int32)
    keep = patch_areas >= max(min_valid_depth_count, circle_area // 3)
    center_ys, center_xs, patch_masks, patch_areas = (
        center_ys[keep],
        center_xs[keep],
        patch_masks[keep],
        patch_areas[keep],
    )
    if center_ys.size == 0:
        return []

    depth_windows = np.lib.stride_tricks.sliding_window_view(depth_arr, (size, size))[
        center_ys - rr, center_xs - rr
    ]
    valid = patch_masks & np.isfinite(depth_windows) & (depth_windows > 0.0)
    valid_counts = valid.sum(axis=(1, 2), dtype=np.int32)
    valid_ratios = valid_counts.astype(np.float32) / np.maximum(patch_areas, 1)
    keep = (valid_counts >= min_valid_depth_count) & (valid_ratios >= min_valid_depth_ratio)
    center_ys, center_xs, patch_areas, depth_windows, valid, valid_counts, valid_ratios = (
        center_ys[keep],
        center_xs[keep],
        patch_areas[keep],
        depth_windows[keep],
        valid[keep],
        valid_counts[keep],
        valid_ratios[keep],
    )
    if center_ys.size == 0:
        return []

    offsets = np.arange(-rr, rr + 1, dtype=np.float32)
    pixel_x = center_xs[:, None, None].astype(np.float32) + offsets[None, None, :]
    pixel_y = center_ys[:, None, None].astype(np.float32) + offsets[None, :, None]
    z_m = np.where(valid, depth_windows * np.float32(intr.depth_scale), 0.0).astype(np.float32)
    x_m = ((pixel_x - np.float32(intr.ppx)) / np.float32(intr.fx)) * z_m
    y_m = ((pixel_y - np.float32(intr.ppy)) / np.float32(intr.fy)) * z_m
    points = np.stack(np.broadcast_arrays(x_m, y_m, z_m), axis=-1).astype(np.float64)
    weights = valid.astype(np.float64)[..., None]
    counts = valid_counts.astype(np.float64)[:, None]
    means = (points * weights).sum(axis=(1, 2), dtype=np.float64) / counts
    centered = (points - means[:, None, None, :]) * weights
    covariance = np.einsum("nhwi,nhwj->nij", centered, centered, optimize=True) / counts[:, :, None]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0].astype(np.float32, copy=False)
    normals[normals[:, 2] > 0.0] *= -1.0
    plane_rmse = np.sqrt(np.maximum(eigenvalues[:, 0], 0.0)).astype(np.float32, copy=False)

    median_depth = _masked_median(z_m, valid, valid_counts)
    depth_mad = _masked_median(np.abs(z_m - median_depth[:, None, None]), valid, valid_counts)
    keep = (plane_rmse <= max_plane_rmse_m) & (depth_mad <= max_depth_mad_m)
    center_ys, center_xs, patch_areas, valid_counts, valid_ratios, normals, plane_rmse, median_depth, depth_mad = (
        center_ys[keep],
        center_xs[keep],
        patch_areas[keep],
        valid_counts[keep],
        valid_ratios[keep],
        normals[keep],
        plane_rmse[keep],
        median_depth[keep],
        depth_mad[keep],
    )
    if center_ys.size == 0:
        return []

    boundary_distances = dist[center_ys, center_xs].astype(np.float32, copy=False)
    centroid_distances = np.hypot(center_ys - centroid_y, center_xs - centroid_x).astype(np.float32, copy=False)
    boundary_scores = np.minimum(boundary_distances / max(float(rr) * 2.0, 1.0), 1.0)
    centroid_scores = np.maximum(0.0, 1.0 - centroid_distances / max(max_centroid_distance, 1.0))
    plane_scores = np.maximum(0.0, 1.0 - plane_rmse / max(max_plane_rmse_m, 1e-9))
    mad_scores = np.maximum(0.0, 1.0 - depth_mad / max(max_depth_mad_m, 1e-9))
    support_scores = np.minimum(valid_counts / np.maximum(patch_areas.astype(np.float32), 1.0), 1.0)
    scores = (
        0.25 * valid_ratios
        + 0.30 * boundary_scores
        + 0.20 * plane_scores
        + 0.10 * mad_scores
        + 0.10 * support_scores
        + 0.05 * centroid_scores
    )
    index = int(np.argmax(scores))
    return [
        GraspPatch(
            u_px=float(center_xs[index]),
            v_px=float(center_ys[index]),
            z_m=float(median_depth[index]),
            normal_xyz=normals[index],
            score=float(scores[index]),
            valid_depth_ratio=float(valid_ratios[index]),
            valid_depth_count=int(valid_counts[index]),
            plane_rmse_m=float(plane_rmse[index]),
            depth_mad_m=float(depth_mad[index]),
            boundary_distance_px=float(boundary_distances[index]),
            component_area_px=component_area,
        )
    ]


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
