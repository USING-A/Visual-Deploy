import numpy as np
import pytest

from visual_deploy.geometry.grasp_patch import (
    GraspPatch,
    _backproject_pixels,
    _candidate_score,
    _chamfer_distance_to_background,
    _largest_component,
    _plane_stats,
    select_grasp_patch,
)
from visual_deploy.geometry.pose import approach_from_normal
from visual_deploy.types import CameraIntrinsics


def test_select_grasp_patch_prefers_flat_interior_depth_support_mm():
    mask = np.zeros((64, 64), dtype=bool)
    yy, xx = np.ogrid[:64, :64]
    mask[(yy - 32) ** 2 + (xx - 32) ** 2 <= 20 ** 2] = True
    depth = np.full((64, 64), np.nan, dtype=np.float32)
    depth[mask] = 800.0
    depth[mask & (xx > 42)] = 900.0
    intr = CameraIntrinsics(fx=100.0, fy=100.0, ppx=32.0, ppy=32.0, depth_scale=0.001)

    out = select_grasp_patch(mask, depth, intr, patch_radius_px=5, stride_px=4)

    assert out is not None
    assert abs(out.u_px - 32.0) <= 4.0
    assert abs(out.v_px - 32.0) <= 4.0
    assert out.z_m == pytest.approx(0.8, abs=0.05)
    assert out.valid_depth_ratio >= 0.95
    assert out.normal_xyz[2] <= 0.0


def test_select_grasp_patch_rejects_unreliable_depth_support():
    mask = np.zeros((48, 48), dtype=bool)
    mask[12:36, 12:36] = True
    depth = np.full((48, 48), np.nan, dtype=np.float32)
    depth[20:24, 20:24] = 700.0
    intr = CameraIntrinsics(fx=100.0, fy=100.0, ppx=24.0, ppy=24.0, depth_scale=0.001)

    out = select_grasp_patch(mask, depth, intr, patch_radius_px=5, stride_px=4, min_valid_depth_ratio=0.6)

    assert out is None


def test_compiled_component_and_distance_preprocessing_preserves_geometry():
    mask = np.zeros((32, 32), dtype=bool)
    mask[2:5, 2:5] = True
    mask[10:26, 8:24] = True

    component = _largest_component(mask)
    distance = _chamfer_distance_to_background(component)

    assert int(component.sum()) == 16 * 16
    assert not component[3, 3]
    assert distance[10, 8] == pytest.approx(1.0)
    assert distance[17, 15] == pytest.approx(8.0)
    assert distance[0, 0] == 0.0


def test_vectorized_candidates_preserve_tilted_surface_reference():
    yy, xx = np.ogrid[:256, :256]
    mask = ((xx - 128) ** 2 / 85**2 + (yy - 126) ** 2 / 70**2) <= 1
    depth = (
        850.0
        + 0.08 * (np.arange(256)[None, :] - 128)
        + 0.05 * (np.arange(256)[:, None] - 126)
        + 1.5 * np.sin(np.arange(256)[None, :] / 19)
    ).astype(np.float32)
    depth[~mask] = np.nan
    intr = CameraIntrinsics(fx=580.0, fy=582.0, ppx=128.0, ppy=126.0, depth_scale=0.001)

    out = select_grasp_patch(mask, depth, intr)

    assert out is not None
    assert (out.u_px, out.v_px) == (127.0, 128.0)
    assert out.z_m == pytest.approx(0.85060555, abs=1e-7)
    np.testing.assert_allclose(out.normal_xyz, [0.10280265, 0.03401249, -0.9941201], atol=2e-5)
    assert out.score == pytest.approx(0.994865013, abs=2e-5)


def test_topk256_preserves_exhaustive_tilted_surface_and_reports_work():
    yy, xx = np.ogrid[:256, :256]
    mask = ((xx - 128) ** 2 / 85**2 + (yy - 126) ** 2 / 70**2) <= 1
    depth = (
        850.0
        + 0.08 * (np.arange(256)[None, :] - 128)
        + 0.05 * (np.arange(256)[:, None] - 126)
        + 1.5 * np.sin(np.arange(256)[None, :] / 19)
    ).astype(np.float32)
    depth[~mask] = np.nan
    intr = CameraIntrinsics(fx=580.0, fy=582.0, ppx=128.0, ppy=126.0, depth_scale=0.001)

    exhaustive = select_grasp_patch(mask, depth, intr)
    topk = select_grasp_patch(mask, depth, intr, candidate_top_k=256)

    assert exhaustive is not None and topk is not None
    assert (topk.u_px, topk.v_px, topk.z_m, topk.score) == (
        exhaustive.u_px,
        exhaustive.v_px,
        exhaustive.z_m,
        exhaustive.score,
    )
    np.testing.assert_allclose(topk.normal_xyz, exhaustive.normal_xyz, atol=3e-5)
    assert topk.search_mode == "topk"
    assert topk.eligible_candidate_count == 950
    assert topk.fine_candidate_count == 256
    assert topk.fallback_used is False


def test_topk_falls_back_to_exhaustive_when_pruned_patch_fails_geometry():
    mask = np.zeros((96, 96), dtype=bool)
    yy, xx = np.ogrid[:96, :96]
    mask[(yy - 48) ** 2 + (xx - 48) ** 2 <= 35**2] = True
    depth = np.full((96, 96), 800.0, dtype=np.float32)
    rng = np.random.default_rng(3)
    depth[38:59, 38:59] += rng.normal(0.0, 40.0, (21, 21)).astype(np.float32)
    intr = CameraIntrinsics(fx=420.0, fy=425.0, ppx=48.0, ppy=47.0, depth_scale=0.001)

    pruned = select_grasp_patch(mask, depth, intr, patch_radius_px=5, stride_px=4, candidate_top_k=1, exhaustive_fallback=False)
    recovered = select_grasp_patch(mask, depth, intr, patch_radius_px=5, stride_px=4, candidate_top_k=1, exhaustive_fallback=True)

    assert pruned is None
    assert recovered is not None
    assert recovered.search_mode == "topk_fallback_exhaustive"
    assert recovered.fallback_used is True
    assert recovered.fine_candidate_count == recovered.eligible_candidate_count + 1


def test_shadow_verification_returns_exhaustive_authority_and_records_miss():
    mask = np.zeros((96, 96), dtype=bool)
    yy, xx = np.ogrid[:96, :96]
    mask[(yy - 48) ** 2 + (xx - 48) ** 2 <= 35**2] = True
    depth = np.full((96, 96), 800.0, dtype=np.float32)
    rng = np.random.default_rng(3)
    depth[38:59, 38:59] += rng.normal(0.0, 40.0, (21, 21)).astype(np.float32)
    intr = CameraIntrinsics(fx=420.0, fy=425.0, ppx=48.0, ppy=47.0, depth_scale=0.001)

    out = select_grasp_patch(
        mask,
        depth,
        intr,
        patch_radius_px=5,
        stride_px=4,
        candidate_top_k=1,
        exhaustive_fallback=False,
        shadow_verify=True,
    )

    assert out is not None
    assert out.search_mode == "topk_shadow_exhaustive"
    assert out.shadow_verified is True
    assert out.shadow_exact_match is False
    assert out.fallback_used is True


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_candidate_top_k_rejects_invalid_values(value):
    mask = np.ones((32, 32), dtype=bool)
    depth = np.full((32, 32), 800.0, dtype=np.float32)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, ppx=16.0, ppy=16.0, depth_scale=0.001)
    with pytest.raises(ValueError, match="candidate_top_k"):
        select_grasp_patch(mask, depth, intr, patch_radius_px=4, candidate_top_k=value)


@pytest.mark.parametrize("name", ["exhaustive_fallback", "shadow_verify"])
def test_grasp_search_switches_require_booleans(name):
    mask = np.ones((32, 32), dtype=bool)
    depth = np.full((32, 32), 800.0, dtype=np.float32)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, ppx=16.0, ppy=16.0, depth_scale=0.001)
    with pytest.raises(ValueError, match=name):
        select_grasp_patch(mask, depth, intr, patch_radius_px=4, **{name: "false"})


@pytest.mark.parametrize("seed", [2, 7, 19])
def test_vectorized_selector_matches_scalar_reference_on_noisy_depth(seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:96, :96]
    mask = ((xx - 48) ** 2 / 31**2 + (yy - 47) ** 2 / 27**2) <= 1
    depth = (780.0 + 0.12 * (xx - 48) + 0.07 * (yy - 47) + rng.normal(0.0, 0.8, (96, 96))).astype(np.float32)
    depth[~mask] = np.nan
    depth[mask & (rng.random((96, 96)) < 0.04)] = np.nan
    intr = CameraIntrinsics(fx=420.0, fy=425.0, ppx=48.0, ppy=47.0, depth_scale=0.001)

    expected = _scalar_reference(mask, depth, intr, patch_radius_px=5, stride_px=4)
    actual = select_grasp_patch(mask, depth, intr, patch_radius_px=5, stride_px=4)

    assert expected is not None and actual is not None
    assert (actual.u_px, actual.v_px) == (expected.u_px, expected.v_px)
    assert actual.z_m == pytest.approx(expected.z_m, abs=1e-7)
    np.testing.assert_allclose(actual.normal_xyz, expected.normal_xyz, atol=3e-5)
    assert actual.score == pytest.approx(expected.score, abs=3e-5)
    assert actual.valid_depth_count == expected.valid_depth_count


def _scalar_reference(mask, depth, intr, *, patch_radius_px, stride_px):
    component = _largest_component(mask)
    component_area = int(component.sum())
    distance = _chamfer_distance_to_background(component)
    ys, xs = np.nonzero(component)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    centroid_y, centroid_x = float(ys.mean()), float(xs.mean())
    max_centroid_distance = float(np.hypot(y1 - y0, x1 - x0))
    rr = patch_radius_px
    oy, ox = np.ogrid[-rr : rr + 1, -rr : rr + 1]
    circle = (oy * oy + ox * ox) <= rr * rr
    circle_area = int(circle.sum())
    candidates = []
    for cy in range(y0, y1 + 1, stride_px):
        for cx in range(x0, x1 + 1, stride_px):
            if not component[cy, cx] or distance[cy, cx] < rr:
                continue
            py0, py1, px0, px1 = cy - rr, cy + rr + 1, cx - rr, cx + rr + 1
            if py0 < 0 or px0 < 0 or py1 > mask.shape[0] or px1 > mask.shape[1]:
                continue
            patch_mask = circle & component[py0:py1, px0:px1]
            patch_area = int(patch_mask.sum())
            if patch_area < max(20, circle_area // 3):
                continue
            local_depth = depth[py0:py1, px0:px1]
            valid = patch_mask & np.isfinite(local_depth) & (local_depth > 0.0)
            valid_count = int(valid.sum())
            valid_ratio = valid_count / max(patch_area, 1)
            if valid_count < 20 or valid_ratio < 0.7:
                continue
            vy, vx = np.nonzero(valid)
            depths_mm = local_depth[valid]
            points = _backproject_pixels(vx + px0, vy + py0, depths_mm, intr)
            normal, plane_rmse = _plane_stats(points)
            if plane_rmse > 0.006:
                continue
            depths_m = depths_mm.astype(np.float32) * np.float32(intr.depth_scale)
            z_m = float(np.median(depths_m))
            depth_mad = float(np.median(np.abs(depths_m - z_m)))
            if depth_mad > 0.012:
                continue
            boundary = float(distance[cy, cx])
            centroid = float(np.hypot(cy - centroid_y, cx - centroid_x))
            score = _candidate_score(
                valid_ratio,
                boundary,
                rr,
                centroid,
                max_centroid_distance,
                plane_rmse,
                0.006,
                depth_mad,
                0.012,
                valid_count,
                patch_area,
            )
            candidates.append(
                GraspPatch(
                    float(cx),
                    float(cy),
                    z_m,
                    normal,
                    score,
                    valid_ratio,
                    valid_count,
                    plane_rmse,
                    depth_mad,
                    boundary,
                    component_area,
                )
            )
    return max(candidates, key=lambda candidate: candidate.score) if candidates else None


@pytest.mark.parametrize("depth_scale", [0.0, -0.001])
def test_select_grasp_patch_rejects_non_positive_depth_scale(depth_scale):
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    depth = np.full((32, 32), np.nan, dtype=np.float32)
    depth[mask] = 800.0
    intr = CameraIntrinsics(fx=100.0, fy=100.0, ppx=16.0, ppy=16.0, depth_scale=depth_scale)

    with pytest.raises(ValueError, match="depth_scale"):
        select_grasp_patch(mask, depth, intr, patch_radius_px=4, stride_px=4)


@pytest.mark.parametrize(("field", "value"), [("ppx", np.nan), ("ppy", np.inf)])
def test_select_grasp_patch_rejects_non_finite_principal_point(field, value):
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    depth = np.full((32, 32), np.nan, dtype=np.float32)
    depth[mask] = 800.0
    kwargs = dict(fx=100.0, fy=100.0, ppx=16.0, ppy=16.0, depth_scale=0.001)
    kwargs[field] = value
    intr = CameraIntrinsics(**kwargs)

    with pytest.raises(ValueError, match=field):
        select_grasp_patch(mask, depth, intr, patch_radius_px=4, stride_px=4)


def test_grasp_patch_equality_is_identity_based_with_array_normals():
    normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    patch_a = GraspPatch(
        u_px=16.0,
        v_px=16.0,
        z_m=0.8,
        normal_xyz=normal,
        score=1.0,
        valid_depth_ratio=1.0,
        valid_depth_count=32,
        plane_rmse_m=0.0,
        depth_mad_m=0.0,
        boundary_distance_px=8.0,
        component_area_px=256,
    )
    patch_b = GraspPatch(
        u_px=16.0,
        v_px=16.0,
        z_m=0.8,
        normal_xyz=normal.copy(),
        score=1.0,
        valid_depth_ratio=1.0,
        valid_depth_count=32,
        plane_rmse_m=0.0,
        depth_mad_m=0.0,
        boundary_distance_px=8.0,
        component_area_px=256,
    )

    assert patch_a == patch_a
    assert patch_a != patch_b


def test_approach_axis_is_negative_normal_for_suction():
    normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    approach = approach_from_normal(normal)
    np.testing.assert_allclose(approach, np.array([-0.0, -0.0, 1.0], dtype=np.float32))


def test_approach_from_normal_rejects_near_zero_vector():
    with pytest.raises(ValueError, match="near-zero"):
        approach_from_normal(np.array([1e-12, 0.0, 0.0], dtype=np.float32))
