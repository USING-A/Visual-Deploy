import numpy as np
import pytest

from visual_deploy.geometry.grasp_patch import GraspPatch, select_grasp_patch
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
