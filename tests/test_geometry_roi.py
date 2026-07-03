import numpy as np
import pytest

from visual_deploy.geometry.backproject import backproject_pixel
from visual_deploy.geometry.roi import RoiTransform, square_pad_bbox
from visual_deploy.types import CameraIntrinsics


def test_square_pad_bbox_uses_larger_side_with_margin():
    assert square_pad_bbox((10, 20, 50, 40), 0.2, 200, 100) == (6, 6, 54, 54)


def test_square_pad_bbox_rounds_all_edges_before_clipping():
    assert square_pad_bbox((10, 20, 51, 40), 0.2, 200, 100) == (6, 5, 55, 55)


def test_roi_round_trip_center_bbox():
    roi = RoiTransform.from_bbox((100, 50, 300, 250), 0.2, 640, 480, 256)
    u_roi, v_roi = roi.image_to_roi(200.0, 150.0)
    u_img, v_img = roi.roi_to_image(u_roi, v_roi)
    assert u_img == pytest.approx(200.0)
    assert v_img == pytest.approx(150.0)


def test_edge_clipped_bbox_keeps_independent_scales():
    roi = RoiTransform.from_bbox((0, 0, 120, 40), 0.2, 160, 80, 256)
    assert roi.crop_xyxy[0] == 0
    assert roi.crop_xyxy[1] == 0
    assert roi.scale_x != roi.scale_y


def test_roi_intrinsics_match_crop_then_resize():
    intr = CameraIntrinsics(fx=600.0, fy=610.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    roi = RoiTransform(crop_xyxy=(100, 50, 300, 250), roi_size=256)
    cam_roi = roi.adjust_intrinsics(intr)
    assert cam_roi.fx == pytest.approx(600.0 * roi.scale_x)
    assert cam_roi.fy == pytest.approx(610.0 * roi.scale_y)
    assert cam_roi.ppx == pytest.approx((320.0 - 100.0) * roi.scale_x)
    assert cam_roi.ppy == pytest.approx((240.0 - 50.0) * roi.scale_y)
    assert cam_roi.depth_scale == pytest.approx(0.001)


def test_roi_and_full_backprojection_are_consistent():
    intr = CameraIntrinsics(fx=600.0, fy=610.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    roi = RoiTransform(crop_xyxy=(100, 50, 300, 250), roi_size=256)
    cam_roi = roi.adjust_intrinsics(intr)
    u_img, v_img, depth_mm = 210.0, 170.0, 500.0
    u_roi, v_roi = roi.image_to_roi(u_img, v_img)
    xyz_full = backproject_pixel(u_img, v_img, depth_mm, intr)
    xyz_roi = backproject_pixel(u_roi, v_roi, depth_mm, cam_roi)
    np.testing.assert_allclose(xyz_roi, xyz_full, atol=1e-6)
