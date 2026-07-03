import numpy as np

from visual_deploy.debug.overlay import DebugOverlayOptions, depth_to_bgr, render_debug_overlay
from visual_deploy.types import Detection, GraspTarget


def test_render_debug_overlay_draws_sparse_bbox_mask_and_target():
    color = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=bool)
    mask[40:80, 60:100] = True
    target = GraspTarget(valid=True, frame_id=1, track_id=7, u_px=80.0, v_px=60.0, z_mm=500.0, target_score=0.8)
    detection = Detection(bbox_xyxy=(50.0, 30.0, 110.0, 90.0), class_id=0, confidence=0.91, track_id=7)

    overlay = render_debug_overlay(
        color,
        detections=[detection],
        target=target,
        masks=[mask],
        options=DebugOverlayOptions(show_labels=False, show_status=False),
    )

    assert overlay.shape == color.shape
    assert overlay.dtype == np.uint8
    assert np.count_nonzero(overlay) > 0
    assert overlay[60, 80].max() > 0
    assert np.count_nonzero(overlay != color) < color.size // 4


def test_depth_to_bgr_ignores_zero_and_nan_depth():
    depth = np.array(
        [
            [0.0, np.nan, 100.0],
            [200.0, 300.0, 5000.0],
        ],
        dtype=np.float32,
    )

    preview = depth_to_bgr(depth, min_mm=100.0, max_mm=300.0)

    assert preview.shape == (2, 3, 3)
    assert preview.dtype == np.uint8
    assert np.all(preview[0, 0] == 0)
    assert np.all(preview[0, 1] == 0)
    assert preview[1, 1].max() > 0
