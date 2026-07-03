import numpy as np
import pytest

from visual_deploy.geometry.roi import RoiTransform
from visual_deploy.tracking.depth_fusion import DepthFusionBuffer


def test_depth_fusion_returns_current_frame_until_enough_history():
    buffer = DepthFusionBuffer(window_size=3, min_depth_mm=100, max_depth_mm=5000, min_valid_ratio=0.3)
    depth = np.full((4, 4), 500.0, dtype=np.float32)
    fused = buffer.update(
        track_id=1,
        frame_id=1,
        timestamp_ms=0.0,
        roi_transform=RoiTransform((0, 0, 4, 4), 4),
        depth_roi_mm=depth,
    )
    assert fused.source_frame_count == 1
    np.testing.assert_allclose(fused.depth_roi_mm, depth)


def test_depth_fusion_uses_per_pixel_median():
    buffer = DepthFusionBuffer(window_size=3, min_depth_mm=100, max_depth_mm=5000, min_valid_ratio=0.3)
    roi = RoiTransform((0, 0, 2, 2), 2)
    for idx, value in enumerate([400.0, 500.0, 1000.0], 1):
        out = buffer.update(1, idx, float(idx), roi, np.full((2, 2), value, dtype=np.float32))
    assert out.source_frame_count == 3
    np.testing.assert_allclose(out.depth_roi_mm, np.full((2, 2), 500.0, dtype=np.float32))


def test_depth_fusion_ignores_invalid_values():
    buffer = DepthFusionBuffer(window_size=3, min_depth_mm=100, max_depth_mm=5000, min_valid_ratio=0.3)
    roi = RoiTransform((0, 0, 2, 2), 2)
    buffer.update(1, 1, 1.0, roi, np.array([[0, 500], [6000, 500]], dtype=np.float32))
    out = buffer.update(1, 2, 2.0, roi, np.array([[400, 700], [800, 900]], dtype=np.float32))
    assert out.valid_ratio >= 0.5
    assert np.isfinite(out.depth_roi_mm).all()


def test_depth_fusion_requires_positive_window_size():
    with pytest.raises(ValueError, match="window_size"):
        DepthFusionBuffer(window_size=0)


def test_depth_fusion_rejects_fractional_window_size():
    with pytest.raises(ValueError, match="window_size"):
        DepthFusionBuffer(window_size=0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_depth_mm": 0.0}, "min_depth_mm"),
        ({"min_depth_mm": np.inf}, "min_depth_mm"),
        ({"max_depth_mm": 0.0}, "max_depth_mm"),
        ({"max_depth_mm": np.nan}, "max_depth_mm"),
        ({"min_depth_mm": 5000.0, "max_depth_mm": 100.0}, "min_depth_mm"),
        ({"min_valid_ratio": -0.1}, "min_valid_ratio"),
        ({"min_valid_ratio": 1.1}, "min_valid_ratio"),
    ],
)
def test_depth_fusion_rejects_invalid_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DepthFusionBuffer(**kwargs)


def test_depth_fusion_requires_2d_depth_roi():
    buffer = DepthFusionBuffer()
    roi = RoiTransform((0, 0, 2, 2), 2)

    with pytest.raises(ValueError, match="2D"):
        buffer.update(1, 1, 1.0, roi, np.ones((2, 2, 1), dtype=np.float32))


def test_depth_fusion_requires_depth_shape_to_match_roi_size():
    buffer = DepthFusionBuffer()
    roi = RoiTransform((0, 0, 4, 4), 4)

    with pytest.raises(ValueError, match="roi_size"):
        buffer.update(1, 1, 1.0, roi, np.ones((2, 2), dtype=np.float32))


def test_depth_fusion_rejects_shape_mismatch_for_same_track():
    buffer = DepthFusionBuffer()
    roi = RoiTransform((0, 0, 2, 2), 2)
    buffer.update(1, 1, 1.0, roi, np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        buffer.update(1, 2, 2.0, roi, np.ones((3, 3), dtype=np.float32))


def test_depth_fusion_resets_track_history_when_crop_changes():
    buffer = DepthFusionBuffer(window_size=3)
    first_roi = RoiTransform((0, 0, 2, 2), 2)
    second_roi = RoiTransform((1, 1, 3, 3), 2)
    buffer.update(1, 1, 1.0, first_roi, np.ones((2, 2), dtype=np.float32) * 400)

    out = buffer.update(1, 2, 2.0, second_roi, np.ones((2, 2), dtype=np.float32) * 800)

    assert out.source_frame_count == 1
    assert out.age_ms == 0.0
    np.testing.assert_allclose(out.depth_roi_mm, np.ones((2, 2), dtype=np.float32) * 800)


def test_depth_fusion_fails_closed_when_valid_ratio_is_too_low():
    buffer = DepthFusionBuffer(window_size=1, min_valid_ratio=0.75)
    roi = RoiTransform((0, 0, 2, 2), 2)

    out = buffer.update(1, 1, 1.0, roi, np.array([[400, 0], [0, 0]], dtype=np.float32))

    assert out.valid_ratio == 0.0
    np.testing.assert_allclose(out.depth_roi_mm, np.zeros((2, 2), dtype=np.float32))


def test_depth_fusion_output_equality_is_identity_based():
    buffer = DepthFusionBuffer()
    roi = RoiTransform((0, 0, 2, 2), 2)

    first = buffer.update(1, 1, 1.0, roi, np.ones((2, 2), dtype=np.float32) * 400)
    second = buffer.update(1, 2, 2.0, roi, np.ones((2, 2), dtype=np.float32) * 400)

    assert first == first
    assert (first == second) is False


def test_depth_fusion_rejects_timestamp_going_backwards_for_same_track():
    buffer = DepthFusionBuffer()
    roi = RoiTransform((0, 0, 2, 2), 2)
    buffer.update(1, 1, 10.0, roi, np.ones((2, 2), dtype=np.float32) * 400)

    with pytest.raises(ValueError, match="timestamp"):
        buffer.update(1, 2, 9.0, roi, np.ones((2, 2), dtype=np.float32) * 500)


@pytest.mark.parametrize("timestamp_ms", [np.nan, np.inf, -np.inf])
def test_depth_fusion_rejects_non_finite_timestamp(timestamp_ms):
    buffer = DepthFusionBuffer()
    roi = RoiTransform((0, 0, 2, 2), 2)

    with pytest.raises(ValueError, match="timestamp_ms"):
        buffer.update(1, 1, timestamp_ms, roi, np.ones((2, 2), dtype=np.float32) * 400)


def test_depth_fusion_history_does_not_share_input_array_mutation():
    buffer = DepthFusionBuffer(window_size=2)
    roi = RoiTransform((0, 0, 2, 2), 2)
    depth = np.ones((2, 2), dtype=np.float32) * 400
    buffer.update(1, 1, 1.0, roi, depth)
    depth[:, :] = 1000

    out = buffer.update(1, 2, 2.0, roi, np.ones((2, 2), dtype=np.float32) * 600)

    np.testing.assert_allclose(out.depth_roi_mm, np.ones((2, 2), dtype=np.float32) * 500)


def test_depth_fusion_clear_removes_one_track_or_all_tracks():
    buffer = DepthFusionBuffer()
    roi = RoiTransform((0, 0, 2, 2), 2)
    buffer.update(1, 1, 1.0, roi, np.ones((2, 2), dtype=np.float32) * 400)
    buffer.update(2, 1, 1.0, roi, np.ones((2, 2), dtype=np.float32) * 800)

    buffer.clear(track_id=1)
    out = buffer.update(1, 2, 2.0, roi, np.ones((2, 2), dtype=np.float32) * 1200)
    assert out.source_frame_count == 1

    buffer.clear()
    out = buffer.update(2, 2, 2.0, roi, np.ones((2, 2), dtype=np.float32) * 1600)
    assert out.source_frame_count == 1
