import numpy as np
import pytest

from visual_deploy.tracking.tracker import ConfidenceGate, DepthRoiStats, DualThresholdIoUTracker, SimpleIoUTracker
from visual_deploy.types import Detection


def test_confidence_gate_drops_low_and_weights_mid_confidence():
    gate = ConfidenceGate(high_threshold=0.7, low_threshold=0.4)
    kept = gate.apply(
        [
            Detection((0, 0, 10, 10), 0, 0.3),
            Detection((0, 0, 10, 10), 0, 0.55),
            Detection((0, 0, 10, 10), 0, 0.8),
        ]
    )
    assert len(kept) == 2
    assert kept[0].weighted_confidence < kept[0].confidence
    assert kept[1].weighted_confidence == kept[1].confidence


def test_iou_tracker_confirms_after_min_hits():
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost=3, min_hits=2)
    first = tracker.update([Detection((10, 10, 50, 50), 0, 0.9)])
    second = tracker.update([Detection((12, 12, 52, 52), 0, 0.8)])
    assert first[0].state == "tentative"
    assert second[0].state == "confirmed"
    assert second[0].track_id == first[0].track_id


def test_depth_roi_stats_uses_mm_thresholds():
    depth = np.array([[0, 200, 300], [6000, 400, 500]], dtype=np.float32)
    stats = DepthRoiStats(min_depth_mm=100, max_depth_mm=5000).extract(depth, (0, 0, 3, 2))
    assert stats.valid_count == 4
    assert stats.median_mm == 350.0
    assert stats.state == "valid"


def test_iou_tracker_keeps_classes_separate_and_removes_lost_tracks():
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost=1, min_hits=1)
    first = tracker.update([Detection((10, 10, 50, 50), 0, 0.9)])
    second = tracker.update([Detection((12, 12, 52, 52), 1, 0.8)])
    third = tracker.update([Detection((12, 12, 52, 52), 1, 0.7)])

    assert second[0].track_id != first[0].track_id
    assert third[0].track_id == second[0].track_id


def test_depth_roi_stats_marks_sparse_valid_pixels_partial():
    depth = np.array([[0, 0, 300], [0, 0, 500]], dtype=np.float32)
    stats = DepthRoiStats(min_depth_mm=100, max_depth_mm=5000).extract(depth, (0, 0, 3, 2))
    assert stats.valid_count == 2
    assert stats.valid_ratio == 2 / 6
    assert stats.state == "partial"


def test_iou_tracker_counts_one_lost_update_per_missed_frame():
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost=2, min_hits=1)
    track_id = tracker.update([Detection((10, 10, 50, 50), 0, 0.9)])[0].track_id

    tracker.update([])
    assert tracker._tracks[track_id].lost == 1

    tracker.update([])
    assert tracker._tracks[track_id].lost == 2

    tracker.update([])
    assert track_id not in tracker._tracks


def test_iou_tracker_honors_zero_weighted_confidence_at_low_threshold():
    gate = ConfidenceGate(high_threshold=0.7, low_threshold=0.4)
    kept = gate.apply([Detection((0, 0, 10, 10), 0, 0.4)])

    tracks = SimpleIoUTracker(min_hits=1).update(kept)

    assert kept[0].weighted_confidence == 0.0
    assert tracks[0].confidence == 0.0


def test_depth_roi_stats_invalid_channel_shapes_fail_closed():
    empty_channel = np.empty((2, 3, 0), dtype=np.float32)
    higher_rank = np.ones((1, 2, 3, 1), dtype=np.float32) * 300

    extractor = DepthRoiStats()

    assert extractor.extract(empty_channel, (0, 0, 3, 2)).state == "invalid"
    assert extractor.extract(higher_rank, (0, 0, 3, 2)).state == "invalid"


def test_dual_threshold_low_detection_cannot_create_track():
    tracker = DualThresholdIoUTracker(high_threshold=0.5, low_threshold=0.1, min_hits=1)
    assert tracker.update([Detection((0, 0, 10, 10), 0, 0.4)]) == []


def test_dual_threshold_low_detection_continues_high_created_track():
    tracker = DualThresholdIoUTracker(high_threshold=0.5, low_threshold=0.1, min_hits=1)
    created = tracker.update([Detection((0, 0, 10, 10), 0, 0.8)])[0]
    continued = tracker.update([Detection((1, 0, 11, 10), 0, 0.2)])[0]
    assert continued.track_id == created.track_id
    assert continued.confidence == 0.2


def test_dual_threshold_high_association_has_priority():
    tracker = DualThresholdIoUTracker(high_threshold=0.5, low_threshold=0.1, min_hits=1)
    track_id = tracker.update([Detection((0, 0, 10, 10), 0, 0.9)])[0].track_id
    outputs = tracker.update([Detection((2, 0, 12, 10), 0, 0.7), Detection((0, 0, 10, 10), 0, 0.2)])
    assert len(outputs) == 1
    assert outputs[0].track_id == track_id
    assert outputs[0].confidence == 0.7


def test_dual_threshold_rejects_reversed_thresholds():
    with pytest.raises(ValueError, match="low_threshold"):
        DualThresholdIoUTracker(high_threshold=0.1, low_threshold=0.5)


def test_dual_threshold_tracker_exposes_only_bounded_confirmed_coast():
    tracker = DualThresholdIoUTracker(high_threshold=0.5, low_threshold=0.1, max_lost=5, min_hits=1)
    track_id = tracker.update([Detection((0, 0, 10, 10), 0, 0.9)])[0].track_id

    tracker.update([])
    first = tracker.get_coasting_track(track_id, max_coast_frames=2)
    tracker.update([])
    second = tracker.get_coasting_track(track_id, max_coast_frames=2)
    tracker.update([])

    assert first is not None and first.state == "coasting" and first.lost == 1
    assert second is not None and second.lost == 2
    assert tracker.get_coasting_track(track_id, max_coast_frames=2) is None
