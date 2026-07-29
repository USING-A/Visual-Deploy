import json

import numpy as np
import pytest

from visual_deploy.detection.yolo_detector import MockDetector
from visual_deploy.pipeline.offline_pipeline import OfflinePipeline
from visual_deploy.segmentation.gcnet_segmentor import SegmentResult
from scripts.run_offline_smoke import _build_detector, _build_segmentor
from visual_deploy.types import CameraIntrinsics, DeployFrame, Detection
from visual_deploy.observability.runtime_telemetry import FrameRuntimeTelemetry, attach_runtime_telemetry


class MockSegmentor:
    def __init__(self) -> None:
        self.color_shape = None
        self.depth_shape = None

    def infer(self, color_bgr_256, depth_mm_256):
        self.color_shape = color_bgr_256.shape
        self.depth_shape = depth_mm_256.shape
        mask = np.zeros((256, 256), dtype=bool)
        yy, xx = np.ogrid[:256, :256]
        mask[(yy - 128) ** 2 + (xx - 128) ** 2 <= 60 ** 2] = True
        return SegmentResult(mask.astype(np.float32), mask, int(mask.sum()), 1.0)


class TwoAppleDetector:
    def infer(self, color_bgr):
        return [
            Detection((120.0, 120.0, 280.0, 360.0), 0, 0.9, "apple"),
            Detection((360.0, 120.0, 520.0, 360.0), 0, 0.9, "apple"),
        ]


class CountingSegmentor(MockSegmentor):
    def __init__(self, empty_on_calls=()) -> None:
        super().__init__()
        self.calls = 0
        self.empty_on_calls = set(empty_on_calls)

    def infer(self, color_bgr_256, depth_mm_256):
        self.calls += 1
        if self.calls in self.empty_on_calls:
            empty = np.zeros((256, 256), dtype=bool)
            return SegmentResult(empty.astype(np.float32), empty, 0, 0.0)
        return super().infer(color_bgr_256, depth_mm_256)


def _frame(frame_id: int = 1, timestamp_ms: float = 1.0, depth_value_mm: float = 500.0) -> DeployFrame:
    return DeployFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        color_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_mm=np.full((480, 640), depth_value_mm, dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001),
    )


def test_offline_pipeline_emits_target_with_mock_detector_segmentor(tmp_path):
    segmentor = MockSegmentor()
    pipeline = OfflinePipeline(
        detector=MockDetector(0.9),
        segmentor=segmentor,
        config={"recording": {"output_root": str(tmp_path)}},
    )

    target = pipeline.process_frame(_frame())

    assert target.valid is True
    assert target.track_id == 1
    assert target.u_px == 320.0
    assert target.v_px == 240.0
    assert target.z_mm == 500.0
    assert target.xyz_camera_m is not None
    np.testing.assert_allclose(target.xyz_camera_m, (0.0, 0.0, 0.5), atol=1e-6)
    assert target.approach_axis == (-0.0, -0.0, 1.0)
    assert segmentor.color_shape == (256, 256, 3)
    assert segmentor.depth_shape == (256, 256)


def test_offline_pipeline_returns_invalid_when_no_candidate(tmp_path):
    class EmptyDetector:
        def infer(self, color_bgr):
            return []

    pipeline = OfflinePipeline(
        detector=EmptyDetector(),
        segmentor=MockSegmentor(),
        config={
            "profiling": {"enabled": True},
            "recording": {"enabled": True, "save_timings": True, "output_root": str(tmp_path)},
        },
    )

    target = pipeline.process_frame(_frame())

    assert target.valid is False
    assert target.reason == "no_valid_grasp_candidate"
    run_dir = next(tmp_path.iterdir())
    timing = json.loads((run_dir / "timings.jsonl").read_text(encoding="utf-8").strip())
    assert timing["depth_fusion_ms"] == 0.0
    assert timing["segmentation_ms"] == 0.0
    assert timing["grasp_ms"] == 0.0


def test_offline_pipeline_records_ranked_candidate_score_after_ranking(tmp_path):
    pipeline = OfflinePipeline(
        detector=MockDetector(0.9),
        segmentor=MockSegmentor(),
        config={
            "profiling": {"enabled": True},
            "grasp": {"candidate_top_k": 256},
            "recording": {"enabled": True, "save_candidates": True, "save_timings": True, "output_root": str(tmp_path)},
        },
    )

    target = pipeline.process_frame(_frame())

    run_dir = next(tmp_path.iterdir())
    record = json.loads((run_dir / "candidates.jsonl").read_text(encoding="utf-8").strip())
    candidate = record["candidates"][0]
    assert target.valid is True
    assert candidate["scores"]["target_score"] == pytest.approx(target.target_score)
    assert candidate["rank"] == 1
    assert candidate["selected"] is True
    assert candidate["safety"]["valid"] is True
    assert candidate["grasp"]["search"]["mode"] == "topk"
    assert candidate["grasp"]["search"]["eligible_candidate_count"] > 256
    assert candidate["grasp"]["search"]["fine_candidate_count"] == 256
    timing = json.loads((run_dir / "timings.jsonl").read_text(encoding="utf-8").strip())
    assert timing["valid_target"] is True
    assert timing["detection_ms"] >= 0.0
    assert timing["segmentation_ms"] >= 0.0
    assert timing["grasp_ms"] >= 0.0
    assert timing["total_ms"] >= timing["detection_ms"]
    assert timing["workload"] == {
        "raw_detections": 1,
        "detector_output_candidates": 1,
        "detector_finite_confidence_candidates": 1,
        "detector_above_threshold_candidates": 1,
        "detector_max_confidence": 0.9,
        "eligible_detections": 1,
        "tracks": 1,
        "confirmed_tracks": 1,
        "coasting_tracks": 0,
        "depth_fused_tracks": 1,
        "depth_history_frames": 1,
        "depth_remap_calls": 0,
        "segmentation_calls": 1,
        "segmented_tracks": 1,
        "grasp_searches": 1,
        "grasp_candidates": 1,
        "pipeline_candidates": 1,
        "rejections": 0,
        "active_target_fast_path": 0,
        "active_target_fallback": 0,
        "active_target_refresh": 0,
        "active_target_coast": 0,
        "deferred_tracks": 0,
    }


def test_active_target_coasts_for_two_zero_detection_frames_then_fails_closed(tmp_path):
    class DropoutDetector:
        def __init__(self):
            self.calls = 0

        def infer(self, color_bgr):
            self.calls += 1
            if self.calls in {2, 3, 4}:
                return []
            return [Detection((160.0, 120.0, 480.0, 360.0), 0, 0.9, "apple")]

    segmentor = CountingSegmentor()
    pipeline = OfflinePipeline(
        DropoutDetector(),
        segmentor,
        config={
            "runtime": {
                "active_target": {
                    "enabled": True,
                    "refresh_interval_frames": 30,
                    "max_coast_frames": 2,
                }
            },
            "profiling": {"enabled": True},
            "recording": {"enabled": True, "save_timings": True, "output_root": str(tmp_path)},
        },
    )

    targets = [pipeline.process_frame(_frame(frame_id=i, timestamp_ms=i * 33.0)) for i in range(1, 6)]
    records = [json.loads(line) for line in (pipeline.recorder.run_dir / "timings.jsonl").read_text().splitlines()]

    assert [target.valid for target in targets] == [True, True, True, False, True]
    assert targets[0].track_id == targets[1].track_id == targets[2].track_id == targets[4].track_id
    assert [record["workload"]["active_target_coast"] for record in records] == [0, 1, 1, 0, 0]
    assert [record["workload"]["coasting_tracks"] for record in records] == [0, 1, 1, 0, 0]
    assert segmentor.calls == 4


def test_active_target_coast_stops_after_current_frame_downstream_failure(tmp_path):
    class OneThenEmptyDetector:
        def __init__(self):
            self.calls = 0

        def infer(self, color_bgr):
            self.calls += 1
            if self.calls == 1:
                return [Detection((160.0, 120.0, 480.0, 360.0), 0, 0.9, "apple")]
            return []

    segmentor = CountingSegmentor(empty_on_calls={2})
    pipeline = OfflinePipeline(
        OneThenEmptyDetector(),
        segmentor,
        config={"runtime": {"active_target": {"enabled": True, "max_coast_frames": 2}}},
    )

    first = pipeline.process_frame(_frame(frame_id=1, timestamp_ms=33.0))
    failed_coast = pipeline.process_frame(_frame(frame_id=2, timestamp_ms=66.0))
    no_second_coast = pipeline.process_frame(_frame(frame_id=3, timestamp_ms=99.0))

    assert first.valid is True
    assert failed_coast.valid is False
    assert no_second_coast.valid is False
    assert segmentor.calls == 2


def test_active_target_fast_path_skips_deferred_track_and_periodically_refreshes(tmp_path):
    segmentor = CountingSegmentor()
    pipeline = OfflinePipeline(
        TwoAppleDetector(),
        segmentor,
        config={
            "runtime": {"active_target": {"enabled": True, "refresh_interval_frames": 3}},
            "profiling": {"enabled": True},
            "recording": {
                "enabled": True,
                "save_timings": True,
                "save_candidates": True,
                "output_root": str(tmp_path),
            },
        },
    )

    first = pipeline.process_frame(_frame(frame_id=1, timestamp_ms=0.0))
    second = pipeline.process_frame(_frame(frame_id=2, timestamp_ms=33.0))
    third = pipeline.process_frame(_frame(frame_id=3, timestamp_ms=66.0))

    records = [json.loads(line) for line in (pipeline.recorder.run_dir / "timings.jsonl").read_text().splitlines()]
    candidate_records = [
        json.loads(line) for line in (pipeline.recorder.run_dir / "candidates.jsonl").read_text().splitlines()
    ]
    assert first.valid and second.valid and third.valid
    assert first.track_id == second.track_id
    assert segmentor.calls == 5
    assert records[0]["workload"]["segmentation_calls"] == 2
    assert records[1]["workload"]["segmentation_calls"] == 1
    assert records[1]["workload"]["active_target_fast_path"] == 1
    assert records[1]["workload"]["deferred_tracks"] == 1
    assert records[2]["workload"]["segmentation_calls"] == 2
    assert records[2]["workload"]["active_target_refresh"] == 1
    refresh_history = {
        candidate["track_id"]: candidate["depth_fusion"]["source_frame_count"]
        for candidate in candidate_records[2]["candidates"]
    }
    assert refresh_history[first.track_id] == 3
    assert refresh_history[next(track_id for track_id in refresh_history if track_id != first.track_id)] == 1


def test_active_target_failure_runs_full_fallback_in_same_frame(tmp_path):
    segmentor = CountingSegmentor(empty_on_calls={3})
    pipeline = OfflinePipeline(
        TwoAppleDetector(),
        segmentor,
        config={
            "runtime": {"active_target": {"enabled": True, "refresh_interval_frames": 30}},
            "profiling": {"enabled": True},
            "recording": {"enabled": True, "save_timings": True, "output_root": str(tmp_path)},
        },
    )

    first = pipeline.process_frame(_frame(frame_id=1, timestamp_ms=0.0))
    second = pipeline.process_frame(_frame(frame_id=2, timestamp_ms=33.0))

    records = [json.loads(line) for line in (pipeline.recorder.run_dir / "timings.jsonl").read_text().splitlines()]
    assert first.valid is True
    assert second.valid is True
    assert second.track_id != first.track_id
    assert segmentor.calls == 4
    assert records[1]["workload"]["active_target_fast_path"] == 1
    assert records[1]["workload"]["active_target_fallback"] == 1
    assert records[1]["workload"]["segmentation_calls"] == 2


def test_offline_pipeline_rejects_candidate_that_fails_safety_gate(tmp_path):
    pipeline = OfflinePipeline(
        detector=MockDetector(0.9),
        segmentor=MockSegmentor(),
        config={
            "safety": {"enabled": True, "min_target_score": 1.0},
            "recording": {"enabled": True, "save_candidates": True, "output_root": str(tmp_path)},
        },
    )

    target = pipeline.process_frame(_frame())

    run_dir = next(tmp_path.iterdir())
    record = json.loads((run_dir / "candidates.jsonl").read_text(encoding="utf-8").strip())
    assert target.valid is False
    assert target.reason == "no_safe_grasp_candidate"
    assert record["candidates"][0]["selected"] is False
    assert record["candidates"][0]["safety"]["reason"] == "target_score_below_threshold"
    assert record["rejections"][0]["stage"] == "safety"


def test_offline_pipeline_saves_event_artifacts_for_safety_rejection(tmp_path):
    pipeline = OfflinePipeline(
        detector=MockDetector(0.9),
        segmentor=MockSegmentor(),
        config={
            "safety": {"enabled": True, "min_target_score": 1.0},
            "diagnostics": {"enabled": True},
            "recording": {
                "enabled": True,
                "output_root": str(tmp_path),
                "save_events": True,
                "save_event_artifacts": True,
                "event_stages": ["safety"],
                "save_frames": True,
                "save_depth": True,
                "save_masks": True,
                "save_overlays": True,
            },
        },
    )

    target = pipeline.process_frame(_frame())

    run_dir = next(tmp_path.iterdir())
    event = json.loads((run_dir / "events.jsonl").read_text(encoding="utf-8").strip())
    assert target.valid is False
    assert event["rejections"][0]["stage"] == "safety"
    assert (run_dir / event["artifacts"]["frame"]).is_file()
    assert (run_dir / event["artifacts"]["depth"]).is_file()
    assert (run_dir / event["artifacts"]["masks"][0]).is_file()
    assert (run_dir / event["artifacts"]["overlay"]).is_file()


def test_offline_pipeline_rejects_large_same_track_depth_jump(tmp_path):
    pipeline = OfflinePipeline(
        detector=MockDetector(0.9),
        segmentor=MockSegmentor(),
        config={
            "depth_fusion": {"window_size": 1},
            "safety": {
                "enabled": True,
                "max_depth_step_mm": 30.0,
                "max_pixel_step_px": 40.0,
                "max_history_age_ms": 1000.0,
            },
            "recording": {"enabled": True, "save_candidates": True, "output_root": str(tmp_path)},
        },
    )

    first = pipeline.process_frame(_frame(frame_id=1, timestamp_ms=0.0, depth_value_mm=500.0))
    second = pipeline.process_frame(_frame(frame_id=2, timestamp_ms=33.0, depth_value_mm=700.0))

    run_dir = next(tmp_path.iterdir())
    records = [json.loads(line) for line in (run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert first.valid is True
    assert second.valid is False
    assert second.reason == "no_safe_grasp_candidate"
    assert records[1]["candidates"][0]["safety"]["reason"] == "target_depth_jump"
    assert records[1]["rejections"][0]["stage"] == "continuity"


def test_offline_pipeline_keeps_gcnet_roi_size_fixed_at_256(tmp_path):
    segmentor = MockSegmentor()
    pipeline = OfflinePipeline(
        detector=MockDetector(0.9),
        segmentor=segmentor,
        config={"roi": {"size": 128}, "recording": {"output_root": str(tmp_path)}},
    )

    target = pipeline.process_frame(_frame())

    assert target.valid is True
    assert segmentor.color_shape == (256, 256, 3)
    assert segmentor.depth_shape == (256, 256)


def test_offline_smoke_requires_explicit_mock_models_for_missing_weights(tmp_path):
    config_path = tmp_path / "configs" / "deploy.yaml"
    config_path.parent.mkdir()
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="YOLOv10 model"):
        _build_detector(config_path, {"detection": {"weights": "missing_yolo.onnx"}}, use_mock_models=False)

    with pytest.raises(FileNotFoundError, match="GCNet model"):
        _build_segmentor(config_path, {"segmentation": {"weights": "missing_gcnet.onnx"}}, use_mock_models=False)

    assert isinstance(
        _build_detector(config_path, {"detection": {"weights": "missing_yolo.onnx"}}, use_mock_models=True),
        MockDetector,
    )
    assert hasattr(_build_segmentor(config_path, {"segmentation": {"weights": "missing_gcnet.onnx"}}, use_mock_models=True), "infer")


def test_pipeline_profiling_records_capture_and_future_queue_fields(tmp_path):
    frame = _frame()
    attach_runtime_telemetry(
        frame,
        FrameRuntimeTelemetry(
            capture_ms=3.0,
            queue_wait_ms={"capture_to_inference": 1.5},
            queue_depth={"capture_to_inference": 1},
            queue_capacity={"capture_to_inference": 2},
            dropped_frames={"capture": 4},
            resources={"gpu_util_percent": 50.0},
        ),
    )
    pipeline = OfflinePipeline(
        MockDetector(0.9),
        MockSegmentor(),
        config={
            "profiling": {"enabled": True},
            "recording": {"enabled": True, "save_timings": True, "output_root": str(tmp_path)},
        },
    )
    pipeline.process_frame(frame)
    timing = json.loads((pipeline.recorder.run_dir / "timings.jsonl").read_text(encoding="utf-8"))
    assert timing["capture_ms"] == 3.0
    assert timing["queue_wait_ms"] == {"capture_to_inference": 1.5}
    assert timing["queue_depth"] == {"capture_to_inference": 1}
    assert timing["queue_capacity"] == {"capture_to_inference": 2}
    assert timing["dropped_frames"] == {"capture": 4}
    assert timing["resources"] == {"gpu_util_percent": 50.0}
    assert timing["completed_monotonic_ms"] > 0.0


def test_pipeline_debug_and_recording_are_off_by_default(tmp_path):
    pipeline = OfflinePipeline(MockDetector(0.9), MockSegmentor(), config={"recording": {"output_root": str(tmp_path)}})
    pipeline.process_frame(_frame())
    assert pipeline.last_debug is None
    assert not pipeline.recorder.run_dir.exists()


def test_pipeline_close_is_idempotent_and_releases_models(tmp_path):
    class CloseableDetector(MockDetector):
        def __init__(self):
            super().__init__(0.9)
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class CloseableSegmentor(MockSegmentor):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    detector = CloseableDetector()
    segmentor = CloseableSegmentor()
    pipeline = OfflinePipeline(detector, segmentor, config={"recording": {"output_root": str(tmp_path)}})

    pipeline.close()
    pipeline.close()

    assert detector.close_calls == 1
    assert segmentor.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.process_frame(_frame())


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_pipeline_rejects_invalid_grasp_shadow_interval(value):
    with pytest.raises(ValueError, match="shadow_verify_every_n_frames"):
        OfflinePipeline(
            MockDetector(0.9),
            MockSegmentor(),
            config={"grasp": {"shadow_verify_every_n_frames": value}},
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_pipeline_rejects_invalid_active_target_refresh_interval(value):
    with pytest.raises(ValueError, match="refresh_interval_frames"):
        OfflinePipeline(
            MockDetector(0.9),
            MockSegmentor(),
            config={"runtime": {"active_target": {"refresh_interval_frames": value}}},
        )


def test_pipeline_rejects_non_mapping_active_target_config():
    with pytest.raises(ValueError, match="active_target"):
        OfflinePipeline(MockDetector(0.9), MockSegmentor(), config={"runtime": {"active_target": True}})


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_pipeline_rejects_invalid_active_target_max_coast_frames(value):
    with pytest.raises(ValueError, match="max_coast_frames"):
        OfflinePipeline(
            MockDetector(0.9),
            MockSegmentor(),
            config={"runtime": {"active_target": {"max_coast_frames": value}}},
        )
