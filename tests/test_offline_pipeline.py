import json

import numpy as np
import pytest

from visual_deploy.detection.yolo_detector import MockDetector
from visual_deploy.pipeline.offline_pipeline import OfflinePipeline
from visual_deploy.segmentation.gcnet_segmentor import SegmentResult
from scripts.run_offline_smoke import _build_detector, _build_segmentor
from visual_deploy.types import CameraIntrinsics, DeployFrame
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
    timing = json.loads((run_dir / "timings.jsonl").read_text(encoding="utf-8").strip())
    assert timing["valid_target"] is True
    assert timing["detection_ms"] >= 0.0
    assert timing["segmentation_ms"] >= 0.0
    assert timing["grasp_ms"] >= 0.0
    assert timing["total_ms"] >= timing["detection_ms"]


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
