import numpy as np
import pytest

from visual_deploy.detection.yolo_detector import MockDetector
from visual_deploy.pipeline.offline_pipeline import OfflinePipeline
from visual_deploy.segmentation.gcnet_segmentor import SegmentResult
from scripts.run_offline_smoke import _build_detector, _build_segmentor
from visual_deploy.types import CameraIntrinsics, DeployFrame


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


def _frame() -> DeployFrame:
    return DeployFrame(
        frame_id=1,
        timestamp_ms=1.0,
        color_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_mm=np.full((480, 640), 500.0, dtype=np.float32),
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
        config={"recording": {"output_root": str(tmp_path)}},
    )

    target = pipeline.process_frame(_frame())

    assert target.valid is False
    assert target.reason == "no_valid_grasp_candidate"


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

    with pytest.raises(FileNotFoundError, match="YOLO weights"):
        _build_detector(config_path, {"detection": {"weights": "missing_yolo.pt"}}, use_mock_models=False)

    with pytest.raises(FileNotFoundError, match="GCNet weights"):
        _build_segmentor(config_path, {"segmentation": {"weights": "missing_gcnet.pt"}}, use_mock_models=False)

    assert isinstance(
        _build_detector(config_path, {"detection": {"weights": "missing_yolo.pt"}}, use_mock_models=True),
        MockDetector,
    )
    assert hasattr(_build_segmentor(config_path, {"segmentation": {"weights": "missing_gcnet.pt"}}, use_mock_models=True), "infer")
