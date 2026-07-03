import numpy as np

from visual_deploy.detection.yolo_detector import MockDetector
from visual_deploy.pipeline.offline_pipeline import OfflinePipeline
from visual_deploy.segmentation.gcnet_segmentor import SegmentResult
from visual_deploy.types import CameraIntrinsics, DeployFrame


class MockSegmentor:
    def infer(self, color_bgr_256, depth_mm_256):
        mask = np.zeros((256, 256), dtype=bool)
        yy, xx = np.ogrid[:256, :256]
        mask[(yy - 128) ** 2 + (xx - 128) ** 2 <= 60 ** 2] = True
        return SegmentResult(mask.astype(np.float32), mask, int(mask.sum()), 1.0)


def test_pipeline_exposes_sparse_debug_snapshot_with_image_space_mask(tmp_path):
    frame = DeployFrame(
        frame_id=5,
        timestamp_ms=12.5,
        color_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_mm=np.full((480, 640), 500.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001),
    )
    pipeline = OfflinePipeline(
        detector=MockDetector(0.9),
        segmentor=MockSegmentor(),
        config={"recording": {"output_root": str(tmp_path)}},
    )

    target = pipeline.process_frame(frame)
    snapshot = pipeline.last_debug

    assert target.valid is True
    assert snapshot is not None
    assert snapshot.frame_id == 5
    assert len(snapshot.detections) == 1
    assert snapshot.target == target
    assert len(snapshot.masks) == 1
    assert snapshot.masks[0].shape == frame.depth_mm.shape
    assert snapshot.masks[0][240, 320]
