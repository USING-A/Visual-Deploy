from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_debug_viewer
from scripts.run_debug_viewer import _append_status_bar, _viewer_status_lines, _window_was_closed
from visual_deploy.types import CameraIntrinsics, DeployFrame, GraspTarget


def test_debug_viewer_status_lines_stay_sparse():
    target = GraspTarget(valid=True, frame_id=2, track_id=4, u_px=100.0, v_px=80.0, z_mm=510.0, target_score=0.77)

    lines = _viewer_status_lines(frame_id=2, fps=18.5, target=target, paused=False)

    assert len(lines) <= 4
    assert lines[0] == "frame=2 fps=18.5 live"
    assert lines[1] == "target id=4 u=100.0 v=80.0 z=510mm score=0.77"


def test_debug_viewer_status_bar_keeps_main_image_size_width_only():
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    view = _append_status_bar(image, ["frame=1", "keys: q quit"])

    assert view.shape[1] == image.shape[1]
    assert view.shape[0] > image.shape[0]
    assert np.array_equal(view[: image.shape[0]], image)


def test_debug_viewer_detects_window_manager_close(monkeypatch):
    monkeypatch.setattr(run_debug_viewer.cv2, "getWindowProperty", lambda name, prop: 0.0)

    assert _window_was_closed("viewer") is True


def test_debug_viewer_window_close_releases_source_and_pipeline(monkeypatch):
    frame = DeployFrame(
        frame_id=1,
        timestamp_ms=1.0,
        color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_mm=np.ones((2, 2), dtype=np.float32),
        intrinsics=CameraIntrinsics(1.0, 1.0, 0.0, 0.0),
    )

    class FakeSource:
        closed = False

        def __iter__(self):
            return iter([frame])

        def close(self):
            self.closed = True

    class FakePipeline:
        def __init__(self):
            self.last_debug = None
            self.closed = False

        def process_frame(self, current):
            return GraspTarget(valid=False, frame_id=current.frame_id, reason="test")

        def close(self):
            self.closed = True

    args = SimpleNamespace(
        config="configs/deploy.yaml",
        use_mock_models=False,
        window_name="test",
        wait_ms=1,
        labels=False,
        max_frames=None,
        rgb=None,
        depth=None,
    )
    source = FakeSource()
    pipeline = FakePipeline()
    monkeypatch.setattr(run_debug_viewer, "_parse_args", lambda: args)
    monkeypatch.setattr(run_debug_viewer, "load_config", lambda path: {})
    monkeypatch.setattr(run_debug_viewer, "_build_detector", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_debug_viewer, "_build_segmentor", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_debug_viewer, "OfflinePipeline", lambda **kwargs: pipeline)
    monkeypatch.setattr(run_debug_viewer, "_build_source", lambda *args: source)
    monkeypatch.setattr(run_debug_viewer, "_render_view", lambda *args, **kwargs: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(run_debug_viewer.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(run_debug_viewer.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(run_debug_viewer.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(run_debug_viewer.cv2, "getWindowProperty", lambda *args: 0.0)
    monkeypatch.setattr(run_debug_viewer.cv2, "destroyAllWindows", lambda: None)

    run_debug_viewer.main()

    assert source.closed is True
    assert pipeline.closed is True


def test_debug_viewer_releases_detector_when_later_model_construction_fails(monkeypatch):
    class CloseableDetector:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    args = SimpleNamespace(config="configs/deploy.yaml", use_mock_models=False)
    detector = CloseableDetector()
    monkeypatch.setattr(run_debug_viewer, "_parse_args", lambda: args)
    monkeypatch.setattr(run_debug_viewer, "load_config", lambda path: {})
    monkeypatch.setattr(run_debug_viewer, "_build_detector", lambda *args, **kwargs: detector)
    monkeypatch.setattr(
        run_debug_viewer,
        "_build_segmentor",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("segmentor failed")),
    )
    monkeypatch.setattr(run_debug_viewer.cv2, "destroyAllWindows", lambda: None)

    with pytest.raises(RuntimeError, match="segmentor failed"):
        run_debug_viewer.main()

    assert detector.closed is True
