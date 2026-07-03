from visual_deploy.types import GraspTarget
import numpy as np

from scripts.run_debug_viewer import _append_status_bar, _viewer_status_lines


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
