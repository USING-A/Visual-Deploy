from __future__ import annotations

from time import sleep

import numpy as np
import pytest

from visual_deploy.observability.runtime_telemetry import FrameRuntimeTelemetry, attach_runtime_telemetry, read_runtime_telemetry
from visual_deploy.runtime.latest_frame_source import LatestFrameSource
from visual_deploy.types import CameraIntrinsics, DeployFrame


def _frame(frame_id: int) -> DeployFrame:
    frame = DeployFrame(
        frame_id=frame_id,
        timestamp_ms=float(frame_id),
        color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_mm=np.zeros((2, 2), dtype=np.float32),
        intrinsics=CameraIntrinsics(1.0, 1.0, 0.0, 0.0),
    )
    attach_runtime_telemetry(
        frame,
        FrameRuntimeTelemetry(capture_ms=1.5, captured_monotonic_ms=10.0 + frame_id),
    )
    return frame


def test_latest_frame_source_keeps_queue_bounded_and_records_drops():
    source = LatestFrameSource(_frame(index) for index in range(100))
    source.start()
    sleep(0.02)

    frames = list(source)

    assert frames[-1].frame_id == 99
    assert source.dropped_frames > 0
    telemetry = read_runtime_telemetry(frames[-1])
    assert telemetry.capture_ms == 1.5
    assert telemetry.queue_capacity == {"capture_to_inference": 1}
    assert telemetry.queue_depth == {"capture_to_inference": 0}
    assert telemetry.dropped_frames["capture_to_inference"] == source.dropped_frames
    assert telemetry.queue_wait_ms["capture_to_inference"] >= 0.0


def test_latest_frame_source_propagates_capture_failure():
    def failing_source():
        yield _frame(1)
        raise RuntimeError("camera disconnected")

    source = LatestFrameSource(failing_source())

    assert next(source).frame_id == 1
    with pytest.raises(RuntimeError, match="camera disconnected"):
        next(source)
    source.close()


def test_latest_frame_source_close_is_idempotent():
    source = LatestFrameSource(iter([_frame(1)]))
    source.start()
    source.close()
    source.close()
