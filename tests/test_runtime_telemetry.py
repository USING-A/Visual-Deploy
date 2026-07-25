from visual_deploy.observability.runtime_telemetry import (
    FrameRuntimeTelemetry,
    attach_runtime_telemetry,
    read_runtime_telemetry,
)
from visual_deploy.types import CameraIntrinsics, DeployFrame

import numpy as np


def test_runtime_telemetry_contract_carries_future_queue_data():
    frame = DeployFrame(
        1,
        0.0,
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
        CameraIntrinsics(1.0, 1.0, 0.0, 0.0),
    )
    expected = FrameRuntimeTelemetry(
        capture_ms=2.5,
        captured_monotonic_ms=100.0,
        queue_wait_ms={"capture_to_inference": 1.2},
        queue_depth={"capture_to_inference": 1},
        queue_capacity={"capture_to_inference": 2},
        dropped_frames={"capture": 3},
        resources={"gpu_util_percent": 55.0},
    )
    attach_runtime_telemetry(frame, expected)
    assert read_runtime_telemetry(frame) == expected


def test_runtime_telemetry_defaults_for_offline_frames():
    frame = DeployFrame(
        1,
        0.0,
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
        CameraIntrinsics(1.0, 1.0, 0.0, 0.0),
    )
    assert read_runtime_telemetry(frame) == FrameRuntimeTelemetry()
