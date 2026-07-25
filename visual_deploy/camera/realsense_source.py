from __future__ import annotations

import numpy as np
from time import perf_counter

from visual_deploy.observability.runtime_telemetry import FrameRuntimeTelemetry, attach_runtime_telemetry
from visual_deploy.types import CameraIntrinsics, DeployFrame

try:
    import pyrealsense2 as rs
    _RS_IMPORT_ERROR = None
except Exception as exc:
    rs = None
    _RS_IMPORT_ERROR = exc


class RealSenseUnavailableError(RuntimeError):
    pass


class RealSenseSource:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_to_color: bool = True,
        spatial_filter: bool = True,
        temporal_filter: bool = False,
        hole_filling_filter: bool = False,
    ) -> None:
        width = _validate_positive_int(width, "width")
        height = _validate_positive_int(height, "height")
        fps = _validate_positive_int(fps, "fps")
        if not align_to_color:
            raise ValueError("align_to_color must be True for color-space detection and backprojection")
        if rs is None:
            message = "pyrealsense2 is unavailable"
            if _RS_IMPORT_ERROR is not None:
                message = f"{message}: {_RS_IMPORT_ERROR}"
            raise RealSenseUnavailableError(message) from _RS_IMPORT_ERROR

        self.frame_id = 0
        self.align_to_color = bool(align_to_color)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color) if align_to_color else None
        self.spatial = rs.spatial_filter() if spatial_filter else None
        self.temporal = rs.temporal_filter() if temporal_filter else None
        self.hole = rs.hole_filling_filter() if hole_filling_filter else None

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale_m = float(depth_sensor.get_depth_scale())

    def __iter__(self) -> RealSenseSource:
        return self

    def __next__(self) -> DeployFrame:
        capture_started = perf_counter()
        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise StopIteration

        depth_frame = self._filter_depth(depth_frame)
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_mm = depth_raw * np.float32(self.depth_scale_m * 1000.0)

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        frame = DeployFrame(
            frame_id=self.frame_id,
            timestamp_ms=float(color_frame.get_timestamp()),
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            intrinsics=CameraIntrinsics(
                fx=float(intr.fx),
                fy=float(intr.fy),
                ppx=float(intr.ppx),
                ppy=float(intr.ppy),
                depth_scale=0.001,
            ),
            meta={
                "source": "realsense",
                "aligned": self.align is not None,
                "sensor_depth_scale_m": self.depth_scale_m,
            },
        )
        captured_monotonic_ms = perf_counter() * 1000.0
        attach_runtime_telemetry(
            frame,
            FrameRuntimeTelemetry(
                capture_ms=(captured_monotonic_ms - capture_started * 1000.0),
                captured_monotonic_ms=captured_monotonic_ms,
            ),
        )
        self.frame_id += 1
        return frame

    def close(self) -> None:
        self.pipeline.stop()

    def _filter_depth(self, depth_frame):
        if self.spatial is not None:
            depth_frame = self.spatial.process(depth_frame)
        if self.temporal is not None:
            depth_frame = self.temporal.process(depth_frame)
        if self.hole is not None:
            depth_frame = self.hole.process(depth_frame)
        return depth_frame


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
