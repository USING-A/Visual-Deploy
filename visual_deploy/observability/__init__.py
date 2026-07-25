"""Runtime telemetry contracts for profiling and future threaded queues."""

from .runtime_telemetry import FrameRuntimeTelemetry, attach_runtime_telemetry, read_runtime_telemetry
from .profile_analysis import build_profile_summary, parse_tegrastats_line, write_profile_reports

__all__ = [
    "FrameRuntimeTelemetry",
    "attach_runtime_telemetry",
    "read_runtime_telemetry",
    "build_profile_summary",
    "parse_tegrastats_line",
    "write_profile_reports",
]
