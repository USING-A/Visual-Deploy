"""Runtime telemetry contracts for profiling and future threaded queues."""

from .runtime_telemetry import FrameRuntimeTelemetry, attach_runtime_telemetry, read_runtime_telemetry

__all__ = ["FrameRuntimeTelemetry", "attach_runtime_telemetry", "read_runtime_telemetry"]
