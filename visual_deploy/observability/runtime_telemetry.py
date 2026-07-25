from __future__ import annotations

from dataclasses import asdict, dataclass, field

from visual_deploy.types import DeployFrame


@dataclass(frozen=True)
class FrameRuntimeTelemetry:
    capture_ms: float = 0.0
    captured_monotonic_ms: float = 0.0
    queue_wait_ms: dict[str, float] = field(default_factory=dict)
    queue_depth: dict[str, int] = field(default_factory=dict)
    queue_capacity: dict[str, int] = field(default_factory=dict)
    dropped_frames: dict[str, int] = field(default_factory=dict)
    resources: dict[str, float] = field(default_factory=dict)


def attach_runtime_telemetry(frame: DeployFrame, telemetry: FrameRuntimeTelemetry) -> None:
    frame.meta["runtime_telemetry"] = asdict(telemetry)


def read_runtime_telemetry(frame: DeployFrame) -> FrameRuntimeTelemetry:
    value = frame.meta.get("runtime_telemetry", {})
    if not isinstance(value, dict):
        return FrameRuntimeTelemetry()
    fields = FrameRuntimeTelemetry.__dataclass_fields__
    return FrameRuntimeTelemetry(**{name: value[name] for name in fields if name in value})
