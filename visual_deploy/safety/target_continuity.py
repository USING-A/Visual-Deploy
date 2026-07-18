from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from visual_deploy.safety.target_validator import TargetValidation


@dataclass(frozen=True)
class _AcceptedTarget:
    timestamp_ms: float
    u_px: float
    v_px: float
    z_mm: float


class TargetContinuityValidator:
    def __init__(
        self,
        *,
        enabled: bool = False,
        max_pixel_step_px: float = 1e9,
        max_depth_step_mm: float = 1e9,
        max_history_age_ms: float = 1000.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_pixel_step_px = _non_negative(max_pixel_step_px, "max_pixel_step_px")
        self.max_depth_step_mm = _non_negative(max_depth_step_mm, "max_depth_step_mm")
        self.max_history_age_ms = _non_negative(max_history_age_ms, "max_history_age_ms")
        self._accepted: dict[int, _AcceptedTarget] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> TargetContinuityValidator:
        source = config or {}
        return cls(
            enabled=bool(source.get("enabled", False)),
            max_pixel_step_px=float(source.get("max_pixel_step_px", 1e9)),
            max_depth_step_mm=float(source.get("max_depth_step_mm", 1e9)),
            max_history_age_ms=float(source.get("max_history_age_ms", 1000.0)),
        )

    def validate(
        self,
        track_id: int,
        timestamp_ms: float,
        u_px: float,
        v_px: float,
        z_mm: float,
    ) -> TargetValidation:
        values = np.asarray([timestamp_ms, u_px, v_px, z_mm], dtype=np.float64)
        if not np.isfinite(values).all():
            return TargetValidation(valid=False, reason="non_finite_target")
        if not self.enabled:
            return TargetValidation(valid=True)

        previous = self._accepted.get(int(track_id))
        if previous is None:
            return TargetValidation(valid=True)
        age_ms = float(timestamp_ms) - previous.timestamp_ms
        if age_ms < 0.0:
            return TargetValidation(
                valid=False,
                reason="target_timestamp_regression",
                metric="target_age_ms",
                value=age_ms,
                threshold=0.0,
            )
        if age_ms > self.max_history_age_ms:
            return TargetValidation(valid=True)

        depth_step = abs(float(z_mm) - previous.z_mm)
        if depth_step > self.max_depth_step_mm:
            return TargetValidation(
                valid=False,
                reason="target_depth_jump",
                metric="depth_step_mm",
                value=depth_step,
                threshold=self.max_depth_step_mm,
            )
        pixel_step = float(np.hypot(float(u_px) - previous.u_px, float(v_px) - previous.v_px))
        if pixel_step > self.max_pixel_step_px:
            return TargetValidation(
                valid=False,
                reason="target_pixel_jump",
                metric="pixel_step_px",
                value=pixel_step,
                threshold=self.max_pixel_step_px,
            )
        return TargetValidation(valid=True)

    def accept(self, track_id: int, timestamp_ms: float, u_px: float, v_px: float, z_mm: float) -> None:
        self._accepted[int(track_id)] = _AcceptedTarget(
            timestamp_ms=float(timestamp_ms),
            u_px=float(u_px),
            v_px=float(v_px),
            z_mm=float(z_mm),
        )

    def clear(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._accepted.clear()
        else:
            self._accepted.pop(int(track_id), None)


def _non_negative(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric
