from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from visual_deploy.geometry.grasp_patch import GraspPatch
from visual_deploy.ranking.target_ranker import CandidateScores


@dataclass(frozen=True)
class TargetValidation:
    valid: bool
    reason: str | None = None
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None


class TargetSafetyValidator:
    def __init__(
        self,
        *,
        enabled: bool = False,
        min_target_score: float = 0.0,
        min_track_confidence: float = 0.0,
        min_mask_quality: float = 0.0,
        max_normal_tilt_deg: float = 180.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_target_score = _unit_interval(min_target_score, "min_target_score")
        self.min_track_confidence = _unit_interval(min_track_confidence, "min_track_confidence")
        self.min_mask_quality = _unit_interval(min_mask_quality, "min_mask_quality")
        self.max_normal_tilt_deg = _bounded(max_normal_tilt_deg, "max_normal_tilt_deg", 0.0, 180.0)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> TargetSafetyValidator:
        source = config or {}
        return cls(
            enabled=bool(source.get("enabled", False)),
            min_target_score=float(source.get("min_target_score", 0.0)),
            min_track_confidence=float(source.get("min_track_confidence", 0.0)),
            min_mask_quality=float(source.get("min_mask_quality", 0.0)),
            max_normal_tilt_deg=float(source.get("max_normal_tilt_deg", 180.0)),
        )

    def validate(self, scores: CandidateScores, patch: GraspPatch) -> TargetValidation:
        if not self.enabled:
            return TargetValidation(valid=True)

        checks = (
            ("target_score_below_threshold", "target_score", float(scores.target_score), self.min_target_score, "min"),
            (
                "track_confidence_below_threshold",
                "track_confidence",
                float(scores.track_confidence),
                self.min_track_confidence,
                "min",
            ),
            (
                "mask_quality_below_threshold",
                "mask_quality_score",
                float(scores.mask_quality_score),
                self.min_mask_quality,
                "min",
            ),
            (
                "normal_tilt_above_threshold",
                "normal_tilt_deg",
                _normal_tilt_deg(patch.normal_xyz),
                self.max_normal_tilt_deg,
                "max",
            ),
        )
        for reason, metric, value, threshold, direction in checks:
            failed = value < threshold if direction == "min" else value > threshold
            if failed:
                return TargetValidation(
                    valid=False,
                    reason=reason,
                    metric=metric,
                    value=value,
                    threshold=threshold,
                )
        return TargetValidation(valid=True)


def _normal_tilt_deg(normal_xyz: np.ndarray) -> float:
    normal = np.asarray(normal_xyz, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        return 180.0
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return 180.0
    cosine = float(np.clip(-normal[2] / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _unit_interval(value: float, name: str) -> float:
    return _bounded(value, name, 0.0, 1.0)


def _bounded(value: float, name: str, minimum: float, maximum: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < minimum or numeric > maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return numeric
