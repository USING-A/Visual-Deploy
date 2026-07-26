from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_WEIGHTS: dict[str, float] = {
    "grasp_score": 0.45,
    "depth_valid_score": 0.20,
    "track_confidence": 0.15,
    "track_stability": 0.10,
    "mask_quality_score": 0.10,
}


@dataclass
class CandidateScores:
    track_id: int
    grasp_score: float
    depth_valid_score: float
    track_confidence: float
    track_stability: float
    mask_quality_score: float
    valid: bool = True
    target_score: float = 0.0


class TargetRanker:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = _validate_weights(DEFAULT_WEIGHTS if weights is None else weights)

    def score(self, candidate: CandidateScores) -> float:
        values = {
            "grasp_score": candidate.grasp_score,
            "depth_valid_score": candidate.depth_valid_score,
            "track_confidence": candidate.track_confidence,
            "track_stability": candidate.track_stability,
            "mask_quality_score": candidate.mask_quality_score,
        }
        _validate_score_values(values)
        return float(sum(self.weights[name] * float(values[name]) for name in DEFAULT_WEIGHTS))

    def rank(self, candidates: list[CandidateScores]) -> list[CandidateScores]:
        valid_candidates = [candidate for candidate in candidates if candidate.valid]
        scores = [self.score(candidate) for candidate in valid_candidates]
        for candidate, score in zip(valid_candidates, scores):
            candidate.target_score = score
        return sorted(valid_candidates, key=lambda item: item.target_score, reverse=True)


def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
    missing = set(DEFAULT_WEIGHTS) - set(weights)
    extra = set(weights) - set(DEFAULT_WEIGHTS)
    if missing or extra:
        raise ValueError(f"weights must contain exactly these keys: {sorted(DEFAULT_WEIGHTS)}")

    out: dict[str, float] = {}
    for name, value in weights.items():
        weight = float(value)
        if not np.isfinite(weight):
            raise ValueError("weights must be finite")
        out[name] = weight
    return out


def _validate_score_values(values: dict[str, float]) -> None:
    for name, value in values.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
