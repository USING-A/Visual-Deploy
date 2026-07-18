import numpy as np
import pytest

from visual_deploy.geometry.grasp_patch import GraspPatch
from visual_deploy.ranking.target_ranker import CandidateScores
from visual_deploy.safety.target_validator import TargetSafetyValidator


def _scores(**overrides) -> CandidateScores:
    values = {
        "track_id": 1,
        "grasp_score": 0.95,
        "depth_valid_score": 0.8,
        "track_confidence": 0.9,
        "track_stability": 1.0,
        "mask_quality_score": 0.9,
        "target_score": 0.88,
    }
    values.update(overrides)
    return CandidateScores(**values)


def _patch(normal=(0.0, 0.0, -1.0)) -> GraspPatch:
    return GraspPatch(
        u_px=10.0,
        v_px=10.0,
        z_m=0.25,
        normal_xyz=np.asarray(normal, dtype=np.float32),
        score=0.95,
        valid_depth_ratio=1.0,
        valid_depth_count=100,
        plane_rmse_m=0.0,
        depth_mad_m=0.0,
        boundary_distance_px=10.0,
        component_area_px=1000,
    )


def test_disabled_validator_preserves_existing_behavior():
    validator = TargetSafetyValidator(enabled=False, min_target_score=1.0)
    assert validator.validate(_scores(target_score=0.1), _patch()).valid is True


@pytest.mark.parametrize(
    ("scores", "patch", "reason"),
    [
        (_scores(target_score=0.79), _patch(), "target_score_below_threshold"),
        (_scores(track_confidence=0.19), _patch(), "track_confidence_below_threshold"),
        (_scores(mask_quality_score=0.59), _patch(), "mask_quality_below_threshold"),
        (_scores(), _patch((0.0, -0.7071068, -0.7071068)), "normal_tilt_above_threshold"),
    ],
)
def test_validator_reports_first_failed_safety_check(scores, patch, reason):
    validator = TargetSafetyValidator(
        enabled=True,
        min_target_score=0.8,
        min_track_confidence=0.2,
        min_mask_quality=0.6,
        max_normal_tilt_deg=35.0,
    )
    result = validator.validate(scores, patch)
    assert result.valid is False
    assert result.reason == reason
    assert result.metric is not None
    assert result.value is not None
    assert result.threshold is not None


def test_validator_accepts_candidate_at_configured_limits():
    validator = TargetSafetyValidator(
        enabled=True,
        min_target_score=0.8,
        min_track_confidence=0.2,
        min_mask_quality=0.6,
        max_normal_tilt_deg=45.0,
    )
    assert validator.validate(_scores(target_score=0.8, track_confidence=0.2, mask_quality_score=0.6), _patch()).valid


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_target_score": -0.1},
        {"min_track_confidence": 1.1},
        {"min_mask_quality": float("nan")},
        {"max_normal_tilt_deg": 181.0},
    ],
)
def test_validator_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TargetSafetyValidator(**kwargs)
