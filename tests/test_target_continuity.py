import pytest

from visual_deploy.safety.target_continuity import TargetContinuityValidator


def _validator() -> TargetContinuityValidator:
    return TargetContinuityValidator(
        enabled=True,
        max_pixel_step_px=40.0,
        max_depth_step_mm=30.0,
        max_history_age_ms=1000.0,
    )


def test_continuity_accepts_first_and_nearby_target():
    validator = _validator()
    assert validator.validate(1, 0.0, 100.0, 100.0, 250.0).valid
    validator.accept(1, 0.0, 100.0, 100.0, 250.0)
    assert validator.validate(1, 33.0, 110.0, 110.0, 255.0).valid


def test_continuity_rejects_depth_jump():
    validator = _validator()
    validator.accept(1, 0.0, 100.0, 100.0, 250.0)
    result = validator.validate(1, 33.0, 100.0, 100.0, 300.1)
    assert result.valid is False
    assert result.reason == "target_depth_jump"


def test_continuity_rejects_pixel_jump():
    validator = _validator()
    validator.accept(1, 0.0, 100.0, 100.0, 250.0)
    result = validator.validate(1, 33.0, 141.0, 100.0, 250.0)
    assert result.valid is False
    assert result.reason == "target_pixel_jump"


def test_continuity_allows_reacquisition_after_history_expires():
    validator = _validator()
    validator.accept(1, 0.0, 100.0, 100.0, 250.0)
    assert validator.validate(1, 1001.0, 500.0, 400.0, 700.0).valid


def test_continuity_tracks_targets_independently():
    validator = _validator()
    validator.accept(1, 0.0, 100.0, 100.0, 250.0)
    assert validator.validate(2, 33.0, 500.0, 400.0, 700.0).valid


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pixel_step_px": -1.0},
        {"max_depth_step_mm": float("nan")},
        {"max_history_age_ms": -1.0},
    ],
)
def test_continuity_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TargetContinuityValidator(**kwargs)
