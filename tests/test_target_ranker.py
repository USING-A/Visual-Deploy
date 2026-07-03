import pytest

from visual_deploy.ranking.target_ranker import CandidateScores, TargetRanker


def test_target_ranker_chooses_highest_weighted_score():
    ranker = TargetRanker()
    low = CandidateScores(
        track_id=1,
        grasp_score=0.5,
        depth_valid_score=0.9,
        track_confidence=0.9,
        track_stability=1.0,
        mask_quality_score=1.0,
    )
    high = CandidateScores(
        track_id=2,
        grasp_score=0.95,
        depth_valid_score=0.8,
        track_confidence=0.7,
        track_stability=0.8,
        mask_quality_score=0.7,
    )

    ranked = ranker.rank([low, high])

    assert ranked[0].track_id == 2
    assert ranked[0].target_score > low.target_score


def test_target_ranker_ignores_invalid_candidates():
    ranker = TargetRanker()
    invalid = CandidateScores(
        track_id=1,
        grasp_score=1.0,
        depth_valid_score=1.0,
        track_confidence=1.0,
        track_stability=1.0,
        mask_quality_score=1.0,
        valid=False,
    )

    assert ranker.rank([invalid]) == []


def test_target_ranker_rejects_missing_weight_key():
    with pytest.raises(ValueError, match="weights"):
        TargetRanker(weights={"grasp_score": 1.0})


def test_target_ranker_rejects_empty_custom_weights():
    with pytest.raises(ValueError, match="weights"):
        TargetRanker(weights={})


def test_target_ranker_rejects_non_finite_score():
    ranker = TargetRanker()
    candidate = CandidateScores(
        track_id=1,
        grasp_score=float("nan"),
        depth_valid_score=1.0,
        track_confidence=1.0,
        track_stability=1.0,
        mask_quality_score=1.0,
    )

    with pytest.raises(ValueError, match="finite"):
        ranker.rank([candidate])


def test_target_ranker_does_not_partially_mutate_on_failure():
    ranker = TargetRanker()
    valid = CandidateScores(
        track_id=1,
        grasp_score=1.0,
        depth_valid_score=1.0,
        track_confidence=1.0,
        track_stability=1.0,
        mask_quality_score=1.0,
    )
    malformed = CandidateScores(
        track_id=2,
        grasp_score=float("nan"),
        depth_valid_score=1.0,
        track_confidence=1.0,
        track_stability=1.0,
        mask_quality_score=1.0,
    )

    with pytest.raises(ValueError, match="finite"):
        ranker.rank([valid, malformed])

    assert valid.target_score == 0.0
