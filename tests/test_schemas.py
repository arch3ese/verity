"""Unit tests for Verity data schemas and Pydantic models."""

import pytest
from pydantic import ValidationError
from verity.schemas.models import (
    DimensionScore,
    DPOTriplet,
    ErrorTag,
    PairwiseVerdict,
    PreferenceChoice,
    PromptSample,
    RatingDimension,
)


def test_prompt_sample_creation():
    sample = PromptSample(
        id="test-01",
        prompt="Explain quantum entanglement without using the word physics.",
        constraints=["Do not use the word 'physics'."],
        reference_ground_truth="Quantum entanglement is a phenomenon...",
        human_preferred=PreferenceChoice.MODEL_A,
    )
    assert sample.id == "test-01"
    assert len(sample.constraints) == 1
    assert sample.human_preferred == PreferenceChoice.MODEL_A


def test_dimension_score_bounds():
    # Valid score
    score = DimensionScore(
        dimension=RatingDimension.FACTUALITY,
        score_a=4.5,
        score_b=2.0,
        justification="Model A is grounded; Model B hallucinated.",
    )
    assert score.score_a == 4.5

    # Invalid score > 5.0
    with pytest.raises(ValidationError):
        DimensionScore(
            dimension=RatingDimension.FACTUALITY,
            score_a=5.5,
            score_b=3.0,
            justification="Invalid score",
        )

    # Invalid score < 1.0
    with pytest.raises(ValidationError):
        DimensionScore(
            dimension=RatingDimension.FACTUALITY,
            score_a=0.5,
            score_b=3.0,
            justification="Invalid score",
        )


def test_dpo_triplet_serialization():
    triplet = DPOTriplet(
        prompt="What is 2+2?",
        chosen="2+2 is 4.",
        rejected="2+2 is 5.",
        margin="decisive",
        justification="Rejected answer contains elementary arithmetic error.",
        metadata={"category": "math"},
    )
    json_str = triplet.model_dump_json()
    assert "chosen" in json_str
    assert "rejected" in json_str
    assert "2+2 is 4." in json_str
