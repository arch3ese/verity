"""Unit tests for rubric definitions and prompt formatting."""

from verity.rubrics.definitions import (
    ERROR_TAXONOMY_DESCRIPTIONS,
    RUBRIC_CRITERIA,
    format_rubric_prompt,
)
from verity.schemas.models import ErrorTag, RatingDimension


def test_rubric_criteria_completeness():
    # Every rating dimension must be defined in RUBRIC_CRITERIA
    for dim in RatingDimension:
        assert dim in RUBRIC_CRITERIA
        criteria = RUBRIC_CRITERIA[dim]
        assert "description" in criteria
        for score_key in ["score_1", "score_2", "score_3", "score_4", "score_5"]:
            assert score_key in criteria


def test_error_taxonomy_completeness():
    # Every error tag must have an explanatory description
    for tag in ErrorTag:
        assert tag in ERROR_TAXONOMY_DESCRIPTIONS
        assert len(ERROR_TAXONOMY_DESCRIPTIONS[tag]) > 10


def test_format_rubric_prompt():
    prompt = format_rubric_prompt()
    assert "## RLHF MULTI-DIMENSIONAL EVALUATION RUBRIC" in prompt
    assert "INSTRUCTION_FOLLOWING" in prompt
    assert "FACTUALITY" in prompt
    assert "ERROR TAXONOMY" in prompt
    assert "negative_constraint_violation" in prompt
