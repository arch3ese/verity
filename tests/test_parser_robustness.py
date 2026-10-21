"""Tests for Judge Parser Robustness (Task 3.7 & EVAL-03)."""

import json
import pytest
from verity.judges.pairwise import JudgeParseError, PairwiseJudge
from verity.models.factory import MockChatModel
from verity.schemas.models import PreferenceChoice, PromptSample


SAMPLE = PromptSample(id="test-1", prompt="Write a haiku about nature.")
RESP_A = "Green leaves in the breeze\nBirds sing on the branch so high\nSummer sun is warm"
RESP_B = "Winter snow falls down\nCold winds blow across the plain\nSilence everywhere"

VALID_VERDICT_DICT = {
    "winner": "model_a",
    "margin": "slight",
    "dimension_scores": [
        {"dimension": "instruction_following", "score_a": 5.0, "score_b": 5.0, "justification": "Both wrote haikus."},
        {"dimension": "factuality", "score_a": 5.0, "score_b": 5.0, "justification": "Creative task."},
        {"dimension": "reasoning_logic", "score_a": 5.0, "score_b": 5.0, "justification": "Valid logic."},
        {"dimension": "safety_harm", "score_a": 5.0, "score_b": 5.0, "justification": "Safe."},
        {"dimension": "style_conciseness", "score_a": 4.5, "score_b": 4.0, "justification": "Model A has better imagery."}
    ],
    "overall_reasoning": "Model A has slightly better poetic phrasing.",
    "detected_errors_model_a": [],
    "detected_errors_model_b": []
}


def test_parser_extracts_markdown_fenced_json_with_surrounding_text():
    """Verify parser extracts JSON verdict inside markdown fences with leading and trailing text (3.7)."""
    raw_response = (
        "Sure, here is the detailed evaluation:\n\n"
        "```json\n"
        f"{json.dumps(VALID_VERDICT_DICT, indent=2)}\n"
        "```\n\n"
        "I hope this helps your evaluation pipeline!"
    )
    judge = PairwiseJudge(judge_model=MockChatModel(mock_response=raw_response))
    verdict = judge.evaluate_pair(SAMPLE, RESP_A, RESP_B)

    assert verdict.winner == PreferenceChoice.MODEL_A
    assert verdict.margin == "slight"
    assert "poetic phrasing" in verdict.overall_reasoning


def test_parser_extracts_json_with_trailing_conversational_text():
    """Verify parser extracts valid JSON even with trailing commentary without fences (3.7)."""
    raw_response = (
        f"{json.dumps(VALID_VERDICT_DICT)}\n\n"
        "Note: Candidate A demonstrated stronger poetic cadence."
    )
    judge = PairwiseJudge(judge_model=MockChatModel(mock_response=raw_response))
    verdict = judge.evaluate_pair(SAMPLE, RESP_A, RESP_B)

    assert verdict.winner == PreferenceChoice.MODEL_A


def test_parser_fails_with_clear_error_on_refusal():
    """Verify model refusal raises JudgeParseError and never produces a confident verdict (3.7)."""
    refusal_text = "I apologize, but I am unable to process this request due to safety policies."
    judge = PairwiseJudge(judge_model=MockChatModel(mock_response=refusal_text))

    with pytest.raises(JudgeParseError) as exc_info:
        judge.evaluate_pair(SAMPLE, RESP_A, RESP_B)

    err = str(exc_info.value).lower()
    assert "refusal" in err or "failed to parse" in err or "cannot fulfill" in err


def test_parser_fails_with_clear_error_on_empty_output():
    """Verify empty model output raises JudgeParseError and never produces a confident verdict (3.7)."""
    judge = PairwiseJudge(judge_model=MockChatModel(mock_response="   "))

    with pytest.raises(JudgeParseError) as exc_info:
        judge.evaluate_pair(SAMPLE, RESP_A, RESP_B)

    err = str(exc_info.value).lower()
    assert "empty" in err or "failed to parse" in err


def test_parser_fails_with_clear_error_on_truncated_output():
    """Verify truncated model output raises JudgeParseError and never produces a confident verdict (3.7)."""
    truncated_text = '{"winner": "model_a", "margin": "decisive", "dimension_scores": [{"dimension":'
    judge = PairwiseJudge(judge_model=MockChatModel(mock_response=truncated_text))

    with pytest.raises(JudgeParseError) as exc_info:
        judge.evaluate_pair(SAMPLE, RESP_A, RESP_B)

    err = str(exc_info.value).lower()
    assert "truncated" in err or "failed to parse" in err
