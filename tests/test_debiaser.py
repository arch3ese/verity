"""Unit tests for the Debiasing Engine (Positional Swapping & Verbosity Checks)."""

import json
from aligncraft.judges.debiaser import DebiasingJudge
from aligncraft.judges.pairwise import PairwiseJudge
from aligncraft.models.factory import MockChatModel
from aligncraft.schemas.models import (
    DimensionScore,
    PairwiseVerdict,
    PreferenceChoice,
    PromptSample,
    RatingDimension,
)


def test_debiaser_order_invariant_consensus():
    """Test that an unbiased judge evaluates content consistently across orders."""
    # When Response A is superior:
    # Forward pass (A first, B second): winner is model_a
    # Reverse pass (B first, A second): winner is model_b (since A is in second position)
    def fair_handler(msg: str) -> str:
        cand_a_part = msg.split("### CANDIDATE B:")[0]
        # Superior answer contains "Superior content"
        if "Superior content" in cand_a_part:
            winner = "model_a"
        else:
            winner = "model_b"

        return json.dumps({
            "winner": winner,
            "margin": "clear",
            "dimension_scores": [
                {"dimension": "factuality", "score_a": 5.0, "score_b": 3.0, "justification": "Superior grounding"}
            ],
            "overall_reasoning": "Candidate with superior content wins.",
            "detected_errors_model_a": [],
            "detected_errors_model_b": []
        })

    mock_llm = MockChatModel(response_handler=fair_handler)
    judge = PairwiseJudge(mock_llm)
    debiaser = DebiasingJudge(judge)

    sample = PromptSample(id="test-order", prompt="Test prompt")
    eval_result = debiaser.evaluate_with_debiasing(
        sample=sample,
        response_a="Superior content here.",
        response_b="Inferior content here.",
    )

    assert eval_result.positional_bias_detected is False
    assert eval_result.final_consensus_winner == PreferenceChoice.MODEL_A
    assert eval_result.confidence_score == 0.95


def test_debiaser_detects_positional_bias():
    """Test that a biased judge favoring position 1 is detected and flagged."""
    # Biased judge ALWAYS chooses model_a regardless of content
    def biased_handler(msg: str) -> str:
        return json.dumps({
            "winner": "model_a",
            "margin": "slight",
            "dimension_scores": [],
            "overall_reasoning": "I always pick candidate 1.",
            "detected_errors_model_a": [],
            "detected_errors_model_b": []
        })

    mock_llm = MockChatModel(response_handler=biased_handler)
    judge = PairwiseJudge(mock_llm)
    debiaser = DebiasingJudge(judge)

    sample = PromptSample(id="test-pos-bias", prompt="Test prompt")
    eval_result = debiaser.evaluate_with_debiasing(
        sample=sample,
        response_a="Response A",
        response_b="Response B",
    )

    assert eval_result.positional_bias_detected is True
    assert eval_result.final_consensus_winner == PreferenceChoice.INDETERMINATE
    assert eval_result.confidence_score == 0.30
    assert "Positional bias detected" in eval_result.summary_critique


def test_verbosity_bias_flag():
    """Test that an excessively long response without factual advantage is flagged."""
    def verbose_favored_handler(msg: str) -> str:
        # Check which candidate is verbose in this pass between candidate markers
        cand_a_text = msg.split("### CANDIDATE A:")[1].split("### CANDIDATE B:")[0]
        cand_a_is_verbose = len(cand_a_text) > 300
        winner = "model_a" if cand_a_is_verbose else "model_b"

        return json.dumps({
            "winner": winner,
            "margin": "slight",
            "dimension_scores": [
                {"dimension": "factuality", "score_a": 3.0, "score_b": 4.0, "justification": "Equal or lower factuality"}
            ],
            "overall_reasoning": "I picked the longer candidate because it was so comprehensive!",
            "detected_errors_model_a": [],
            "detected_errors_model_b": []
        })

    mock_llm = MockChatModel(response_handler=verbose_favored_handler)
    judge = PairwiseJudge(mock_llm)
    debiaser = DebiasingJudge(judge, verbosity_ratio_threshold=2.0)

    sample = PromptSample(id="test-verbosity", prompt="Brief question")
    # Response A is much longer than Response B with realistic diverse words
    long_response_a = "This is a comprehensive and detailed explanation of the concept with many sentences and explanations. " * 15
    short_response_b = "This is a direct and concise response answering the prompt."

    eval_result = debiaser.evaluate_with_debiasing(
        sample=sample,
        response_a=long_response_a,
        response_b=short_response_b,
    )

    assert eval_result.verbosity_bias_flag is True
