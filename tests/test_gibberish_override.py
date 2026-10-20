"""
Test for Task 2.6: Gibberish winner override test covering surviving mutation MUT-5.
"""

import json
from aligncraft.models.factory import MockChatModel
from aligncraft.judges.pairwise import PairwiseJudge
from aligncraft.schemas.models import ErrorTag, PreferenceChoice, PromptSample, RatingDimension


def test_gibberish_winner_override_overturns_hallucinated_win():
    """
    MUT-5 / Task 2.6: When the LLM judge votes for a gibberish/pure noise candidate,
    PairwiseJudge must overturn the win to the communicative candidate and cap scores at 1.0.
    """
    # LLM judge erroneously votes for the gibberish candidate
    def biased_llm_handler(msg: str) -> str:
        return json.dumps({
            "winner": "model_a",
            "margin": "decisive",
            "dimension_scores": [
                {"dimension": "instruction_following", "score_a": 5.0, "score_b": 3.0, "justification": "Model A is amazing."},
                {"dimension": "factuality", "score_a": 5.0, "score_b": 4.0, "justification": "Model A is factual."}
            ],
            "overall_reasoning": "Model A is concise and avant-garde.",
            "detected_errors_model_a": [],
            "detected_errors_model_b": []
        })

    mock_llm = MockChatModel(response_handler=biased_llm_handler)
    judge = PairwiseJudge(mock_llm)
    sample = PromptSample(id="gibberish-test-01", prompt="Explain photosynthesis")

    gibberish_candidate = "EeEeEeEeEeEeEeEe"  # Repeated non-lexical noise
    valid_candidate = "Photosynthesis is the biological process converting light into chemical energy."

    verdict = judge.evaluate_pair(
        sample=sample,
        response_a=gibberish_candidate,
        response_b=valid_candidate
    )

    # Overturned: Model B must win over the noise candidate
    assert verdict.winner == PreferenceChoice.MODEL_B, (
        f"MUT-5 Defect: Gibberish winner override failed! Expected MODEL_B, got {verdict.winner}"
    )
    assert ErrorTag.SEMANTIC_VOID in verdict.detected_errors_model_a
    assert "disqualified" in verdict.overall_reasoning.lower() or "override" in verdict.overall_reasoning.lower()

    # Scores for gibberish must be capped at 1.0
    for ds in verdict.dimension_scores:
        if ds.dimension == RatingDimension.INSTRUCTION_FOLLOWING:
            assert ds.score_a == 1.0, f"Expected gibberish instruction_following score 1.0, got {ds.score_a}"
