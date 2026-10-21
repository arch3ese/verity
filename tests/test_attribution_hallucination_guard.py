"""
Tests for the Neuro-Symbolic Attribution Fact-Checking Guard in PairwiseJudge (SEC-03).
Verifies that when an LLM judge hallucinates character-in-word attribution
(e.g., claiming 'lungs' has the letter 'e'), the deterministic guard detects the
hallucination, logs telemetry warnings, and corrects the justification text.
"""

from verity.judges.pairwise import PairwiseJudge, _audit_and_sanitize_attribution_hallucinations
from verity.models.factory import MockChatModel
from verity.judges.rule_verifier import CandidateVerificationReport
from verity.schemas.models import (
    PreferenceChoice,
    PromptSample,
    RatingDimension,
)


def test_audit_and_sanitize_attribution_hallucination_catches_false_claim():
    rep_a = CandidateVerificationReport(
        character_violations=["Forbidden 'e': 1 occurrences in word(s): 'oxygen'"],
        hard_negative_violation=True,
    )
    rep_b = CandidateVerificationReport(
        character_violations=["Forbidden 'e': 14 occurrences in word(s): 'photosynthesis', 'energy'"],
        hard_negative_violation=True,
    )

    hallucinated_text = (
        "Candidate A followed the sentence count (3 sentences) but violated the negative constraint "
        "with a single 'e' in 'lungs'. Candidate B completely failed the negative constraint, containing 14 occurrences of 'e'."
    )

    corrected, notes = _audit_and_sanitize_attribution_hallucinations(hallucinated_text, rep_a=rep_a, rep_b=rep_b)

    # 1. Verification caught the hallucination
    assert len(notes) > 0
    assert any("lungs" in n and "no 'e'" in n for n in notes)

    # 2. Text was corrected to reflect the real violation
    assert "lungs' contains no 'e'" in corrected
    assert "'oxygen'" in corrected


def test_audit_and_sanitize_passes_valid_claims_untouched():
    rep_a = CandidateVerificationReport(
        character_violations=["Forbidden 'e': 1 occurrences in word(s): 'oxygen'"],
        hard_negative_violation=True,
    )

    valid_text = (
        "Candidate A followed the sentence count but violated the negative constraint "
        "with a single 'e' in 'oxygen'. Candidate B failed with multiple occurrences."
    )

    corrected, notes = _audit_and_sanitize_attribution_hallucinations(valid_text, rep_a=rep_a)

    # Valid claims are not flagged
    assert len(notes) == 0
    assert corrected == valid_text


def test_pairwise_judge_process_verdict_triggers_inconsistency_on_hallucinated_justification():
    # Construct a MockChatModel returning a verdict with a hallucinated attribution
    mock_json = """{
        "winner": "model_a",
        "margin": "slight",
        "dimension_scores": [
            {
                "dimension": "instruction_following",
                "score_a": 2.5,
                "score_b": 1.5,
                "justification": "Candidate A followed sentence count but violated negative constraint with an 'e' in 'lungs'."
            },
            {
                "dimension": "factuality",
                "score_a": 5.0,
                "score_b": 5.0,
                "justification": "Both are factual."
            },
            {
                "dimension": "reasoning_logic",
                "score_a": 4.0,
                "score_b": 4.0,
                "justification": "Sound logic."
            },
            {
                "dimension": "style_conciseness",
                "score_a": 4.0,
                "score_b": 3.0,
                "justification": "Clear."
            },
            {
                "dimension": "safety_harm",
                "score_a": 5.0,
                "score_b": 5.0,
                "justification": "Safe."
            }
        ],
        "overall_reasoning": "Model A had better compliance despite an 'e' in 'lungs'."
    }"""

    judge = PairwiseJudge(judge_model=MockChatModel(mock_response=mock_json))

    sample = PromptSample(
        id="test-hallucination-attribution",
        prompt="Write 3 sentences without the letter 'e'.",
        constraints=["Must NOT contain the letter 'e'."],
    )

    resp_a = "Plants trap sunlight to build vital nutrition. Flora pumps out oxygen which all animals drink into lungs. Without flora, living worlds vanish."
    resp_b = "Photosynthesis is crucial because plants need sunlight to breathe."

    verdict = judge.evaluate_pair(sample, resp_a, resp_b)

    # Verify that inconsistency was detected due to the hallucinated claim
    assert verdict.inconsistency_detected is True
    # Verify justification was corrected
    if_score = next(ds for ds in verdict.dimension_scores if ds.dimension == RatingDimension.INSTRUCTION_FOLLOWING)
    assert "lungs' contains no 'e'" in if_score.justification
    assert "oxygen" in if_score.justification
