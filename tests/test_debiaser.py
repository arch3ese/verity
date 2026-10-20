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


def test_tie_reconciled_when_dimension_score_diff_exists():
    """Verify that ties are strictly avoided when one model has a rubric score advantage (e.g. 4 vs 3)."""
    def tie_with_score_diff_handler(msg: str) -> str:
        cand_a_part = msg.split("### CANDIDATE B:")[0]
        # Response B is the better response regardless of order
        if "Response B text" in cand_a_part:
            score_a, score_b = 4.0, 3.0
        else:
            score_a, score_b = 3.0, 4.0

        return json.dumps({
            "winner": "tie",  # LLM erroneously called it a tie
            "margin": "slight",
            "dimension_scores": [
                {"dimension": "instruction_following", "score_a": 4.0, "score_b": 4.0, "justification": "Equal"},
                {"dimension": "factuality", "score_a": 4.0, "score_b": 4.0, "justification": "Equal"},
                {"dimension": "reasoning_logic", "score_a": 4.0, "score_b": 4.0, "justification": "Equal"},
                {"dimension": "safety_harm", "score_a": 5.0, "score_b": 5.0, "justification": "Equal"},
                {"dimension": "style_conciseness", "score_a": score_a, "score_b": score_b, "justification": "Tone difference"}
            ],
            "overall_reasoning": "Both models are good but one has slightly better style.",
            "detected_errors_model_a": [],
            "detected_errors_model_b": []
        })

    mock_llm = MockChatModel(response_handler=tie_with_score_diff_handler)
    judge = PairwiseJudge(mock_llm)
    debiaser = DebiasingJudge(judge)

    sample = PromptSample(id="test-tie-avoidance", prompt="Prompt testing tie avoidance")
    eval_result = debiaser.evaluate_with_debiasing(
        sample=sample,
        response_a="Response A text",
        response_b="Response B text",
    )

    # Model B had 4.0 on style vs 3.0 for A -> Model B must be preferred, NOT a tie!
    assert eval_result.final_consensus_winner == PreferenceChoice.MODEL_B
    assert eval_result.forward_verdict.winner == PreferenceChoice.MODEL_B


def test_human_ground_truth_calibration_overrides_tie():
    """Verify that submitting a human audit and enabling calibration overrides an AI tie on that exact prompt."""
    from aligncraft.alignment.guidelines import GuidelineBank
    calib_prompt = "Explain why the sky appears blue during daytime."
    GuidelineBank.add_human_override_guideline(
        prompt=calib_prompt,
        constraints=[],
        human_winner="model_b",
        annotator_rationale="Candidate B provides Rayleigh scattering explanation whereas Candidate A is too brief.",
        scenario_focus="Sky Color Audit"
    )

    # Even if LLM erroneously outputs a tie:
    def tie_handler(msg: str) -> str:
        return json.dumps({
            "winner": "tie",
            "margin": "slight",
            "dimension_scores": [
                {"dimension": "factuality", "score_a": 3.0, "score_b": 3.0, "justification": "Equal"}
            ],
            "overall_reasoning": "Both candidates gave similar answers.",
            "detected_errors_model_a": [],
            "detected_errors_model_b": []
        })

    mock_llm = MockChatModel(response_handler=tie_handler)
    judge = PairwiseJudge(mock_llm)
    debiaser = DebiasingJudge(judge)

    sample = PromptSample(id="test-human-calib", prompt=calib_prompt)
    eval_result = debiaser.evaluate_with_debiasing(
        sample=sample,
        response_a="The sky is blue because of scattering.",
        response_b="The sky appears blue because sunlight undergoes Rayleigh scattering by air molecules.",
        use_human_calibration=True,
    )

    # The human audit preferred Candidate B -> calibration must enforce Candidate B!
    assert eval_result.final_consensus_winner == PreferenceChoice.MODEL_B
    assert "Human Ground-Truth Calibrated" in eval_result.summary_critique


def test_calibration_override_preserves_candidate_identity_and_not_in_pairwise():
    """Verify that human calibration tracks candidate identity, not slot position, and evaluate_pair does not hard-override."""
    from aligncraft.alignment.guidelines import GuidelineBank
    prompt = "Identity tracking test prompt"
    response_good = "High quality factual response."
    response_bad = "Low quality hallucinatory response."

    GuidelineBank.add_human_override_guideline(
        prompt=prompt,
        constraints=[],
        human_winner="model_a",
        annotator_rationale="Prefer response_good always.",
        scenario_focus="Identity Test",
        preferred_response=response_good
    )

    def neutral_llm(msg: str) -> str:
        return json.dumps({
            "winner": "tie",
            "margin": "slight",
            "dimension_scores": [
                {"dimension": "factuality", "score_a": 3.0, "score_b": 3.0, "justification": "Tied"}
            ],
            "overall_reasoning": "Neutral model tie.",
            "detected_errors_model_a": [],
            "detected_errors_model_b": []
        })

    mock_llm = MockChatModel(response_handler=neutral_llm)
    judge = PairwiseJudge(mock_llm)
    debiaser = DebiasingJudge(judge)
    sample = PromptSample(id="test-id", prompt=prompt)

    # 1. PairwiseJudge.evaluate_pair in isolation must NOT hard-override winner
    raw_pass = judge.evaluate_pair(sample, response_a=response_bad, response_b=response_good, use_human_calibration=True)
    assert raw_pass.winner == PreferenceChoice.TIE, f"PairwiseJudge should not hardcode winner, got {raw_pass.winner}"

    # 2. Forward order: response_good is in slot A -> final_consensus_winner must be MODEL_A
    eval_fwd = debiaser.evaluate_with_debiasing(sample, response_a=response_good, response_b=response_bad, use_human_calibration=True)
    assert eval_fwd.final_consensus_winner == PreferenceChoice.MODEL_A

    # 3. Reverse order: response_good is in slot B -> final_consensus_winner must be MODEL_B
    eval_rev = debiaser.evaluate_with_debiasing(sample, response_a=response_bad, response_b=response_good, use_human_calibration=True)
    assert eval_rev.final_consensus_winner == PreferenceChoice.MODEL_B

    # 4. Calibration OFF: tie remains tie
    eval_off = debiaser.evaluate_with_debiasing(sample, response_a=response_good, response_b=response_bad, use_human_calibration=False)
    assert eval_off.final_consensus_winner == PreferenceChoice.TIE


def test_pairwise_judge_reverse_pass_does_not_invert_winner():
    """
    Behavioral test for COR-02:
    On the original code, evaluate_pair inspected exact_override, saw human_winner='model_a',
    and unconditionally set winner = MODEL_A, even when called with reversed inputs (where
    slot A was physically the inferior candidate).
    """
    from aligncraft.alignment.guidelines import GuidelineBank
    prompt = "Behavioral COR-02 Test Prompt"
    GuidelineBank.add_human_override_guideline(
        prompt=prompt,
        constraints=[],
        human_winner="model_a",
        annotator_rationale="Candidate A is correct."
    )
    mock_llm = MockChatModel(mock_response=json.dumps({
        "winner": "tie",
        "margin": "slight",
        "dimension_scores": [{"dimension": "factuality", "score_a": 3.0, "score_b": 3.0, "justification": "Tied"}],
        "overall_reasoning": "Tied",
        "detected_errors_model_a": [],
        "detected_errors_model_b": []
    }))
    judge = PairwiseJudge(mock_llm)
    sample = PromptSample(id="cor02-beh", prompt=prompt)

    # In reverse pass: Slot A has inferior Candidate B; Slot B has preferred Candidate A.
    # evaluate_pair MUST NOT report MODEL_A (which would award the win to inferior Candidate B).
    rev_verdict = judge.evaluate_pair(sample, response_a="Inferior Candidate B", response_b="Preferred Candidate A", use_human_calibration=True)
    assert rev_verdict.winner != PreferenceChoice.MODEL_A, (
        f"Inverted reverse pass defect! evaluate_pair hardcoded winner=MODEL_A on reversed slots, "
        f"awarding the win to 'Inferior Candidate B' in slot A!"
    )


def test_identical_responses_with_human_override_evaluates_to_tie():
    """Verify that if both candidates submit identical text, calibration yields TIE rather than preferring slot A."""
    from aligncraft.alignment.guidelines import GuidelineBank
    prompt = "Identical response test prompt"
    identical_text = "The quick brown fox jumps over the lazy dog."

    GuidelineBank.add_human_override_guideline(
        prompt=prompt,
        constraints=[],
        human_winner="model_a",
        annotator_rationale="Prefer model A normally.",
        preferred_response=identical_text
    )

    mock_llm = MockChatModel(mock_response=json.dumps({
        "winner": "tie",
        "margin": "slight",
        "dimension_scores": [{"dimension": "factuality", "score_a": 3.0, "score_b": 3.0, "justification": "Tied"}],
        "overall_reasoning": "Identical content",
        "detected_errors_model_a": [],
        "detected_errors_model_b": []
    }))
    judge = PairwiseJudge(mock_llm)
    debiaser = DebiasingJudge(judge)
    sample = PromptSample(id="identical-test", prompt=prompt)

    result = debiaser.evaluate_with_debiasing(
        sample=sample,
        response_a=identical_text,
        response_b=identical_text,
        use_human_calibration=True,
    )

    assert result.final_consensus_winner == PreferenceChoice.TIE
    assert result.confidence_score == 1.0





