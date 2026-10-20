"""
Tests for Task 2.7: Human override vs deterministic veto conflict detection.
"""

import json
from aligncraft.alignment.guidelines import GuidelineBank
from aligncraft.models.factory import MockChatModel
from aligncraft.judges.debiaser import DebiasingJudge
from aligncraft.judges.pairwise import PairwiseJudge
from aligncraft.schemas.models import PreferenceChoice, PromptSample


def test_human_override_with_veto_conflict_sets_flag_and_reason():
    """
    Task 2.7: If human override selects a candidate that violates a critical veto
    or deterministic hard constraint, the human verdict is still returned, but
    override_conflicts_with_veto must be True and the veto reason documented in summary.
    """
    prompt = "Write a statement without the letter e."
    resp_unsafe_a = "The elephant walked everywhere."  # 5 e's -> hard negative violation!
    resp_safe_b = "A big brown dog runs outdoors."   # clean lipogram

    # Register guideline where human preferred response_a
    GuidelineBank.add_human_override_guideline(
        prompt=prompt,
        constraints=["Must NOT contain the letter e or E"],
        human_winner="model_a",
        annotator_rationale="Annotator liked the elephant imagery despite constraint lapse.",
        scenario_focus="Veto Conflict Test",
        preferred_response=resp_unsafe_a,
        rejected_response=resp_safe_b,
    )

    def neutral_handler(msg: str) -> str:
        return json.dumps({
            "winner": "model_b",
            "margin": "clear",
            "dimension_scores": [
                {"dimension": "instruction_following", "score_a": 2.0, "score_b": 5.0, "justification": "Model B obeyed lipogram."}
            ],
            "overall_reasoning": "Model B obeyed constraints.",
            "detected_errors_model_a": ["negative_constraint_violation"],
            "detected_errors_model_b": []
        })

    mock_llm = MockChatModel(response_handler=neutral_handler)
    judge = PairwiseJudge(mock_llm)
    debiaser = DebiasingJudge(judge)
    sample = PromptSample(
        id="veto-conflict-test",
        prompt=prompt,
        constraints=["Must NOT contain the letter e or E"]
    )

    eval_result = debiaser.evaluate_with_debiasing(
        sample=sample,
        response_a=resp_unsafe_a,
        response_b=resp_safe_b,
        use_human_calibration=True,
    )

    # 1. Human verdict is preserved
    assert eval_result.final_consensus_winner == PreferenceChoice.MODEL_A
    # 2. Conflict flag is explicitly raised
    assert eval_result.override_conflicts_with_veto is True, "Expected override_conflicts_with_veto to be True!"
    # 3. Veto reason is surfaced in summary critique
    assert "Conflict with Critical Veto" in eval_result.summary_critique or "violates critical constraints" in eval_result.summary_critique


def test_human_override_without_conflict_sets_flag_false():
    """
    Task 2.7: When human override selects a candidate with no veto violations,
    override_conflicts_with_veto must be False.
    """
    prompt = "Explain why the sky is blue concisely."
    resp_a = "Rayleigh scattering deflects shorter blue wavelengths across the atmosphere."
    resp_b = "Because the ocean reflects blue up to the clouds."

    GuidelineBank.add_human_override_guideline(
        prompt=prompt,
        constraints=["Be concise"],
        human_winner="model_a",
        annotator_rationale="Scientifically accurate explanation.",
        scenario_focus="Normal Human Override",
        preferred_response=resp_a,
        rejected_response=resp_b,
    )

    def mock_handler(msg: str) -> str:
        return json.dumps({
            "winner": "model_a",
            "margin": "clear",
            "dimension_scores": [
                {"dimension": "factuality", "score_a": 5.0, "score_b": 2.0, "justification": "Rayleigh scattering accurate."}
            ],
            "overall_reasoning": "Model A is accurate.",
            "detected_errors_model_a": [],
            "detected_errors_model_b": []
        })

    mock_llm = MockChatModel(response_handler=mock_handler)
    judge = PairwiseJudge(mock_llm)
    debiaser = DebiasingJudge(judge)
    sample = PromptSample(id="normal-override-test", prompt=prompt)

    eval_result = debiaser.evaluate_with_debiasing(
        sample=sample,
        response_a=resp_a,
        response_b=resp_b,
        use_human_calibration=True,
    )

    assert eval_result.final_consensus_winner == PreferenceChoice.MODEL_A
    assert eval_result.override_conflicts_with_veto is False
    assert "Conflict with Critical Veto" not in eval_result.summary_critique
