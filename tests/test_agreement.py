"""Unit tests for statistical Inter-Annotator Agreement (Cohen's Kappa)."""

import pytest
from aligncraft.schemas.models import (
    DebiasedEvaluation,
    PairwiseVerdict,
    PreferenceChoice,
    PromptSample,
)
from aligncraft.stats.agreement import (
    calculate_cohens_kappa,
    compute_benchmark_summary,
)


def test_cohens_kappa_perfect_agreement():
    rater_a = ["model_a", "model_b", "model_a", "tie", "model_b"]
    rater_b = ["model_a", "model_b", "model_a", "tie", "model_b"]

    kappa, interp = calculate_cohens_kappa(rater_a, rater_b)
    assert pytest.approx(kappa, 0.01) == 1.0
    assert "Almost Perfect" in interp


def test_cohens_kappa_worse_than_chance():
    # Rater A and Rater B systematically invert each other
    rater_a = ["model_a", "model_a", "model_b", "model_b"]
    rater_b = ["model_b", "model_b", "model_a", "model_a"]

    kappa, interp = calculate_cohens_kappa(rater_a, rater_b)
    assert kappa < 0.0
    assert "Poor" in interp


def test_cohens_kappa_length_mismatch():
    with pytest.raises(ValueError):
        calculate_cohens_kappa(["model_a"], ["model_a", "model_b"])


def test_benchmark_summary_aggregation():
    # Construct synthetic DebiasedEvaluations
    dummy_verdict = PairwiseVerdict(
        winner=PreferenceChoice.MODEL_A,
        overall_reasoning="Test reasoning",
        dimension_scores=[],
    )

    eval_1 = DebiasedEvaluation(
        sample_id="s1",
        model_a_name="Model A",
        model_b_name="Model B",
        forward_verdict=dummy_verdict,
        reverse_verdict=dummy_verdict,
        final_consensus_winner=PreferenceChoice.MODEL_A,
        positional_bias_detected=False,
        verbosity_bias_flag=False,
        confidence_score=0.95,
        summary_critique="Clean consensus",
    )

    eval_2 = DebiasedEvaluation(
        sample_id="s2",
        model_a_name="Model A",
        model_b_name="Model B",
        forward_verdict=dummy_verdict,
        reverse_verdict=dummy_verdict,
        final_consensus_winner=PreferenceChoice.INDETERMINATE,
        positional_bias_detected=True,
        verbosity_bias_flag=False,
        confidence_score=0.30,
        summary_critique="Pos bias",
    )

    sample_1 = PromptSample(id="s1", prompt="p1", human_preferred=PreferenceChoice.MODEL_A)
    sample_2 = PromptSample(id="s2", prompt="p2", human_preferred=PreferenceChoice.MODEL_A)
    samples_map = {"s1": sample_1, "s2": sample_2}

    summary = compute_benchmark_summary([eval_1, eval_2], samples_map)

    assert summary.total_samples == 2
    assert summary.model_a_wins == 1
    assert summary.indeterminate == 1
    assert summary.positional_bias_rate == 0.50
