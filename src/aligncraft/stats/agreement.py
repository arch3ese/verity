"""
Statistical Inter-Annotator Agreement (IAA) and Evaluation Analytics.
Calculates Cohen's Kappa, Win Rates, and Bias Rates to benchmark LLM Judges
against Human Ground Truth or between two different LLM Judges.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from aligncraft.schemas.models import (
    BenchmarkSummary,
    DebiasedEvaluation,
    PreferenceChoice,
    PromptSample,
)


def calculate_cohens_kappa(
    rater_a_labels: List[str],
    rater_b_labels: List[str],
) -> Tuple[float, str]:
    """
    Computes Cohen's Kappa coefficient between two sets of categorical ratings.
    Returns (kappa_score, qualitative_interpretation).
    """
    if len(rater_a_labels) != len(rater_b_labels):
        raise ValueError("Both rating lists must have the same length.")

    if not rater_a_labels:
        return 0.0, "No data"

    categories = sorted(list(set(rater_a_labels) | set(rater_b_labels)))
    n = len(rater_a_labels)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    k = len(categories)

    # Build confusion matrix
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(rater_a_labels, rater_b_labels):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1

    # Observed agreement (Po)
    po = np.trace(matrix) / n

    # Expected agreement by chance (Pe)
    row_sums = np.sum(matrix, axis=1) / n
    col_sums = np.sum(matrix, axis=0) / n
    pe = np.sum(row_sums * col_sums)

    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    # Landis & Koch (1977) interpretation
    if kappa < 0.0:
        interp = "Poor (Worse than chance)"
    elif kappa <= 0.20:
        interp = "Slight Agreement"
    elif kappa <= 0.40:
        interp = "Fair Agreement"
    elif kappa <= 0.60:
        interp = "Moderate Agreement"
    elif kappa <= 0.80:
        interp = "Substantial Agreement"
    else:
        interp = "Almost Perfect Agreement"

    return float(kappa), interp


def compute_benchmark_summary(
    evaluations: List[DebiasedEvaluation],
    samples_by_id: Optional[Dict[str, PromptSample]] = None,
) -> BenchmarkSummary:
    """
    Aggregates a collection of DebiasedEvaluation results into a BenchmarkSummary,
    including win rates, bias rates, average dimension scores, and Kappa vs human truth.
    """
    total = len(evaluations)
    if total == 0:
        return BenchmarkSummary(
            total_samples=0,
            model_a_wins=0,
            model_b_wins=0,
            ties=0,
            indeterminate=0,
            positional_bias_rate=0.0,
            verbosity_bias_rate=0.0,
        )

    wins_a = 0
    wins_b = 0
    ties = 0
    indet = 0
    pos_bias_count = 0
    verb_bias_count = 0

    dim_scores_a: Dict[str, List[float]] = {}
    dim_scores_b: Dict[str, List[float]] = {}

    judge_labels: List[str] = []
    human_labels: List[str] = []

    for ev in evaluations:
        winner = ev.final_consensus_winner
        if winner == PreferenceChoice.MODEL_A:
            wins_a += 1
        elif winner == PreferenceChoice.MODEL_B:
            wins_b += 1
        elif winner == PreferenceChoice.TIE:
            ties += 1
        else:
            indet += 1

        if ev.positional_bias_detected:
            pos_bias_count += 1
        if ev.verbosity_bias_flag:
            verb_bias_count += 1

        # Collect dimension scores from forward pass
        for ds in ev.forward_verdict.dimension_scores:
            dim_key = ds.dimension.value
            dim_scores_a.setdefault(dim_key, []).append(ds.score_a)
            dim_scores_b.setdefault(dim_key, []).append(ds.score_b)

        # Check against human ground truth if sample is provided
        if samples_by_id and ev.sample_id in samples_by_id:
            sample = samples_by_id[ev.sample_id]
            if sample.human_preferred:
                judge_labels.append(ev.final_consensus_winner.value)
                human_labels.append(sample.human_preferred.value)

    # Average dimension scores
    avg_scores_a = {k: float(np.mean(v)) for k, v in dim_scores_a.items()}
    avg_scores_b = {k: float(np.mean(v)) for k, v in dim_scores_b.items()}

    # Compute Kappa if human labels exist
    kappa_val = None
    if len(human_labels) >= 2:
        k_score, _ = calculate_cohens_kappa(judge_labels, human_labels)
        kappa_val = k_score

    return BenchmarkSummary(
        total_samples=total,
        model_a_wins=wins_a,
        model_b_wins=wins_b,
        ties=ties,
        indeterminate=indet,
        positional_bias_rate=float(pos_bias_count / total),
        verbosity_bias_rate=float(verb_bias_count / total),
        cohens_kappa_vs_human=kappa_val,
        average_dimension_scores_model_a=avg_scores_a,
        average_dimension_scores_model_b=avg_scores_b,
    )
