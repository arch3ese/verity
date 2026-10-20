"""Tests for STAT-01 and STAT-02 agreement metric fixes."""

import math
import pytest
from aligncraft.stats.agreement import (
    MIN_SAMPLE_SIZE,
    calculate_cohens_kappa,
)


def test_zero_variance_returns_nan_undefined():
    """Verify zero-variance ratings return NaN and undefined interpretation (STAT-01)."""
    rater_a = ["model_a", "model_a", "model_a", "model_a", "model_a"]
    rater_b = ["model_a", "model_a", "model_a", "model_a", "model_a"]

    kappa, interp = calculate_cohens_kappa(rater_a, rater_b)
    assert math.isnan(kappa), f"Expected NaN for zero variance, got {kappa}"
    assert "undefined" in interp.lower(), f"Expected 'undefined' in interp, got '{interp}'"


def test_low_sample_size_warning():
    """Verify sample size below MIN_SAMPLE_SIZE reports an explicit warning (STAT-02)."""
    assert MIN_SAMPLE_SIZE == 10
    rater_a = ["model_a", "model_b", "model_a", "model_b"]  # length 4 < 10
    rater_b = ["model_a", "model_b", "model_b", "model_a"]

    with pytest.warns(UserWarning, match=r"[Ll]ow sample"):
        kappa, interp = calculate_cohens_kappa(rater_a, rater_b)

    assert "warning" in interp.lower()
    assert f"<{MIN_SAMPLE_SIZE}" in interp or f"< {MIN_SAMPLE_SIZE}" in interp


def test_weighted_cohens_kappa_linear_and_quadratic_vs_scikit_learn():
    """Verify linear and quadratic weighted Cohen's Kappa against scikit-learn numbers (STAT-02)."""
    from sklearn.metrics import cohen_kappa_score

    # Hand-checked case 1: Standard ordinal 1-5 rubric scores
    r1_a = [1, 2, 3, 4, 5, 2, 3, 4, 1, 5]
    r2_a = [1, 2, 4, 4, 4, 2, 3, 5, 1, 5]

    k_lin_1, _ = calculate_cohens_kappa(r1_a, r2_a, weights="linear")
    sk_lin_1 = cohen_kappa_score(r1_a, r2_a, weights="linear")
    assert pytest.approx(k_lin_1, 1e-6) == sk_lin_1

    k_quad_1, _ = calculate_cohens_kappa(r1_a, r2_a, weights="quadratic")
    sk_quad_1 = cohen_kappa_score(r1_a, r2_a, weights="quadratic")
    assert pytest.approx(k_quad_1, 1e-6) == sk_quad_1

    # Hand-checked case 2: Extreme distance disagreements
    r1_b = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    r2_b = [1, 1, 5, 2, 3, 3, 2, 4, 1, 5]

    k_lin_2, _ = calculate_cohens_kappa(r1_b, r2_b, weights="linear")
    sk_lin_2 = cohen_kappa_score(r1_b, r2_b, weights="linear")
    assert pytest.approx(k_lin_2, 1e-6) == sk_lin_2

    k_quad_2, _ = calculate_cohens_kappa(r1_b, r2_b, weights="quadratic")
    sk_quad_2 = cohen_kappa_score(r1_b, r2_b, weights="quadratic")
    assert pytest.approx(k_quad_2, 1e-6) == sk_quad_2

    # Hand-checked case 3: Complete ordinal agreement
    r1_c = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    r2_c = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

    k_lin_3, _ = calculate_cohens_kappa(r1_c, r2_c, weights="linear")
    sk_lin_3 = cohen_kappa_score(r1_c, r2_c, weights="linear")
    assert pytest.approx(k_lin_3, 1e-6) == sk_lin_3
    assert k_lin_3 == 1.0

    k_quad_3, _ = calculate_cohens_kappa(r1_c, r2_c, weights="quadratic")
    sk_quad_3 = cohen_kappa_score(r1_c, r2_c, weights="quadratic")
    assert pytest.approx(k_quad_3, 1e-6) == sk_quad_3
    assert k_quad_3 == 1.0
