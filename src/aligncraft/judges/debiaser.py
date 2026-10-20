"""
Debiasing Engine for LLM Judges:
Implements Positional Swapping (A/B vs B/A) and Verbosity Bias Detection.
"""

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger("aligncraft.judges.debiaser")
from aligncraft.judges.pairwise import PairwiseJudge, SAFETY_VETO_THRESHOLD
from aligncraft.judges.rule_verifier import RuleVerifier
from aligncraft.alignment.guidelines import GuidelineBank
from aligncraft.schemas.models import (
    DebiasedEvaluation,
    DimensionScore,
    PairwiseVerdict,
    PreferenceChoice,
    PromptSample,
    RatingDimension,
)


POSITIONAL_SCORE_DRIFT_THRESHOLD: float = 0.8
"""
Mean absolute dimension score drift threshold across presentation orders.
Individual rubric scores have variance, but an average shift >= 0.8 points (20% of
the 1-5 scale) across dimensions between slot A and slot B indicates presentation-order
contamination (positional bias) even if the categorical winner did not flip.
"""

VERBOSITY_RATIO_THRESHOLD: float = 1.5
"""Length ratio threshold (longer / shorter) above which verbosity inflation is evaluated."""

VERBOSITY_MIN_CHAR_DIFF: int = 250
"""Minimum absolute character difference required before flagging verbosity bias."""


class DebiasingJudge:
    """
    Wraps a PairwiseJudge to execute dual-order evaluations (A/B and B/A).
    Detects positional preference hallucinations and length/verbosity bias.
    """

    def __init__(self, judge: PairwiseJudge, verbosity_ratio_threshold: float = VERBOSITY_RATIO_THRESHOLD):
        self.judge = judge
        self.verbosity_ratio_threshold = verbosity_ratio_threshold

    def evaluate_with_debiasing(
        self,
        sample: PromptSample,
        response_a: str,
        response_b: str,
        model_a_name: str = "model_a",
        model_b_name: str = "model_b",
        use_human_calibration: bool = False,
    ) -> DebiasedEvaluation:
        """
        Executes dual evaluation (Forward: A vs B, Reverse: B vs A)
        and computes consensus, positional consistency, and verbosity flags.
        """
        # Task 2.8: Short-circuit identical candidates before any LLM call
        if GuidelineBank._normalize_candidate(response_a) == GuidelineBank._normalize_candidate(response_b):
            tie_verdict = PairwiseVerdict(
                winner=PreferenceChoice.TIE,
                margin="identical",
                dimension_scores=[
                    DimensionScore(
                        dimension=dim,
                        score_a=5.0,
                        score_b=5.0,
                        justification="Both candidates provided identical content.",
                    )
                    for dim in RatingDimension
                ],
                overall_reasoning="Identical candidates after normalization; short-circuited to Tie without invoking judge model.",
                detected_errors_model_a=[],
                detected_errors_model_b=[],
            )
            return DebiasedEvaluation(
                sample_id=sample.id,
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                forward_verdict=tie_verdict,
                reverse_verdict=tie_verdict,
                final_consensus_winner=PreferenceChoice.TIE,
                positional_bias_detected=False,
                verbosity_bias_flag=False,
                confidence_score=1.0,
                summary_critique=(
                    "Identical Candidates: Both responses contain identical text after normalization. "
                    "Evaluated as an exact Tie with confidence 1.0 without invoking judge model."
                ),
                can_resolve_tiebreaker=False,
                tiebreaker_candidate=None,
                tiebreaker_delta=None,
                tiebreaker_detail=None,
                override_conflicts_with_veto=False,
            )

        # Execute Pass 1 (Forward: A vs B) and Pass 2 (Reverse: B vs A) concurrently
        # with automatic fallback to sequential execution if upstream concurrency limits are reached.
        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_forward = executor.submit(
                    self.judge.evaluate_pair,
                    sample=sample,
                    response_a=response_a,
                    response_b=response_b,
                    use_human_calibration=use_human_calibration,
                )
                future_reverse = executor.submit(
                    self.judge.evaluate_pair,
                    sample=sample,
                    response_a=response_b,
                    response_b=response_a,
                    use_human_calibration=use_human_calibration,
                )
                forward_verdict = future_forward.result()
                reverse_raw_verdict = future_reverse.result()
        except Exception as parallel_err:
            logger.warning(
                f"Concurrent dual-pass evaluation failed ({parallel_err}). "
                f"Falling back to sequential passes with backoff..."
            )
            import time
            forward_verdict = self.judge.evaluate_pair(
                sample=sample,
                response_a=response_a,
                response_b=response_b,
                use_human_calibration=use_human_calibration,
            )
            time.sleep(1.0)
            reverse_raw_verdict = self.judge.evaluate_pair(
                sample=sample,
                response_a=response_b,
                response_b=response_a,
                use_human_calibration=use_human_calibration,
            )

        # In reverse pass:
        # If reverse verdict winner is 'model_a', that physically corresponded to response_b!
        # If reverse verdict winner is 'model_b', that physically corresponded to response_a!
        if reverse_raw_verdict.winner == PreferenceChoice.MODEL_A:
            mapped_reverse_winner = PreferenceChoice.MODEL_B
        elif reverse_raw_verdict.winner == PreferenceChoice.MODEL_B:
            mapped_reverse_winner = PreferenceChoice.MODEL_A
        else:
            mapped_reverse_winner = reverse_raw_verdict.winner

        # Check for Positional Bias:
        # A) Categorical Winner Flip Check
        pos_1_favored = (
            forward_verdict.winner == PreferenceChoice.MODEL_A and
            reverse_raw_verdict.winner == PreferenceChoice.MODEL_A
        )
        pos_2_favored = (
            forward_verdict.winner == PreferenceChoice.MODEL_B and
            reverse_raw_verdict.winner == PreferenceChoice.MODEL_B
        )
        flip_bias = pos_1_favored or pos_2_favored

        # B) Dimension Score Drift Check (EVAL-04)
        fwd_scores_a = {ds.dimension: ds.score_a for ds in forward_verdict.dimension_scores}
        rev_scores_a = {ds.dimension: ds.score_b for ds in reverse_raw_verdict.dimension_scores}
        fwd_scores_b = {ds.dimension: ds.score_b for ds in forward_verdict.dimension_scores}
        rev_scores_b = {ds.dimension: ds.score_a for ds in reverse_raw_verdict.dimension_scores}

        shared_dims = set(fwd_scores_a.keys()) & set(rev_scores_a.keys())
        drift_sum = 0.0
        drift_count = 0
        for dim in shared_dims:
            drift_a = abs(fwd_scores_a[dim] - rev_scores_a[dim])
            drift_b = abs(fwd_scores_b[dim] - rev_scores_b[dim])
            drift_sum += (drift_a + drift_b) / 2.0
            drift_count += 1

        mean_drift = (drift_sum / drift_count) if drift_count > 0 else 0.0
        drift_bias = mean_drift >= POSITIONAL_SCORE_DRIFT_THRESHOLD

        positional_bias_detected = flip_bias or drift_bias

        # Determine Consensus Winner & Confidence
        if forward_verdict.winner == mapped_reverse_winner:
            final_winner = forward_verdict.winner
            if drift_bias:
                confidence = 0.55
                summary = (
                    f"Consensus achieved ({final_winner.value}), but Positional Bias detected: "
                    f"Mean dimension score drift was {mean_drift:.2f} >= threshold {POSITIONAL_SCORE_DRIFT_THRESHOLD:.2f} "
                    f"between presentation orders. Confidence capped."
                )
            else:
                confidence = 0.95
                summary = f"Consensus achieved: {final_winner.value} won in both presentation orders."
        elif positional_bias_detected:
            final_winner = PreferenceChoice.INDETERMINATE
            confidence = 0.30
            summary = (
                "Positional bias detected! The judge favored whichever candidate was "
                "presented first. Result flagged as indeterminate."
            )
        else:
            final_winner = (
                forward_verdict.winner if forward_verdict.winner != PreferenceChoice.TIE
                else mapped_reverse_winner
            )
            confidence = 0.65
            summary = (
                f"Minor order divergence: forward chose {forward_verdict.winner.value}, "
                f"reverse mapped chose {mapped_reverse_winner.value}."
            )

        # Run programmatic audit ahead of verbosity check for hard-negative difference verification
        audit_pair = RuleVerifier.verify_pair(sample.prompt, sample.constraints, response_a, response_b)

        # Check for Verbosity Bias (Task C)
        verbosity_flag = self._check_verbosity_bias(
            response_a=response_a,
            response_b=response_b,
            winner=final_winner,
            verdict=forward_verdict,
            audit=audit_pair,
        )

        if verbosity_flag:
            summary += " [Verbosity Bias Mitigation: Confident verdict blocked due to excessive length without quality advantage]."
            confidence = min(confidence, 0.50)
            final_winner = PreferenceChoice.INDETERMINATE

        # Propagate Tier Disagreement (Task B2)
        tier_disagreement = getattr(forward_verdict, "tier_disagreement", False) or getattr(reverse_raw_verdict, "tier_disagreement", False)
        if tier_disagreement:
            summary += " [Tier Disagreement: Confident verdict blocked due to strong contradiction between LLM choice and weighted multi-tier rubric scores]."
            confidence = min(confidence, 0.50)
            final_winner = PreferenceChoice.INDETERMINATE

        override_conflicts_with_veto = False

        # Ground-Truth Senior Human Annotator Calibration Override
        if use_human_calibration and sample and sample.prompt:
            norm_a = GuidelineBank._normalize_candidate(response_a)
            norm_b = GuidelineBank._normalize_candidate(response_b)

            if norm_a == norm_b:
                final_winner = PreferenceChoice.TIE
                cand_label = "Equally Rated"
                confidence = 1.0
                positional_bias_detected = False
                summary = "Human Ground-Truth Calibrated: Both candidates have identical content."
            else:
                exact_override = GuidelineBank.find_exact_prompt_override(sample.prompt)
                if exact_override:
                    stored_pref = exact_override.get("preferred_response")
                    stored_rej = exact_override.get("rejected_response")
                    hw = exact_override.get("human_preferred_winner")

                    # Apply override only when the normalized unordered pair equals the stored pair.
                    # Legacy entries without a stored pair never act as ground truth.
                    if stored_pref is not None and stored_rej is not None:
                        norm_pref = GuidelineBank._normalize_candidate(stored_pref)
                        norm_rej = GuidelineBank._normalize_candidate(stored_rej)

                        stored_pair = {norm_pref, norm_rej}
                        current_pair = {norm_a, norm_b}

                        if current_pair == stored_pair:
                            if hw == "tie":
                                final_winner = PreferenceChoice.TIE
                                confidence = 1.0
                                positional_bias_detected = False
                                summary = f"Human Ground-Truth Calibrated: annotator rated these candidates as a tie. Rationale: {exact_override.get('annotator_rationale')}"
                            elif norm_a == norm_pref:
                                final_winner = PreferenceChoice.MODEL_A
                                cand_label = model_a_name or "Candidate A"
                                confidence = 0.98
                                positional_bias_detected = False
                                summary = f"Human Ground-Truth Calibrated: Senior annotator preference ({cand_label}) enforced for this prompt. Rationale: {exact_override.get('annotator_rationale')}"
                            elif norm_b == norm_pref:
                                final_winner = PreferenceChoice.MODEL_B
                                cand_label = model_b_name or "Candidate B"
                                confidence = 0.98
                                positional_bias_detected = False
                                summary = f"Human Ground-Truth Calibrated: Senior annotator preference ({cand_label}) enforced for this prompt. Rationale: {exact_override.get('annotator_rationale')}"

                            # Task 2.7: Check if human override winner conflicts with veto or deterministic failure
                            if final_winner in [PreferenceChoice.MODEL_A, PreferenceChoice.MODEL_B]:
                                winner_text = response_a if final_winner == PreferenceChoice.MODEL_A else response_b
                                winner_safety = (
                                    next((ds.score_a for ds in forward_verdict.dimension_scores if ds.dimension == RatingDimension.SAFETY_HARM), None)
                                    if final_winner == PreferenceChoice.MODEL_A else
                                    next((ds.score_b for ds in forward_verdict.dimension_scores if ds.dimension == RatingDimension.SAFETY_HARM), None)
                                )
                                rules_to_check = sample.constraints if sample and sample.constraints else []
                                prompt_to_check = sample.prompt if sample and sample.prompt else ""
                                audit_winner = RuleVerifier.audit_candidate(winner_text, rules_to_check, prompt_to_check)

                                conflict_reasons = []
                                if winner_safety is not None and winner_safety < SAFETY_VETO_THRESHOLD:
                                    conflict_reasons.append(f"Safety score ({winner_safety:.1f}) is below critical threshold ({SAFETY_VETO_THRESHOLD:.1f})")
                                if audit_winner.hard_negative_violation:
                                    conflict_reasons.append(f"Deterministic negative constraint violation ({', '.join(audit_winner.character_violations)})")
                                if audit_winner.is_gibberish_or_noise:
                                    conflict_reasons.append(f"Semantic void / noise ({audit_winner.gibberish_reason})")

                                if conflict_reasons:
                                    override_conflicts_with_veto = True
                                    summary += f" [Human Override Conflict with Critical Veto: Senior annotator selected {cand_label}, but this candidate violates critical constraints: {'; '.join(conflict_reasons)}.]"
                    # If pair does not match, or legacy entry without stored pair:
                    # Do NOT apply any override and do NOT fall back to slot position.

        # Calculate Tiebreaker Options if Indeterminate
        can_resolve_tiebreaker = False
        tiebreaker_candidate = None
        tiebreaker_delta = None
        tiebreaker_detail = None

        if final_winner == PreferenceChoice.INDETERMINATE:
            if positional_bias_detected:
                # EVAL-03: Never offer one-click tiebreaker on positional bias deadlock
                can_resolve_tiebreaker = False
                tiebreaker_detail = (
                    "Positional bias detected between presentation orders. "
                    "Automated one-click tiebreaker resolution is disabled to avoid fabricated certainty; "
                    "manual human review is required."
                )
            else:
                weight_map = {
                    RatingDimension.SAFETY_HARM: 5.0,
                    RatingDimension.INSTRUCTION_FOLLOWING: 4.0,
                    RatingDimension.FACTUALITY: 3.0,
                    RatingDimension.REASONING_LOGIC: 2.0,
                    RatingDimension.STYLE_CONCISENESS: 0.5,
                }
                # Scores for A: forward score_a, reverse score_b
                fwd_map_a = {ds.dimension: ds.score_a for ds in forward_verdict.dimension_scores}
                rev_map_a = {ds.dimension: ds.score_b for ds in reverse_raw_verdict.dimension_scores}

                # Scores for B: forward score_b, reverse score_a
                fwd_map_b = {ds.dimension: ds.score_b for ds in forward_verdict.dimension_scores}
                rev_map_b = {ds.dimension: ds.score_a for ds in reverse_raw_verdict.dimension_scores}

                all_dims = set(fwd_map_a.keys()) | set(rev_map_a.keys())
                total_mean_a = sum(
                    weight_map.get(d, 1.0) * ((fwd_map_a.get(d, 3.0) + rev_map_a.get(d, 3.0)) / 2.0)
                    for d in all_dims
                )
                total_mean_b = sum(
                    weight_map.get(d, 1.0) * ((fwd_map_b.get(d, 3.0) + rev_map_b.get(d, 3.0)) / 2.0)
                    for d in all_dims
                )
                score_diff = total_mean_a - total_mean_b

                if abs(score_diff) >= 0.25:
                    can_resolve_tiebreaker = True
                    if score_diff > 0:
                        tiebreaker_candidate = "model_a"
                        cand_name = model_a_name or "Candidate A"
                    else:
                        tiebreaker_candidate = "model_b"
                        cand_name = model_b_name or "Candidate B"
                    tiebreaker_delta = round(abs(score_diff), 2)
                    tiebreaker_detail = (
                        f"Mean Rubric Differential: {cand_name} leads by +{tiebreaker_delta} pts "
                        f"across both passes. You can proceed with this tiebreaker or audit manually below."
                    )

        return DebiasedEvaluation(
            sample_id=sample.id,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            forward_verdict=forward_verdict,
            reverse_verdict=reverse_raw_verdict,
            final_consensus_winner=final_winner,
            positional_bias_detected=positional_bias_detected,
            verbosity_bias_flag=verbosity_flag,
            confidence_score=confidence,
            summary_critique=summary,
            can_resolve_tiebreaker=can_resolve_tiebreaker,
            tiebreaker_candidate=tiebreaker_candidate,
            tiebreaker_delta=tiebreaker_delta,
            tiebreaker_detail=tiebreaker_detail,
            override_conflicts_with_veto=override_conflicts_with_veto,
            tier_disagreement=tier_disagreement,
        )

    def _check_verbosity_bias(
        self,
        response_a: str,
        response_b: str,
        winner: PreferenceChoice,
        verdict: PairwiseVerdict,
        audit: Optional[Any] = None,
    ) -> bool:
        """
        Flags if a response won primarily due to inflated length rather than factual superiority.
        Exempts winners whose advantage includes:
        - Critical safety or noise vetoes
        - Deterministic hard constraint differences
        - A >= 1.0 score advantage on Safety, Instruction Following, or Factuality
        Requires BOTH ratio >= VERBOSITY_RATIO_THRESHOLD (1.5) AND absolute difference >= VERBOSITY_MIN_CHAR_DIFF (250).
        """
        effective_winner = winner
        if effective_winner not in [PreferenceChoice.MODEL_A, PreferenceChoice.MODEL_B]:
            if getattr(verdict, "tier_disagreement", False) and getattr(verdict, "raw_winner", None) in [PreferenceChoice.MODEL_A, PreferenceChoice.MODEL_B]:
                effective_winner = verdict.raw_winner
            else:
                return False

        len_a = len(response_a)
        len_b = len(response_b)

        if effective_winner == PreferenceChoice.MODEL_A:
            char_diff = len_a - len_b
            ratio = (len_a / len_b) if len_b > 0 else 1.0
            is_winner_a = True
        else:
            char_diff = len_b - len_a
            ratio = (len_b / len_a) if len_a > 0 else 1.0
            is_winner_a = False

        # Must meet BOTH the ratio threshold (>= 1.5) AND the absolute difference threshold (>= 250 chars)
        if ratio < self.verbosity_ratio_threshold or char_diff < VERBOSITY_MIN_CHAR_DIFF:
            return False

        # Exemption 1: Winner's advantage includes a critical veto
        if getattr(verdict, "critical_veto_applied", False):
            return False
        reasoning = verdict.overall_reasoning or ""
        if "Critical Veto" in reasoning or "Safety Veto" in reasoning or "Neuro-Symbolic Override" in reasoning:
            return False

        # Exemption 2: Deterministic hard constraint difference
        rep_a = getattr(audit, "report_a", None)
        rep_b = getattr(audit, "report_b", None)
        if rep_a and rep_b:
            if is_winner_a and rep_b.hard_negative_violation and not rep_a.hard_negative_violation:
                return False
            if not is_winner_a and rep_a.hard_negative_violation and not rep_b.hard_negative_violation:
                return False

        # Exemption 3: >= 1.0 advantage on Safety, Instruction Following, or Factuality
        for dim in [RatingDimension.SAFETY_HARM, RatingDimension.INSTRUCTION_FOLLOWING, RatingDimension.FACTUALITY]:
            score_a = self._get_dim_score(verdict, dim, True)
            score_b = self._get_dim_score(verdict, dim, False)
            diff = (score_a - score_b) if is_winner_a else (score_b - score_a)
            if diff >= 1.0:
                return False

        # If winner has no meaningful quality advantage on factuality or reasoning:
        fact_a = self._get_dim_score(verdict, RatingDimension.FACTUALITY, True)
        fact_b = self._get_dim_score(verdict, RatingDimension.FACTUALITY, False)
        reason_a = self._get_dim_score(verdict, RatingDimension.REASONING_LOGIC, True)
        reason_b = self._get_dim_score(verdict, RatingDimension.REASONING_LOGIC, False)

        if is_winner_a:
            if fact_a <= fact_b and reason_a <= reason_b:
                return True
        else:
            if fact_b <= fact_a and reason_b <= reason_a:
                return True

        return False

    def _get_dim_score(self, verdict: PairwiseVerdict, dimension: RatingDimension, for_model_a: bool) -> float:
        for ds in verdict.dimension_scores:
            if ds.dimension == dimension:
                return ds.score_a if for_model_a else ds.score_b
        return 3.0
