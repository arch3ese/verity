"""
Debiasing Engine for LLM Judges:
Implements Positional Swapping (A/B vs B/A) and Verbosity Bias Detection.
"""

from typing import Tuple
from aligncraft.judges.pairwise import PairwiseJudge
from aligncraft.alignment.guidelines import GuidelineBank
from aligncraft.schemas.models import (
    DebiasedEvaluation,
    PairwiseVerdict,
    PreferenceChoice,
    PromptSample,
    RatingDimension,
)


class DebiasingJudge:
    """
    Wraps a PairwiseJudge to execute dual-order evaluations (A/B and B/A).
    Detects positional preference hallucinations and length/verbosity bias.
    """

    def __init__(self, judge: PairwiseJudge, verbosity_ratio_threshold: float = 2.0):
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
        # Execute Pass 1 (Forward: A vs B) and Pass 2 (Reverse: B vs A) concurrently
        import concurrent.futures
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

        # In reverse pass:
        # If reverse verdict winner is 'model_a', that physically corresponded to response_b!
        # If reverse verdict winner is 'model_b', that physically corresponded to response_a!
        if reverse_raw_verdict.winner == PreferenceChoice.MODEL_A:
            mapped_reverse_winner = PreferenceChoice.MODEL_B
        elif reverse_raw_verdict.winner == PreferenceChoice.MODEL_B:
            mapped_reverse_winner = PreferenceChoice.MODEL_A
        else:
            mapped_reverse_winner = reverse_raw_verdict.winner

        # Check for Positional Bias
        # If the judge consistently voted for whoever was presented first:
        # forward voted model_a (pos 1), reverse voted model_a (pos 1, which is response_b)
        pos_1_favored = (
            forward_verdict.winner == PreferenceChoice.MODEL_A and
            reverse_raw_verdict.winner == PreferenceChoice.MODEL_A
        )
        pos_2_favored = (
            forward_verdict.winner == PreferenceChoice.MODEL_B and
            reverse_raw_verdict.winner == PreferenceChoice.MODEL_B
        )

        positional_bias_detected = pos_1_favored or pos_2_favored

        # Determine Consensus Winner & Confidence
        if forward_verdict.winner == mapped_reverse_winner:
            # High confidence agreement across both presentation orders
            final_winner = forward_verdict.winner
            confidence = 0.95
            summary = f"Consensus achieved: {final_winner.value} won in both presentation orders."
        elif positional_bias_detected:
            # Order mattered completely: judge is susceptible to presentation order
            final_winner = PreferenceChoice.INDETERMINATE
            confidence = 0.30
            summary = (
                "Positional bias detected! The judge favored whichever candidate was "
                "presented first. Result flagged as indeterminate."
            )
        else:
            # Mixed results (e.g. one pass was tie, one was slight win)
            final_winner = (
                forward_verdict.winner if forward_verdict.winner != PreferenceChoice.TIE
                else mapped_reverse_winner
            )
            confidence = 0.65
            summary = (
                f"Minor order divergence: forward chose {forward_verdict.winner.value}, "
                f"reverse mapped chose {mapped_reverse_winner.value}."
            )

        # Check for Verbosity Bias
        verbosity_flag = self._check_verbosity_bias(
            response_a=response_a,
            response_b=response_b,
            winner=final_winner,
            verdict=forward_verdict,
        )

        if verbosity_flag:
            summary += " [Warning: Verbosity bias flagged - winner has excessive length without quality advantage]."

        # Ground-Truth Senior Human Annotator Calibration Override
        if use_human_calibration and sample and sample.prompt:
            exact_override = GuidelineBank.find_exact_prompt_override(sample.prompt)
            if exact_override:
                hw = exact_override.get("human_preferred_winner")
                if hw in ["model_a", "model_b"]:
                    final_winner = PreferenceChoice.MODEL_A if hw == "model_a" else PreferenceChoice.MODEL_B
                    confidence = 0.98
                    positional_bias_detected = False
                    summary = f"Human Ground-Truth Calibrated: Senior annotator preference ({'Candidate A' if hw == 'model_a' else 'Candidate B'}) enforced for this prompt. Rationale: {exact_override.get('annotator_rationale')}"

        # Calculate Tiebreaker Options if Indeterminate
        can_resolve_tiebreaker = False
        tiebreaker_candidate = None
        tiebreaker_delta = None
        tiebreaker_detail = None

        if final_winner == PreferenceChoice.INDETERMINATE:
            weight_map = {
                RatingDimension.SAFETY_HARM: 3.0,
                RatingDimension.INSTRUCTION_FOLLOWING: 2.5,
                RatingDimension.FACTUALITY: 2.0,
                RatingDimension.REASONING_LOGIC: 1.5,
                RatingDimension.STYLE_CONCISENESS: 1.0,
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
        )

    def _check_verbosity_bias(
        self,
        response_a: str,
        response_b: str,
        winner: PreferenceChoice,
        verdict: PairwiseVerdict,
    ) -> bool:
        """
        Flags if a response won primarily due to inflated length rather than factual superiority.
        """
        len_a = len(response_a)
        len_b = len(response_b)

        if winner == PreferenceChoice.MODEL_A and len_b > 0:
            ratio = len_a / len_b
            if ratio >= self.verbosity_ratio_threshold:
                # Check if instruction following or factuality was strictly better
                fact_score_a = self._get_dim_score(verdict, RatingDimension.FACTUALITY, True)
                fact_score_b = self._get_dim_score(verdict, RatingDimension.FACTUALITY, False)
                if fact_score_a <= fact_score_b:
                    return True

        elif winner == PreferenceChoice.MODEL_B and len_a > 0:
            ratio = len_b / len_a
            if ratio >= self.verbosity_ratio_threshold:
                fact_score_a = self._get_dim_score(verdict, RatingDimension.FACTUALITY, True)
                fact_score_b = self._get_dim_score(verdict, RatingDimension.FACTUALITY, False)
                if fact_score_b <= fact_score_a:
                    return True

        return False

    def _get_dim_score(self, verdict: PairwiseVerdict, dimension: RatingDimension, for_model_a: bool) -> float:
        for ds in verdict.dimension_scores:
            if ds.dimension == dimension:
                return ds.score_a if for_model_a else ds.score_b
        return 3.0
