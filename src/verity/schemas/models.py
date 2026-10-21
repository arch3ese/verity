"""
Core data schemas and Pydantic models for RLHF, RLAIF, and AI Training evaluation.
"""

import os
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

MAX_PROMPT_LENGTH: int = int(os.getenv("ALIGN_MAX_PROMPT_LENGTH", "50000"))
MAX_RESPONSE_LENGTH: int = int(os.getenv("ALIGN_MAX_RESPONSE_LENGTH", "100000"))


class RatingDimension(str, Enum):
    """Core evaluation dimensions modeled after enterprise annotation rubrics."""
    INSTRUCTION_FOLLOWING = "instruction_following"
    FACTUALITY = "factuality"
    REASONING_LOGIC = "reasoning_logic"
    SAFETY_HARM = "safety_harm"
    STYLE_CONCISENESS = "style_conciseness"


class ErrorTag(str, Enum):
    """Fine-grained error taxonomy for failure mode classification."""
    NEGATIVE_CONSTRAINT_VIOLATION = "negative_constraint_violation"
    FORMATTING_VIOLATION = "formatting_violation"
    FACTUAL_HALLUCINATION = "factual_hallucination"
    LOGICAL_FALLACY = "logical_fallacy"
    SYCOPHANCY = "sycophancy"
    REPETITION = "repetition"
    SAFETY_VIOLATION = "safety_violation"
    OVER_REFUSAL = "over_refusal"
    SEMANTIC_VOID = "semantic_void"


class PreferenceChoice(str, Enum):
    """Verdict on pairwise model comparison."""
    MODEL_A = "model_a"
    MODEL_B = "model_b"
    TIE = "tie"
    INDETERMINATE = "indeterminate"


class PromptSample(BaseModel):
    """A benchmark test case or evaluation sample."""
    id: str
    prompt: str = Field(..., max_length=MAX_PROMPT_LENGTH)
    system_prompt: Optional[str] = None
    constraints: List[str] = Field(default_factory=list, description="Explicit hard or negative constraints")
    reference_ground_truth: Optional[str] = None
    human_preferred: Optional[PreferenceChoice] = Field(
        default=None,
        description="Ground truth human annotator label for measuring Inter-Annotator Agreement (IAA)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelOutput(BaseModel):
    """A generated completion from a candidate model under test."""
    model_name: str
    response_text: str
    latency_ms: Optional[float] = None
    token_count: Optional[int] = None


class DimensionScore(BaseModel):
    """Score on a specific evaluation dimension (1.0 - 5.0 scale)."""
    dimension: RatingDimension
    score_a: float = Field(ge=1.0, le=5.0, description="Model A score on 1-5 scale")
    score_b: float = Field(ge=1.0, le=5.0, description="Model B score on 1-5 scale")
    justification: str = Field(description="Step-by-step reasoning supporting this score")


class PairwiseVerdict(BaseModel):
    """Verdict from a single pairwise evaluation pass."""
    winner: PreferenceChoice
    margin: str = Field(
        default="clear",
        description="Magnitude of preference: 'slight', 'moderate', 'clear', 'decisive'"
    )
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    overall_reasoning: str
    detected_errors_model_a: List[ErrorTag] = Field(default_factory=list)
    detected_errors_model_b: List[ErrorTag] = Field(default_factory=list)
    inconsistency_detected: bool = Field(
        default=False,
        description="True if judge scores contradict verified programmatic telemetry"
    )
    tier_disagreement: bool = Field(
        default=False,
        description="True if LLM categorical choice strongly contradicted weighted multi-tier rubric scores"
    )
    raw_winner: Optional[PreferenceChoice] = Field(
        default=None,
        description="Raw categorical winner chosen by LLM before guardrails or downgrades"
    )
    critical_veto_applied: bool = Field(
        default=False,
        description="True if a Tier 1 Safety or Tier 2 Hard Negative critical veto was applied to disqualify a candidate"
    )


class DebiasedEvaluation(BaseModel):
    """
    Composite evaluation with positional debiasing and verbosity checks.
    Evaluates order A vs B and reversed B vs A to catch order hallucination.
    """
    sample_id: str
    model_a_name: str
    model_b_name: str
    forward_verdict: PairwiseVerdict
    reverse_verdict: PairwiseVerdict
    final_consensus_winner: PreferenceChoice
    positional_bias_detected: bool = Field(
        description="True if judge preferred whichever candidate was presented first"
    )
    verbosity_bias_flag: bool = Field(
        default=False,
        description="True if winner is significantly more verbose without higher quality"
    )
    inconsistency_detected: bool = Field(
        default=False,
        description="True if LLM scores contradicted deterministic facts in either pass"
    )
    override_conflicts_with_veto: bool = Field(
        default=False,
        description="True if human ground-truth override favored a candidate that triggered a critical veto"
    )
    tier_disagreement: bool = Field(
        default=False,
        description="True if either pass detected a tier disagreement between LLM verdict and weighted rubric"
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    summary_critique: str
    can_resolve_tiebreaker: bool = Field(default=False)
    tiebreaker_candidate: Optional[str] = Field(default=None)
    tiebreaker_delta: Optional[float] = Field(default=None)
    tiebreaker_detail: Optional[str] = Field(default=None)


class DPOTriplet(BaseModel):
    """Direct Preference Optimization (DPO) triplet formatted for HuggingFace TRL."""
    prompt: str
    chosen: str
    rejected: str
    margin: str
    justification: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkSummary(BaseModel):
    """Aggregate statistics for an entire benchmark evaluation run."""
    total_samples: int
    model_a_wins: int
    model_b_wins: int
    ties: int
    indeterminate: int
    positional_bias_rate: float = Field(description="Percentage of evaluations with positional inconsistency")
    verbosity_bias_rate: float
    cohens_kappa_vs_human: Optional[float] = Field(
        default=None,
        description="Cohen's Kappa agreement score between LLM Judge and Human Annotator"
    )
    average_dimension_scores_model_a: Dict[str, float] = Field(default_factory=dict)
    average_dimension_scores_model_b: Dict[str, float] = Field(default_factory=dict)


class HumanRubricAudit(BaseModel):
    """Human annotator score and review on a single rubric dimension."""
    dimension: RatingDimension
    score_a: float = Field(ge=1.0, le=5.0)
    score_b: float = Field(ge=1.0, le=5.0)
    agreed_with_ai: bool = True
    comment: Optional[str] = None


class DetailedHumanAuditRequest(BaseModel):
    """Complete human annotator audit payload including multi-rubric grading and feedback."""
    sample_id: str
    prompt: str = Field(..., max_length=MAX_PROMPT_LENGTH)
    response_a: str = Field(..., max_length=MAX_RESPONSE_LENGTH)
    response_b: str = Field(..., max_length=MAX_RESPONSE_LENGTH)
    model_a_name: str = "Candidate A"
    model_b_name: str = "Candidate B"
    human_overall_winner: PreferenceChoice
    human_rubrics: List[HumanRubricAudit] = Field(default_factory=list)
    human_rationale: Optional[str] = None
    ai_verdict: str
    ai_rubric_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)

