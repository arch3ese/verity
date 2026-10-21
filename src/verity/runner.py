"""
Async & Batch Evaluation Runner:
Coordinates multi-sample benchmark runs, model completions, debiased evaluations,
and statistical aggregation.
"""

import time
from typing import Callable, Dict, List, Optional
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from verity.judges.debiaser import DebiasingJudge
from verity.judges.pairwise import PairwiseJudge
from verity.schemas.models import (
    BenchmarkSummary,
    DebiasedEvaluation,
    ModelOutput,
    PromptSample,
)
from verity.stats.agreement import compute_benchmark_summary

console = Console()


class EvaluationRunner:
    """
    Orchestrates batch evaluations over a dataset of PromptSample instances.
    """

    def __init__(
        self,
        debiasing_judge: DebiasingJudge,
        model_a_generator: Optional[Callable[[str], str]] = None,
        model_b_generator: Optional[Callable[[str], str]] = None,
        model_a_name: str = "model_a",
        model_b_name: str = "model_b",
    ):
        self.debiasing_judge = debiasing_judge
        self.model_a_generator = model_a_generator
        self.model_b_generator = model_b_generator
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name

    def run_benchmark(
        self,
        samples: List[PromptSample],
        precomputed_responses_a: Optional[Dict[str, str]] = None,
        precomputed_responses_b: Optional[Dict[str, str]] = None,
    ) -> List[DebiasedEvaluation]:
        """
        Executes debiased pairwise evaluations across all samples.
        Supports either precomputed model responses or live generation.
        """
        evaluations: List[DebiasedEvaluation] = []
        samples_by_id = {s.id: s for s in samples}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Evaluating {len(samples)} samples...", total=len(samples))

            for sample in samples:
                # Resolve Response A
                if precomputed_responses_a and sample.id in precomputed_responses_a:
                    resp_a = precomputed_responses_a[sample.id]
                elif self.model_a_generator:
                    resp_a = self.model_a_generator(sample.prompt)
                else:
                    raise ValueError(f"No response generator or precomputed output for Model A on sample {sample.id}")

                # Resolve Response B
                if precomputed_responses_b and sample.id in precomputed_responses_b:
                    resp_b = precomputed_responses_b[sample.id]
                elif self.model_b_generator:
                    resp_b = self.model_b_generator(sample.prompt)
                else:
                    raise ValueError(f"No response generator or precomputed output for Model B on sample {sample.id}")

                # Evaluate with debiasing
                evaluation = self.debiasing_judge.evaluate_with_debiasing(
                    sample=sample,
                    response_a=resp_a,
                    response_b=resp_b,
                    model_a_name=self.model_a_name,
                    model_b_name=self.model_b_name,
                )
                evaluations.append(evaluation)
                progress.update(task, advance=1)

        return evaluations
