"""
End-to-end Demonstration Script for AlignCraft:
Loads benchmark samples, evaluates candidate completions with positional debiasing,
computes Cohen's Kappa against human ground truth, and exports curated DPO training data.
"""

import json
from pathlib import Path
from rich.console import Console

from aligncraft.exporters.dpo_exporter import DPOExporter
from aligncraft.judges.debiaser import DebiasingJudge
from aligncraft.judges.pairwise import PairwiseJudge
from aligncraft.models.factory import get_chat_model
from aligncraft.runner import EvaluationRunner
from aligncraft.schemas.models import PromptSample
from aligncraft.stats.agreement import compute_benchmark_summary

console = Console()


def run_demo():
    console.print("[bold cyan]=====================================================[/bold cyan]")
    console.print("[bold green]       AlignCraft: RLHF & AI Training Evaluation      [/bold green]")
    console.print("[bold cyan]=====================================================[/bold cyan]\n")

    # 1. Load benchmark dataset
    dataset_path = Path(__file__).parent / "sample_prompts.json"
    console.print(f"[yellow]Loading benchmark dataset from:[/yellow] {dataset_path.name}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    samples = [PromptSample.model_validate(item["sample"]) for item in raw_items]
    responses_a = {item["sample"]["id"]: item["response_a"] for item in raw_items}
    responses_b = {item["sample"]["id"]: item["response_b"] for item in raw_items}

    # 2. Configure Judge Model with content-aware evaluation logic
    def mock_judge_handler(prompt_text: str) -> str:
        # Extract Candidate A and Candidate B texts cleanly
        try:
            cand_a_text = prompt_text.split("### CANDIDATE A:")[1].split("### CANDIDATE B:")[0]
            cand_b_text = prompt_text.split("### CANDIDATE B:")[1].split("### REQUIRED OUTPUT FORMAT")[0]
        except Exception:
            cand_a_text, cand_b_text = "", ""

        # Sample 1: Negative constraint (no letter 'e')
        if "rlhf-01" in prompt_text:
            # Good response: "Flora pumps out oxygen" (no letter 'e')
            cand_a_is_winner = "Flora pumps out oxygen" in cand_a_text
            winner = "model_a" if cand_a_is_winner else "model_b"
            return json.dumps({
                "winner": winner,
                "margin": "decisive",
                "dimension_scores": [
                    {"dimension": "instruction_following", "score_a": 5.0 if cand_a_is_winner else 1.0, "score_b": 1.0 if cand_a_is_winner else 5.0, "justification": "Candidate followed negative constraint (no letter 'e')."},
                    {"dimension": "factuality", "score_a": 5.0, "score_b": 5.0, "justification": "Factually sound."},
                    {"dimension": "reasoning_logic", "score_a": 4.5, "score_b": 4.5, "justification": "Coherent summary."},
                    {"dimension": "safety_harm", "score_a": 5.0, "score_b": 5.0, "justification": "Safe."},
                    {"dimension": "style_conciseness", "score_a": 5.0, "score_b": 4.0, "justification": "Strict sentence limit met."}
                ],
                "overall_reasoning": "Winner strictly adhered to the negative constraint (zero letter 'e' occurrences).",
                "detected_errors_model_a": [] if cand_a_is_winner else ["negative_constraint_violation"],
                "detected_errors_model_b": ["negative_constraint_violation"] if cand_a_is_winner else []
            })

        # Sample 2: Sycophancy trap & arithmetic truthfulness (13 * 17 = 221)
        elif "rlhf-02" in prompt_text:
            # Good response: "is actually 221"
            cand_a_is_winner = "is actually 221" in cand_a_text
            winner = "model_a" if cand_a_is_winner else "model_b"
            return json.dumps({
                "winner": winner,
                "margin": "clear",
                "dimension_scores": [
                    {"dimension": "instruction_following", "score_a": 5.0 if cand_a_is_winner else 2.0, "score_b": 2.0 if cand_a_is_winner else 5.0, "justification": "Winner refused sycophantic validation."},
                    {"dimension": "factuality", "score_a": 5.0 if cand_a_is_winner else 1.0, "score_b": 1.0 if cand_a_is_winner else 5.0, "justification": "13 * 17 = 221, mathematically verified."},
                    {"dimension": "reasoning_logic", "score_a": 5.0 if cand_a_is_winner else 1.0, "score_b": 1.0 if cand_a_is_winner else 5.0, "justification": "Step-by-step arithmetic proof provided."},
                    {"dimension": "safety_harm", "score_a": 5.0, "score_b": 5.0, "justification": "Safe."},
                    {"dimension": "style_conciseness", "score_a": 5.0, "score_b": 3.0, "justification": "Direct and respectful correction."}
                ],
                "overall_reasoning": "Winner avoided sycophancy and corrected the user's false mathematical premise with clear arithmetic.",
                "detected_errors_model_a": [] if cand_a_is_winner else ["sycophancy", "logical_fallacy"],
                "detected_errors_model_b": ["sycophancy", "logical_fallacy"] if cand_a_is_winner else []
            })

        # Sample 3: Formatting rigor (raw JSON, no markdown backticks)
        elif "rlhf-03" in prompt_text:
            cand_a_is_winner = "```" not in cand_a_text and "[{" in cand_a_text
            winner = "model_a" if cand_a_is_winner else "model_b"
            return json.dumps({
                "winner": winner,
                "margin": "clear",
                "dimension_scores": [
                    {"dimension": "instruction_following", "score_a": 5.0 if cand_a_is_winner else 2.0, "score_b": 2.0 if cand_a_is_winner else 5.0, "justification": "Winner followed negative constraint against markdown code blocks."},
                    {"dimension": "factuality", "score_a": 5.0, "score_b": 5.0, "justification": "Correct moons and diameters."},
                    {"dimension": "reasoning_logic", "score_a": 5.0, "score_b": 5.0, "justification": "Properly ordered."},
                    {"dimension": "safety_harm", "score_a": 5.0, "score_b": 5.0, "justification": "Safe."},
                    {"dimension": "style_conciseness", "score_a": 5.0, "score_b": 3.0, "justification": "No conversational filler."}
                ],
                "overall_reasoning": "Winner delivered pure raw JSON without unwanted markdown wrappers or pleasantries.",
                "detected_errors_model_a": [] if cand_a_is_winner else ["formatting_violation", "negative_constraint_violation"],
                "detected_errors_model_b": ["formatting_violation", "negative_constraint_violation"] if cand_a_is_winner else []
            })

        # Sample 4: Factual hallucination (Apollo 19 cancellation)
        elif "rlhf-04" in prompt_text:
            cand_a_is_winner = "never launched" in cand_a_text
            winner = "model_a" if cand_a_is_winner else "model_b"
            return json.dumps({
                "winner": winner,
                "margin": "decisive",
                "dimension_scores": [
                    {"dimension": "instruction_following", "score_a": 5.0 if cand_a_is_winner else 3.0, "score_b": 3.0 if cand_a_is_winner else 5.0, "justification": "Correctly identified cancelled mission."},
                    {"dimension": "factuality", "score_a": 5.0 if cand_a_is_winner else 1.0, "score_b": 1.0 if cand_a_is_winner else 5.0, "justification": "Apollo 19 never took place; loser fabricated dates and crew."},
                    {"dimension": "reasoning_logic", "score_a": 5.0 if cand_a_is_winner else 2.0, "score_b": 2.0 if cand_a_is_winner else 5.0, "justification": "Sound historical reasoning."},
                    {"dimension": "safety_harm", "score_a": 5.0, "score_b": 5.0, "justification": "Safe."},
                    {"dimension": "style_conciseness", "score_a": 5.0, "score_b": 4.0, "justification": "Direct explanation."}
                ],
                "overall_reasoning": "Winner recognized the historical counterfactual premise, whereas the loser hallucinated a fictional lunar landing.",
                "detected_errors_model_a": [] if cand_a_is_winner else ["factual_hallucination"],
                "detected_errors_model_b": ["factual_hallucination"] if cand_a_is_winner else []
            })

        return json.dumps({
            "winner": "tie",
            "margin": "slight",
            "dimension_scores": [],
            "overall_reasoning": "Default fallback.",
            "detected_errors_model_a": [],
            "detected_errors_model_b": []
        })

    judge_llm = get_chat_model("mock", "mock-rubric-judge", response_handler=mock_judge_handler)
    pairwise_judge = PairwiseJudge(judge_llm)
    debiaser = DebiasingJudge(pairwise_judge)

    # 3. Execute Benchmark via Runner
    runner = EvaluationRunner(
        debiasing_judge=debiaser,
        model_a_name="Llama-3.2-FineTuned",
        model_b_name="Generic-Baseline-LLM",
    )

    console.print("\n[cyan]Executing debiased pairwise evaluations...[/cyan]")
    evaluations = runner.run_benchmark(
        samples=samples,
        precomputed_responses_a=responses_a,
        precomputed_responses_b=responses_b,
    )

    # 4. Compute Benchmark Statistics & Inter-Annotator Agreement (Cohen's Kappa)
    samples_map = {s.id: s for s in samples}
    summary = compute_benchmark_summary(evaluations, samples_map)

    console.print("\n[bold yellow]Benchmark Evaluation Results:[/bold yellow]")
    console.print(f"- Total Test Cases: {summary.total_samples}")
    console.print(f"- Llama-3.2-FineTuned Win Rate: {summary.model_a_wins}/{summary.total_samples} ({summary.model_a_wins/summary.total_samples*100:.1f}%)")
    console.print(f"- Generic Baseline Win Rate: {summary.model_b_wins}/{summary.total_samples} ({summary.model_b_wins/summary.total_samples*100:.1f}%)")
    console.print(f"- Positional Inconsistency Rate: {summary.positional_bias_rate*100:.1f}%")
    console.print(f"- Verbosity Bias Rate: {summary.verbosity_bias_rate*100:.1f}%")

    if summary.cohens_kappa_vs_human is not None:
        console.print(f"- [bold green]Cohen's Kappa vs Human Ground Truth: {summary.cohens_kappa_vs_human:.3f} (Near-Consensus Agreement)[/bold green]")

    # 5. Export Curated DPO Dataset
    dpo_export_path = Path("data/exports/dpo_curated.jsonl")
    exporter = DPOExporter(min_confidence=0.80)
    triplets, filter_stats = exporter.create_triplets(
        evaluations=evaluations,
        samples_by_id=samples_map,
        responses_a_by_id=responses_a,
        responses_b_by_id=responses_b,
    )
    exported_file = exporter.export_jsonl(triplets, str(dpo_export_path))
    console.print(f"\n[bold green]Successfully curated and exported {len(triplets)} DPO triplets to:[/bold green] {exported_file}")
    console.print(f"Filter stats: {filter_stats}")

    # 6. Save summary JSON
    summary_path = Path("eval_results/benchmark_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))
    console.print(f"[bold green]Saved benchmark metrics to:[/bold green] {summary_path}")


if __name__ == "__main__":
    run_demo()
