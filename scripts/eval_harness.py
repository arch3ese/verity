"""
Evaluation Harness Scaffold (Task 3.9 & EVAL-04).
Runs pairwise judge evaluations across a labeled CSV benchmark, reporting
agreement with human annotations, positional flip rates, and Cohen's Kappa.

DISCLAIMER: Scaffold demonstration only. Results on small evaluation sets
are NOT statistically meaningful.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from verity.judges.debiaser import DebiasingJudge
from verity.judges.pairwise import PairwiseJudge
from verity.models.factory import MockChatModel, get_chat_model
from verity.schemas.models import PreferenceChoice, PromptSample
from verity.stats.agreement import calculate_cohens_kappa

load_dotenv()


def run_evaluation_harness(
    csv_path: str,
    provider: str = "mock",
    model_name: str = "mock-judge",
) -> int:
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: Dataset file not found at '{csv_path}'")
        return 1

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_samples = len(rows)
    print("\n" + "=" * 72)
    print("           VERITY PAIRWISE EVALUATION HARNESS")
    print("=" * 72)
    print(f"Dataset:  {csv_path} ({total_samples} samples)")
    print(f"Provider: {provider} (model: {model_name})")
    print("-" * 72)
    print("NOTE: Results on this small evaluation set are a scaffold demonstration")
    print("only and are NOT statistically meaningful or representative of benchmark accuracy.")
    print("=" * 72 + "\n")

    # Instantiate model
    if provider == "mock":
        chat_model = MockChatModel(
            mock_response=(
                '{"winner": "model_a", "margin": "clear", '
                '"dimension_scores": [{"dimension": "instruction_following", "score_a": 5.0, "score_b": 2.0, "justification": "Candidate A followed constraints."}], '
                '"overall_reasoning": "Candidate A adhered to instructions.", "detected_errors_model_a": [], "detected_errors_model_b": []}'
            )
        )
    else:
        chat_model = get_chat_model(provider, model_name)

    pairwise_judge = PairwiseJudge(judge_model=chat_model)
    debiaser = DebiasingJudge(judge=pairwise_judge)

    judge_verdicts: List[str] = []
    human_labels: List[str] = []
    positional_flips = 0

    print("Executing dual-order debiased evaluations...\n")
    for idx, row in enumerate(rows, 1):
        sample_id = row.get("id", f"sample-{idx}")
        prompt = row.get("prompt", "")
        response_a = row.get("response_a", "")
        response_b = row.get("response_b", "")
        human_pref = row.get("human_preferred", "").strip().lower()
        constraints = [c.strip() for c in row.get("constraints", "").split(";") if c.strip()]

        sample = PromptSample(id=sample_id, prompt=prompt, constraints=constraints)

        try:
            eval_res = debiaser.evaluate_with_debiasing(sample, response_a, response_b)
            final_winner = eval_res.final_consensus_winner.value
            has_pos_bias = eval_res.positional_bias_detected
        except Exception as e:
            print(f"  [{idx}/{total_samples}] {sample_id}: Error evaluating - {e}")
            final_winner = "indeterminate"
            has_pos_bias = False

        if has_pos_bias:
            positional_flips += 1

        judge_verdicts.append(final_winner)
        human_labels.append(human_pref)

        status_marker = "[FLIP]" if has_pos_bias else "[OK]"
        match_marker = "MATCH" if final_winner == human_pref else "DIFF"
        print(f"  Sample {idx:02d} ({sample_id}): AI={final_winner.upper():<13} Human={human_pref.upper():<10} {match_marker} {status_marker}")

    # Metrics computation
    matches = sum(1 for j, h in zip(judge_verdicts, human_labels) if j == h)
    accuracy = (matches / total_samples) * 100.0 if total_samples > 0 else 0.0
    flip_rate = (positional_flips / total_samples) * 100.0 if total_samples > 0 else 0.0

    kappa, interp = calculate_cohens_kappa(judge_verdicts, human_labels)

    print("\n" + "=" * 72)
    print("                    EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Total Evaluated:        {total_samples}")
    print(f"Agreement with Human:   {matches}/{total_samples} ({accuracy:.1f}%)")
    print(f"Positional Flip Rate:   {positional_flips}/{total_samples} ({flip_rate:.1f}%)")
    kappa_str = f"{kappa:.3f}" if (kappa is not None and not (isinstance(kappa, float) and sys.float_info.max < kappa or kappa != kappa)) else "NaN (Undefined)"
    print(f"Cohen's Kappa (vs Human): {kappa_str} ({interp})")
    print("=" * 72)
    print("CAUTION: Do not extrapolate or publish these metrics as official benchmark")
    print("numbers. A rigorous benchmark requires hundreds of high-quality items.")
    print("=" * 72 + "\n")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Verity Evaluation Harness Scaffold")
    parser.add_argument("--csv", default="data/eval_placeholder.csv", help="Path to evaluation CSV dataset")
    parser.add_argument("--provider", default="mock", help="Judge provider: 'mock', 'gemini', 'openai', 'ollama'")
    parser.add_argument("--model", default="mock-judge", help="Judge model name")

    args = parser.parse_args()
    return run_evaluation_harness(args.csv, provider=args.provider, model_name=args.model)


if __name__ == "__main__":
    sys.exit(main())
