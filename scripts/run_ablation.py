"""Run candidate-ranker ablations on a validation split and plot results."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from scripts.evaluate_ranker import _examples
from src.correction.candidates import CandidateGenerator, build_training_dictionary
from src.correction.ranker import CandidateRanker
from src.correction.ranking_evaluation import evaluate_ranker
from src.data.io import read_jsonl, write_json
from src.models.trigram_lm import TrigramLanguageModel


@dataclass(frozen=True)
class AblationSpec:
    name: str
    label: str
    description: str
    lambdas: tuple[float, float, float]
    context_weight: float
    edit_weight: float
    frequency_weight: float


def default_ablation_specs() -> tuple[AblationSpec, ...]:
    interpolation = (0.1, 0.3, 0.6)
    return (
        AblationSpec(
            "edit_frequency",
            "Edit + frequency",
            "Remove language-model context.",
            interpolation,
            0.0,
            1.0,
            0.1,
        ),
        AblationSpec(
            "context_edit",
            "Context + edit",
            "Remove the explicit frequency score.",
            interpolation,
            1.0,
            1.0,
            0.0,
        ),
        AblationSpec(
            "full_interpolated",
            "Full interpolated",
            "Use interpolated context, edit, and frequency scores.",
            interpolation,
            1.0,
            1.0,
            0.1,
        ),
        AblationSpec(
            "unigram_context",
            "Unigram context",
            "Replace interpolated context with unigram probability only.",
            (1.0, 0.0, 0.0),
            1.0,
            1.0,
            0.1,
        ),
        AblationSpec(
            "bigram_context",
            "Bigram context",
            "Replace interpolated context with bigram probability only.",
            (0.0, 1.0, 0.0),
            1.0,
            1.0,
            0.1,
        ),
        AblationSpec(
            "trigram_context",
            "Trigram context",
            "Replace interpolated context with trigram probability only.",
            (0.0, 0.0, 1.0),
            1.0,
            1.0,
            0.1,
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ranking.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/ranker_ablation.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("reports/figures/ranker_ablation.png")
    )
    parser.add_argument(
        "--error-figure",
        type=Path,
        default=Path("reports/figures/ranker_error_types.png"),
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("ranking config must contain a YAML object")
    return config


def plot_ablation(results: list[dict[str, Any]], output: Path) -> None:
    """Plot Accuracy@1 and Accuracy@3 for all ablation variants."""

    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [result["label"] for result in results]
    accuracy_at_1 = [result["accuracy_at_1"] * 100 for result in results]
    accuracy_at_3 = [result["accuracy_at_3"] * 100 for result in results]
    positions = list(range(len(results)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.bar(
        [position - width / 2 for position in positions],
        accuracy_at_1,
        width,
        label="Accuracy@1",
        color="#2563eb",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        accuracy_at_3,
        width,
        label="Accuracy@3",
        color="#14b8a6",
    )
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Candidate-ranker ablation on validation")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_error_types(metrics: dict[str, float], output: Path) -> None:
    """Plot full-model Accuracy@1 grouped by synthetic error operation."""

    output.parent.mkdir(parents=True, exist_ok=True)
    labels = list(metrics)
    values = [metrics[label] * 100 for label in labels]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(labels, values, color="#7c3aed")
    axis.set_ylabel("Accuracy@1 (%)")
    axis.set_title("Selected ranker Accuracy@1 by error type")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    candidate_config = config.get("candidates", {})
    language_model_config = config.get("language_model", {})
    train_records = read_jsonl(args.train)
    evaluation_records = read_jsonl(args.evaluation)
    train_sequences = [record["clean_tokens"] for record in train_records]
    examples = _examples(evaluation_records)
    generator = CandidateGenerator(
        build_training_dictionary(train_sequences),
        distance=str(candidate_config.get("distance", "damerau_levenshtein")),
        short_token_max_length=int(candidate_config.get("short_token_max_length", 4)),
        short_token_max_distance=int(
            candidate_config.get("short_token_max_distance", 1)
        ),
        long_token_max_distance=int(candidate_config.get("long_token_max_distance", 2)),
    )
    add_k = float(language_model_config.get("add_k", 0.1))
    candidate_top_k = int(candidate_config.get("top_k", 5))

    results: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None
    full_error_metrics: dict[str, float] | None = None
    for spec in default_ablation_specs():
        language_model = TrigramLanguageModel(
            add_k=add_k,
            lambdas=spec.lambdas,
        ).fit(train_sequences)
        ranker = CandidateRanker(
            language_model,
            context_weight=spec.context_weight,
            edit_weight=spec.edit_weight,
            frequency_weight=spec.frequency_weight,
        )
        metrics = evaluate_ranker(
            generator,
            ranker,
            examples,
            candidate_top_k=candidate_top_k,
        )
        if baseline_metrics is None:
            baseline_metrics = metrics
            baseline_at_1 = metrics["baseline_accuracy_at_k"]["1"]
            baseline_at_3 = metrics["baseline_accuracy_at_k"]["3"]
            results.append(
                {
                    "name": "edit_distance_baseline",
                    "label": "Edit baseline",
                    "description": "Edit distance with frequency tie-breaking.",
                    "accuracy_at_1": baseline_at_1,
                    "accuracy_at_3": baseline_at_3,
                    "delta_at_1_vs_baseline": 0.0,
                }
            )
        accuracy_at_1 = metrics["context_accuracy_at_k"]["1"]
        accuracy_at_3 = metrics["context_accuracy_at_k"]["3"]
        results.append(
            {
                **asdict(spec),
                "accuracy_at_1": accuracy_at_1,
                "accuracy_at_3": accuracy_at_3,
                "delta_at_1_vs_baseline": (
                    accuracy_at_1
                    - baseline_metrics["baseline_accuracy_at_k"]["1"]
                ),
            }
        )
        if spec.name == "context_edit":
            full_error_metrics = metrics["context_accuracy_at_1_by_error_type"]

    if baseline_metrics is None or full_error_metrics is None:
        raise RuntimeError("ablation specification is incomplete")
    report = {
        "split": "validation",
        "examples": baseline_metrics["examples"],
        "target_oov_rate": baseline_metrics["target_oov_rate"],
        "candidate_recall_at_5": baseline_metrics["candidate_recall_at_k"],
        "results": results,
        "selected_accuracy_at_1_by_error_type": full_error_metrics,
    }
    write_json(args.output, report)
    plot_ablation(results, args.figure)
    plot_error_types(full_error_metrics, args.error_figure)

    for result in results:
        print(
            f"{result['name']}: "
            f"Accuracy@1={result['accuracy_at_1']:.4f} "
            f"Accuracy@3={result['accuracy_at_3']:.4f}"
        )
    print(f"Report: {args.output}")
    print(f"Figure: {args.figure}")
    print(f"Error figure: {args.error_figure}")


if __name__ == "__main__":
    main()
