"""Analyze validation errors and produce report-ready figures and examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from src.data.io import read_jsonl, write_json
from src.evaluation.error_analysis import analyze_errors
from src.inference import SpellingCorrectionPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--trigram", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--inference-config", type=Path, default=Path("configs/inference.yaml")
    )
    parser.add_argument(
        "--ranking-config", type=Path, default=Path("configs/ranking.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/error_analysis.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("reports/ERROR_ANALYSIS.md")
    )
    parser.add_argument(
        "--confusion-figure",
        type=Path,
        default=Path("reports/figures/detector_confusion_matrix.png"),
    )
    parser.add_argument(
        "--outcome-figure",
        type=Path,
        default=Path("reports/figures/correction_outcomes.png"),
    )
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML object: {path}")
    return config


def plot_confusion_matrix(matrix: dict[str, int], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [
        [matrix["true_negative"], matrix["false_positive"]],
        [matrix["false_negative"], matrix["true_positive"]],
    ]
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks([0, 1], ["Predicted CORRECT", "Predicted ERROR"])
    axis.set_yticks([0, 1], ["Actual CORRECT", "Actual ERROR"])
    axis.set_title("BiLSTM detector confusion matrix — validation")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(values[row][column]),
                ha="center",
                va="center",
                color="white" if values[row][column] > 1000 else "black",
                fontsize=13,
                fontweight="bold",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.subplots_adjust(left=0.25, right=0.86, bottom=0.14, top=0.88)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_outcomes(counts: dict[str, int], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "Correct\ncorrection",
        "Wrong\ncorrection",
        "Detected,\nkept",
        "Missed\ndetection",
        "False\ncorrection",
        "False positive,\nkept",
    ]
    keys = [
        "correct_correction",
        "wrong_correction",
        "detected_kept",
        "missed_detection",
        "false_correction",
        "false_positive_kept",
    ]
    values = [counts[key] for key in keys]
    colors = ["#16a34a", "#dc2626", "#eab308", "#f97316", "#be123c", "#64748b"]
    figure, axis = plt.subplots(figsize=(10, 5.8))
    bars = axis.bar(labels, values, color=colors)
    axis.set_ylabel("Token count")
    axis.set_title("End-to-end correction outcomes — validation")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, padding=3)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    matrix = report["detector_confusion_matrix"]
    counts = report["outcome_counts"]
    lines = [
        "# Validation Error Analysis",
        "",
        "This report analyzes the frozen validation configuration. The test split",
        "remains untouched.",
        "",
        "## Detector confusion matrix",
        "",
        f"- True positive: {matrix['true_positive']}.",
        f"- False positive: {matrix['false_positive']}.",
        f"- False negative: {matrix['false_negative']}.",
        f"- True negative: {matrix['true_negative']}.",
        "",
        "## Correction outcomes",
        "",
        f"- Correct corrections: {counts['correct_correction']}.",
        f"- Wrong corrections on real error positions: {counts['wrong_correction']}.",
        f"- Detected errors kept unchanged: {counts['detected_kept']}.",
        f"- Missed error positions: {counts['missed_detection']}.",
        f"- False corrections on originally correct tokens: {counts['false_correction']}.",
        f"- Detector false positives safely kept: {counts['false_positive_kept']}.",
        "",
        "## Results by synthetic error type",
        "",
    ]
    for operation, metrics in report["by_error_type"].items():
        lines.append(
            f"- **{operation}** — total {metrics['total']}; detection recall "
            f"{_percent(metrics['detection_recall'])}; Top-3 recall "
            f"{_percent(metrics['top3_recall'])}; correction recall "
            f"{_percent(metrics['correction_recall'])}."
        )
    lines.extend(["", "## Representative examples", ""])
    for category, category_examples in report["examples"].items():
        lines.extend([f"### {category.replace('_', ' ').title()}", ""])
        if not category_examples:
            lines.extend(["- No example in this validation run.", ""])
            continue
        for example in category_examples[:5]:
            confidence = example["detector_confidence"]
            confidence_text = "n/a" if confidence is None else f"{confidence:.3f}"
            lines.append(
                f"- `{example['noisy_token']}` → `{example['applied_token']}` "
                f"(target `{example['clean_token']}`, confidence {confidence_text}, "
                f"context: “{example['context']}”)."
            )
        lines.append("")
    lines.extend(
        [
            "## Figures",
            "",
            "- `reports/figures/viwiki-20261101-smoke-dedup-detector-confusion.png`.",
            "- `reports/figures/viwiki-20261101-smoke-dedup-correction-outcomes.png`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    inference_config = _load_yaml(args.inference_config)
    ranking_root = _load_yaml(args.ranking_config)
    ranking_config = ranking_root.get("ranker", {})
    candidate_config = ranking_root.get("candidates", {})
    pipeline = SpellingCorrectionPipeline.from_artifacts(
        detector_checkpoint=args.detector,
        vocabulary_artifact=args.vocabulary,
        trigram_artifact=args.trigram,
        detector_threshold=float(inference_config.get("detector_threshold", 0.5)),
        minimum_rank_margin=float(inference_config.get("minimum_rank_margin", 0.0)),
        diacritic_rank_margin=float(
            inference_config.get("diacritic_rank_margin", 0.5)
        ),
        title_case_diacritic_rank_margin=float(
            inference_config.get("title_case_diacritic_rank_margin", 0.5)
        ),
        keep_original_bonus=float(inference_config.get("keep_original_bonus", 0.75)),
        candidate_pool_size=int(inference_config.get("candidate_pool_size", 5)),
        top_k_suggestions=int(inference_config.get("top_k_suggestions", 3)),
        context_weight=float(ranking_config.get("context_weight", 1.0)),
        edit_weight=float(ranking_config.get("edit_weight", 1.0)),
        frequency_weight=float(ranking_config.get("frequency_weight", 0.1)),
        candidate_distance=str(
            candidate_config.get("distance", "damerau_levenshtein")
        ),
        short_token_max_length=int(candidate_config.get("short_token_max_length", 4)),
        short_token_max_distance=int(
            candidate_config.get("short_token_max_distance", 1)
        ),
        long_token_max_distance=int(candidate_config.get("long_token_max_distance", 2)),
        device=torch.device(args.device),
    )
    report = analyze_errors(
        pipeline,
        read_jsonl(args.data),
        max_examples_per_category=args.max_examples,
    )
    write_json(args.output, report)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    plot_confusion_matrix(report["detector_confusion_matrix"], args.confusion_figure)
    plot_outcomes(report["outcome_counts"], args.outcome_figure)
    print(f"Outcomes: {report['outcome_counts']}")
    print(f"JSON: {args.output}")
    print(f"Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
