"""Tune detector threshold and rank margin on a validation split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from src.data.io import read_jsonl, write_json
from src.evaluation.end_to_end import evaluate_end_to_end
from src.inference import SpellingCorrectionPipeline


def _float_grid(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("grid values must be non-negative")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--trigram", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=_float_grid,
        default=_float_grid("0.5,0.6,0.7,0.8,0.9"),
    )
    parser.add_argument(
        "--rank-margins",
        type=_float_grid,
        default=_float_grid("0.25,0.5,0.75"),
    )
    parser.add_argument("--max-false-correction-rate", type=float, default=0.02)
    parser.add_argument(
        "--inference-config", type=Path, default=Path("configs/inference.yaml")
    )
    parser.add_argument(
        "--ranking-config", type=Path, default=Path("configs/ranking.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/inference-tuning.json")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML object: {path}")
    return config


def select_best_result(
    results: Iterable[dict[str, Any]],
    *,
    max_false_correction_rate: float,
) -> dict[str, Any]:
    """Select correction F1 first, then safer and more accurate sentences."""

    candidates = list(results)
    if not candidates:
        raise ValueError("results must not be empty")
    eligible = [
        result
        for result in candidates
        if result["metrics"]["false_correction_rate"]
        <= max_false_correction_rate
    ]
    pool = eligible or candidates
    return max(
        pool,
        key=lambda result: (
            result["metrics"]["correction"]["f1"],
            -result["metrics"]["false_correction_rate"],
            result["metrics"]["sentence_accuracy"],
            result["detector_threshold"],
            result["minimum_rank_margin"],
        ),
    )


def main() -> None:
    args = _parse_args()
    if args.max_false_correction_rate < 0:
        raise ValueError("max false-correction rate must be non-negative")
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
        detector_threshold=args.thresholds[0],
        minimum_rank_margin=args.rank_margins[0],
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
    records = read_jsonl(args.data)
    results: list[dict[str, Any]] = []
    for detector_threshold in args.thresholds:
        if detector_threshold > 1:
            raise ValueError("detector thresholds must be at most 1")
        for minimum_rank_margin in args.rank_margins:
            pipeline.detector_threshold = detector_threshold
            pipeline.minimum_rank_margin = minimum_rank_margin
            metrics = evaluate_end_to_end(pipeline, records)
            result = {
                "detector_threshold": detector_threshold,
                "minimum_rank_margin": minimum_rank_margin,
                "metrics": metrics,
            }
            results.append(result)
            print(
                f"threshold={detector_threshold:.2f} "
                f"margin={minimum_rank_margin:.2f} "
                f"correction_f1={metrics['correction']['f1']:.4f} "
                f"false_correction_rate={metrics['false_correction_rate']:.4f}"
            )

    best = select_best_result(
        results,
        max_false_correction_rate=args.max_false_correction_rate,
    )
    write_json(
        args.output,
        {
            "selection": {
                "primary_metric": "correction.f1",
                "max_false_correction_rate": args.max_false_correction_rate,
                "tie_breakers": [
                    "lower false_correction_rate",
                    "higher sentence_accuracy",
                    "higher detector_threshold",
                    "higher minimum_rank_margin",
                ],
            },
            "best": best,
            "results": results,
        },
    )
    print(
        "Best: "
        f"threshold={best['detector_threshold']:.2f} "
        f"margin={best['minimum_rank_margin']:.2f} "
        f"correction_f1={best['metrics']['correction']['f1']:.4f}"
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
