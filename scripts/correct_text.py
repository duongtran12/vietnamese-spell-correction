"""Correct one Vietnamese sentence using local project artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

from src.inference import SpellingCorrectionPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Vietnamese sentence to check")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--trigram", type=Path, required=True)
    parser.add_argument(
        "--inference-config", type=Path, default=Path("configs/inference.yaml")
    )
    parser.add_argument(
        "--ranking-config", type=Path, default=Path("configs/ranking.yaml")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML object: {path}")
    return config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
    result = pipeline.correct_text(args.text)

    print(f"Original:  {result.original_text}")
    print(f"Corrected: {result.corrected_text}")
    for decision in result.decisions:
        suggestions = ", ".join(
            suggestion.token for suggestion in decision.suggestions
        ) or "(none)"
        print(
            f"[{decision.token_index}] {decision.original_token} "
            f"confidence={decision.detector_confidence:.3f} "
            f"suggestions={suggestions} applied={decision.applied_token}"
        )


if __name__ == "__main__":
    main()

