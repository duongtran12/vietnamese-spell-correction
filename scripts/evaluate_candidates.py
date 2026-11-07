"""Evaluate the edit-distance candidate baseline on a processed split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.correction.candidates import CandidateGenerator, build_training_dictionary
from src.correction.evaluation import CandidateExample, evaluate_candidate_generator
from src.data.io import read_jsonl, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Processed train JSONL")
    parser.add_argument(
        "--evaluation", type=Path, required=True, help="Processed validation or test JSONL"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ranking.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/candidate_baseline.json"),
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("ranking config must contain a YAML object")
    return config


def _examples(records: list[dict[str, Any]]) -> list[CandidateExample]:
    examples: list[CandidateExample] = []
    for record in records:
        for event in record.get("noise_events", []):
            examples.append(
                CandidateExample(
                    noisy_token=str(event["noisy_token"]),
                    clean_token=str(event["clean_token"]),
                    error_type=str(event["operation"]),
                )
            )
    return examples


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config).get("candidates", {})
    train_records = read_jsonl(args.train)
    evaluation_records = read_jsonl(args.evaluation)
    frequencies = build_training_dictionary(
        record["clean_tokens"] for record in train_records
    )
    generator = CandidateGenerator(
        frequencies,
        distance=str(config.get("distance", "damerau_levenshtein")),
        short_token_max_length=int(config.get("short_token_max_length", 4)),
        short_token_max_distance=int(config.get("short_token_max_distance", 1)),
        long_token_max_distance=int(config.get("long_token_max_distance", 2)),
    )
    top_k = int(config.get("top_k", 5))
    metrics = evaluate_candidate_generator(
        generator,
        _examples(evaluation_records),
        k_values=tuple(k for k in (1, 3, 5) if k <= top_k),
    )
    metrics["dictionary_size"] = len(frequencies)
    metrics["train_path"] = str(args.train)
    metrics["evaluation_path"] = str(args.evaluation)
    write_json(args.output, metrics)

    print(f"Evaluated {metrics['examples']} spelling errors")
    print(f"Dictionary size: {metrics['dictionary_size']}")
    print(f"Recall@k: {metrics['recall_at_k']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

