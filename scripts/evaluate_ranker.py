"""Evaluate context-aware ranking against the edit-distance baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.correction.candidates import CandidateGenerator, build_training_dictionary
from src.correction.ranker import CandidateRanker
from src.correction.ranking_evaluation import RankingExample, evaluate_ranker
from src.data.io import read_jsonl, write_json
from src.models.trigram_lm import TrigramLanguageModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ranking.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/ranker_validation.json")
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("ranking config must contain a YAML object")
    return config


def _examples(records: list[dict[str, Any]]) -> list[RankingExample]:
    examples: list[RankingExample] = []
    for record in records:
        tokens = tuple(str(token) for token in record["noisy_tokens"])
        for event in record.get("noise_events", []):
            examples.append(
                RankingExample(
                    tokens=tokens,
                    token_index=int(event["token_index"]),
                    clean_token=str(event["clean_token"]),
                    error_type=str(event["operation"]),
                )
            )
    return examples


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    candidate_config = config.get("candidates", {})
    language_model_config = config.get("language_model", {})
    ranker_config = config.get("ranker", {})

    train_records = read_jsonl(args.train)
    evaluation_records = read_jsonl(args.evaluation)
    train_token_sequences = [record["clean_tokens"] for record in train_records]
    frequencies = build_training_dictionary(train_token_sequences)
    generator = CandidateGenerator(
        frequencies,
        distance=str(candidate_config.get("distance", "damerau_levenshtein")),
        short_token_max_length=int(candidate_config.get("short_token_max_length", 4)),
        short_token_max_distance=int(candidate_config.get("short_token_max_distance", 1)),
        long_token_max_distance=int(candidate_config.get("long_token_max_distance", 2)),
    )
    language_model = TrigramLanguageModel(
        add_k=float(language_model_config.get("add_k", 0.1)),
        lambdas=tuple(language_model_config.get("lambdas", (0.1, 0.3, 0.6))),  # type: ignore[arg-type]
    ).fit(train_token_sequences)
    ranker = CandidateRanker(
        language_model,
        context_weight=float(ranker_config.get("context_weight", 1.0)),
        edit_weight=float(ranker_config.get("edit_weight", 1.0)),
        frequency_weight=float(ranker_config.get("frequency_weight", 0.1)),
    )
    metrics = evaluate_ranker(
        generator,
        ranker,
        _examples(evaluation_records),
        candidate_top_k=int(candidate_config.get("top_k", 5)),
    )
    metrics["weights"] = {
        "context": ranker.context_weight,
        "edit": ranker.edit_weight,
        "frequency": ranker.frequency_weight,
    }
    write_json(args.output, metrics)

    print(f"Evaluated {metrics['examples']} spelling errors")
    print(f"Baseline Accuracy@k: {metrics['baseline_accuracy_at_k']}")
    print(f"Context Accuracy@k: {metrics['context_accuracy_at_k']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

