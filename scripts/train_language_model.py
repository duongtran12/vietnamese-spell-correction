"""Fit and save a trigram language model from processed training data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.data.io import read_jsonl, write_json
from src.models.trigram_lm import TrigramLanguageModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ranking.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/trigram_lm.json")
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("ranking config must contain a YAML object")
    return config


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config).get("language_model", {})
    records = read_jsonl(args.train)
    model = TrigramLanguageModel(
        add_k=float(config.get("add_k", 0.1)),
        lambdas=tuple(config.get("lambdas", (0.1, 0.3, 0.6))),  # type: ignore[arg-type]
    ).fit(record["clean_tokens"] for record in records)
    artifact = model.to_dict()
    artifact["metadata"] = {
        "artifact_type": "statistical_trigram_language_model",
        "fit_split": "train",
        "train_path": str(args.train),
        "training_sentences": len(records),
    }
    write_json(args.output, artifact)

    print(f"Trained on {len(records)} sentences")
    print(f"Vocabulary size: {len(model.vocabulary)}")
    print(f"Artifact: {args.output}")


if __name__ == "__main__":
    main()

