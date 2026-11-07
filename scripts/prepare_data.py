"""Prepare JSONL corpus splits and train-only vocabulary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.data.io import read_clean_jsonl, write_json, write_jsonl
from src.data.prepare import prepare_dataset


DEFAULT_OPERATIONS = ("delete", "insert", "substitute", "transpose", "diacritic")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Clean corpus JSONL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed splits and metadata",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Data preparation YAML configuration",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("data config must contain a YAML object")
    return config


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    split_config = config.get("split", {})
    deduplication_config = config.get("deduplication", {})
    noise_config = config.get("noise", {})

    records = read_clean_jsonl(args.input)
    dataset = prepare_dataset(
        records,
        seed=int(config.get("seed", 42)),
        train_ratio=float(split_config.get("train", 0.8)),
        validation_ratio=float(split_config.get("validation", 0.1)),
        test_ratio=float(split_config.get("test", 0.1)),
        sentence_error_probability=float(
            noise_config.get("sentence_error_probability", 0.6)
        ),
        max_errors=int(noise_config.get("max_errors_per_sentence", 1)),
        operations=tuple(noise_config.get("operations", DEFAULT_OPERATIONS)),
        near_duplicate_threshold=(
            float(deduplication_config["near_duplicate_threshold"])
            if "near_duplicate_threshold" in deduplication_config
            else None
        ),
        maximum_near_duplicate_comparisons=int(
            deduplication_config.get("maximum_comparisons", 1_000_000)
        ),
    )

    for split_name, split_records in dataset.splits.items():
        write_jsonl(
            args.output_dir / f"{split_name}.jsonl",
            (record.to_dict() for record in split_records),
        )
    write_json(args.output_dir / "vocabulary.json", dataset.vocabulary.to_dict())
    write_json(args.output_dir / "statistics.json", dataset.report)

    print(f"Prepared {dataset.report['unique_sentences']} unique sentences")
    print(f"Vocabulary size: {dataset.report['vocabulary_size']}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

