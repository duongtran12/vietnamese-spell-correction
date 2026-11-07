"""Audit a clean JSONL corpus before splitting or synthetic noise generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.audit import audit_clean_corpus
from src.data.io import read_clean_jsonl, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data_audit.json")
    )
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    parser.add_argument("--maximum-comparisons", type=int, default=100_000)
    parser.add_argument("--maximum-examples", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = read_clean_jsonl(args.input)
    report = audit_clean_corpus(
        records,
        near_duplicate_threshold=args.near_duplicate_threshold,
        maximum_near_duplicate_comparisons=args.maximum_comparisons,
        maximum_near_duplicate_examples=args.maximum_examples,
    )
    write_json(args.output, report)

    print(f"Documents: {report['documents']}")
    print(f"Sentences: {report['sentences']}")
    print(f"Tokens: {report['tokens']}")
    print(f"Exact duplicate records: {report['exact_duplicate_records']}")
    print(f"Near-duplicate examples: {report['near_duplicates']['pairs_found']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

