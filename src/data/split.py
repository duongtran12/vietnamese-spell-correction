"""Leakage-safe dataset splitting by document identifier."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Iterable, Mapping


SPLIT_NAMES = ("train", "validation", "test")


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remainder = total - sum(counts)
    priority = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in priority[:remainder]:
        counts[index] += 1
    return counts


def split_by_document(
    records: Iterable[Mapping[str, Any]],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Mapping[str, Any]]]:
    """Split records while keeping every document in exactly one split."""

    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must be non-negative and sum to 1")

    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if "document_id" not in record:
            raise ValueError("every record must contain document_id")
        grouped[record["document_id"]].append(record)

    document_ids = sorted(grouped, key=str)
    random.Random(seed).shuffle(document_ids)
    counts = _allocate_counts(len(document_ids), ratios)

    result = {name: [] for name in SPLIT_NAMES}
    offset = 0
    for name, count in zip(SPLIT_NAMES, counts, strict=True):
        selected_ids = document_ids[offset : offset + count]
        for document_id in selected_ids:
            result[name].extend(grouped[document_id])
        offset += count

    assert_no_document_leakage(result)
    return result


def assert_no_document_leakage(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    """Raise an error when a document identifier occurs in multiple splits."""

    seen: dict[Any, str] = {}
    for split_name, records in splits.items():
        for record in records:
            document_id = record["document_id"]
            previous_split = seen.setdefault(document_id, split_name)
            if previous_split != split_name:
                raise ValueError(
                    f"document_id {document_id!r} appears in both "
                    f"{previous_split!r} and {split_name!r}"
                )

