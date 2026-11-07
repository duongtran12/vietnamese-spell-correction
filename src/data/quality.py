"""Duplicate removal and cross-split sentence leakage checks."""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable, Mapping

from src.data.normalize import normalize_text
from src.data.schema import CleanSentence


WHITESPACE_PATTERN = re.compile(r"\s+")


def canonical_sentence(text: str) -> str:
    """Return the representation used for exact normalized duplicate checks."""

    return WHITESPACE_PATTERN.sub(" ", normalize_text(text)).strip()


def near_duplicate_block_key(text: str) -> tuple[int, tuple[str, ...]]:
    """Place likely near duplicates in a deterministic comparison block."""

    tokens = canonical_sentence(text).casefold().split()
    return len(tokens) // 5, tuple(tokens[:2])


def near_duplicate_similarity(left: str, right: str) -> float:
    """Return character-sequence similarity after canonicalization."""

    left = canonical_sentence(left).casefold()
    right = canonical_sentence(right).casefold()
    return SequenceMatcher(None, left, right).ratio()


def deduplicate_sentences(
    records: Iterable[CleanSentence],
) -> tuple[list[CleanSentence], int]:
    """Keep the first occurrence of every exact normalized sentence."""

    unique: list[CleanSentence] = []
    seen: set[str] = set()
    removed = 0
    for record in records:
        canonical = canonical_sentence(record.text)
        if canonical in seen:
            removed += 1
            continue
        seen.add(canonical)
        unique.append(record)
    return unique, removed


def deduplicate_near_sentences(
    records: Iterable[CleanSentence],
    *,
    threshold: float = 0.95,
    maximum_comparisons: int = 1_000_000,
) -> tuple[list[CleanSentence], dict[str, int | float]]:
    """Remove later near-duplicate sentences within conservative blocks.

    The comparison limit raises an error instead of silently returning a
    partially deduplicated corpus.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be between 0 and 1")
    if maximum_comparisons < 1:
        raise ValueError("maximum comparisons must be positive")

    blocks: dict[tuple[int, tuple[str, ...]], list[CleanSentence]] = defaultdict(list)
    unique: list[CleanSentence] = []
    removed = 0
    comparisons = 0
    for record in records:
        block = blocks[near_duplicate_block_key(record.text)]
        is_duplicate = False
        for existing in block:
            if comparisons >= maximum_comparisons:
                raise RuntimeError(
                    "near-duplicate comparison limit reached; increase the limit "
                    "or strengthen blocking before preparing final splits"
                )
            comparisons += 1
            if near_duplicate_similarity(record.text, existing.text) >= threshold:
                is_duplicate = True
                removed += 1
                break
        if not is_duplicate:
            block.append(record)
            unique.append(record)

    return unique, {
        "threshold": threshold,
        "comparisons": comparisons,
        "records_removed": removed,
    }


def assert_unique_sentence_ids(records: Iterable[CleanSentence]) -> None:
    """Raise when a sentence identifier is reused within the corpus."""

    seen: set[str] = set()
    for record in records:
        if record.sentence_id in seen:
            raise ValueError(f"duplicate sentence_id: {record.sentence_id!r}")
        seen.add(record.sentence_id)


def assert_no_sentence_leakage(
    splits: Mapping[str, Iterable[CleanSentence]],
) -> None:
    """Raise when an exact normalized sentence appears in multiple splits."""

    seen: dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            canonical = canonical_sentence(record.text)
            previous_split = seen.setdefault(canonical, split_name)
            if previous_split != split_name:
                raise ValueError(
                    f"sentence {record.sentence_id!r} appears in both "
                    f"{previous_split!r} and {split_name!r}"
                )

