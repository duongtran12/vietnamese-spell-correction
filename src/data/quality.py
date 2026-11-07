"""Duplicate removal and cross-split sentence leakage checks."""

from __future__ import annotations

import re
from typing import Iterable, Mapping

from src.data.normalize import normalize_text
from src.data.schema import CleanSentence


WHITESPACE_PATTERN = re.compile(r"\s+")


def canonical_sentence(text: str) -> str:
    """Return the representation used for exact normalized duplicate checks."""

    return WHITESPACE_PATTERN.sub(" ", normalize_text(text)).strip()


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

