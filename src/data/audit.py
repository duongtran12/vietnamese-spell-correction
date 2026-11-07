"""Data-quality statistics for a clean Vietnamese sentence corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import math
import re
import unicodedata
from typing import Any, Sequence

from src.data.normalize import normalize_text, tokenize
from src.data.quality import canonical_sentence
from src.data.schema import CleanSentence


URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _abnormal_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    abnormal = sum(
        not (
            character.isspace()
            or unicodedata.category(character)[0] in {"L", "M", "N", "P"}
        )
        for character in text
    )
    return abnormal / len(text)


def _near_duplicate_report(
    records: Sequence[CleanSentence],
    *,
    threshold: float,
    maximum_comparisons: int,
    maximum_examples: int,
) -> dict[str, Any]:
    blocks: dict[tuple[int, tuple[str, ...]], list[tuple[CleanSentence, str]]] = defaultdict(list)
    for record in records:
        canonical = canonical_sentence(record.text).casefold()
        tokens = canonical.split()
        block_key = (len(tokens) // 5, tuple(tokens[:2]))
        blocks[block_key].append((record, canonical))

    comparisons = 0
    pairs_found = 0
    examples: list[dict[str, Any]] = []
    truncated = False
    for block in blocks.values():
        for left_index in range(len(block)):
            left_record, left_text = block[left_index]
            for right_index in range(left_index + 1, len(block)):
                if comparisons >= maximum_comparisons:
                    truncated = True
                    break
                right_record, right_text = block[right_index]
                comparisons += 1
                if left_text == right_text:
                    continue
                similarity = SequenceMatcher(None, left_text, right_text).ratio()
                if similarity >= threshold:
                    pairs_found += 1
                    if len(examples) < maximum_examples:
                        examples.append(
                            {
                                "left_sentence_id": left_record.sentence_id,
                                "right_sentence_id": right_record.sentence_id,
                                "similarity": similarity,
                            }
                        )
            if truncated:
                break
        if truncated:
            break
    return {
        "threshold": threshold,
        "comparisons": comparisons,
        "truncated": truncated,
        "pairs_found": pairs_found,
        "examples": examples,
    }


def audit_clean_corpus(
    records: Sequence[CleanSentence],
    *,
    near_duplicate_threshold: float = 0.9,
    maximum_near_duplicate_comparisons: int = 100_000,
    maximum_near_duplicate_examples: int = 100,
) -> dict[str, Any]:
    """Return deterministic quality statistics without modifying the corpus."""

    if not 0.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    if maximum_near_duplicate_comparisons < 0:
        raise ValueError("maximum comparisons must be non-negative")
    if maximum_near_duplicate_examples < 0:
        raise ValueError("maximum examples must be non-negative")

    sentence_ids = Counter(record.sentence_id for record in records)
    canonical_counts = Counter(canonical_sentence(record.text) for record in records)
    character_lengths = [len(record.text) for record in records]
    token_lengths = [len(tokenize(record.text)) for record in records]
    source_counts = Counter(record.source or "unspecified" for record in records)
    abnormal_ratios = [_abnormal_character_ratio(record.text) for record in records]
    duplicate_sentence_groups = sum(count > 1 for count in canonical_counts.values())
    duplicate_sentence_records = sum(count - 1 for count in canonical_counts.values() if count > 1)

    return {
        "documents": len({record.document_id for record in records}),
        "sentences": len(records),
        "sources": dict(sorted(source_counts.items())),
        "characters": sum(character_lengths),
        "tokens": sum(token_lengths),
        "character_length": _distribution(character_lengths),
        "token_length": _distribution(token_lengths),
        "empty_after_normalization": sum(
            not normalize_text(record.text).strip() for record in records
        ),
        "non_nfc_sentences": sum(
            not unicodedata.is_normalized("NFC", record.text) for record in records
        ),
        "duplicate_sentence_id_groups": sum(count > 1 for count in sentence_ids.values()),
        "exact_duplicate_sentence_groups": duplicate_sentence_groups,
        "exact_duplicate_records": duplicate_sentence_records,
        "sentences_with_url": sum(bool(URL_PATTERN.search(record.text)) for record in records),
        "sentences_with_email": sum(bool(EMAIL_PATTERN.search(record.text)) for record in records),
        "sentences_with_abnormal_character_ratio_above_10_percent": sum(
            ratio > 0.1 for ratio in abnormal_ratios
        ),
        "near_duplicates": _near_duplicate_report(
            records,
            threshold=near_duplicate_threshold,
            maximum_comparisons=maximum_near_duplicate_comparisons,
            maximum_examples=maximum_near_duplicate_examples,
        ),
    }

