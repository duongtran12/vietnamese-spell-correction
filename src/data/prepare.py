"""End-to-end preparation of clean records into reproducible data splits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.data.noise import generate_noisy_tokens
from src.data.normalize import normalize_text, tokenize
from src.data.quality import (
    assert_no_sentence_leakage,
    assert_unique_sentence_ids,
    deduplicate_sentences,
)
from src.data.schema import CleanSentence, PreparedSentence
from src.data.split import assert_no_document_leakage, split_by_document
from src.data.vocabulary import Vocabulary, build_vocabulary, oov_statistics


@dataclass(frozen=True)
class PreparedDataset:
    splits: dict[str, list[PreparedSentence]]
    vocabulary: Vocabulary
    report: dict[str, Any]


def _normalize_records(records: Sequence[CleanSentence]) -> list[CleanSentence]:
    return [
        CleanSentence(
            document_id=record.document_id,
            sentence_id=record.sentence_id,
            text=normalize_text(record.text).strip(),
            source=record.source,
        )
        for record in records
    ]


def _prepare_split(
    records: Sequence[CleanSentence],
    *,
    seed: int,
    sentence_error_probability: float,
    max_errors: int,
    operations: Sequence[str],
) -> list[PreparedSentence]:
    prepared: list[PreparedSentence] = []
    for index, record in enumerate(records):
        clean_tokens = tokenize(record.text)
        noisy = generate_noisy_tokens(
            clean_tokens,
            seed=seed + index,
            sentence_error_probability=sentence_error_probability,
            max_errors=max_errors,
            operations=operations,
        )
        prepared.append(
            PreparedSentence.create(
                record,
                clean_tokens,
                noisy.tokens,
                noisy.events,
            )
        )
    return prepared


def _split_report(
    records: Sequence[PreparedSentence], vocabulary: Vocabulary
) -> dict[str, Any]:
    tokens = [token for record in records for token in record.clean_tokens]
    operation_counts = Counter(
        event.operation for record in records for event in record.noise_events
    )
    errors = sum(sum(record.labels) for record in records)
    return {
        "documents": len({record.document_id for record in records}),
        "sentences": len(records),
        "tokens": len(tokens),
        "errors": errors,
        "sentences_with_errors": sum(bool(record.noise_events) for record in records),
        "error_operations": dict(sorted(operation_counts.items())),
        **oov_statistics(tokens, vocabulary),
    }


def prepare_dataset(
    records: Sequence[CleanSentence],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    sentence_error_probability: float = 0.6,
    max_errors: int = 1,
    operations: Sequence[str] = (
        "delete",
        "insert",
        "substitute",
        "transpose",
        "diacritic",
    ),
) -> PreparedDataset:
    """Normalize, deduplicate, split, corrupt, and summarize a clean corpus."""

    normalized = _normalize_records(records)
    assert_unique_sentence_ids(normalized)
    unique_records, duplicates_removed = deduplicate_sentences(normalized)
    split_mappings = split_by_document(
        [record.to_dict() for record in unique_records],
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    clean_splits = {
        name: [CleanSentence.from_mapping(record) for record in split_records]
        for name, split_records in split_mappings.items()
    }
    assert_no_document_leakage(split_mappings)
    assert_no_sentence_leakage(clean_splits)

    split_seed_offsets = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
    prepared_splits = {
        name: _prepare_split(
            split_records,
            seed=seed + split_seed_offsets[name],
            sentence_error_probability=sentence_error_probability,
            max_errors=max_errors,
            operations=operations,
        )
        for name, split_records in clean_splits.items()
    }
    vocabulary = build_vocabulary(
        record.clean_tokens for record in prepared_splits["train"]
    )
    report = {
        "seed": seed,
        "input_sentences": len(records),
        "unique_sentences": len(unique_records),
        "exact_duplicates_removed": duplicates_removed,
        "vocabulary_size": len(vocabulary.token_to_id),
        "splits": {
            name: _split_report(split_records, vocabulary)
            for name, split_records in prepared_splits.items()
        },
    }
    return PreparedDataset(prepared_splits, vocabulary, report)

