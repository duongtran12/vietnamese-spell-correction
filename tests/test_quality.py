import pytest

from src.data.quality import (
    assert_no_sentence_leakage,
    assert_unique_sentence_ids,
    deduplicate_sentences,
)
from src.data.schema import CleanSentence


def test_deduplicate_sentences_normalizes_unicode_and_whitespace() -> None:
    records = [
        CleanSentence("doc-1", "sent-1", "Tiếng Việt rất đẹp."),
        CleanSentence("doc-2", "sent-2", "Tiếng   Việt rất đẹp."),
    ]

    unique, removed = deduplicate_sentences(records)

    assert len(unique) == 1
    assert removed == 1


def test_sentence_leakage_check_detects_cross_split_duplicate() -> None:
    splits = {
        "train": [CleanSentence("doc-1", "sent-1", "Cùng một câu.")],
        "validation": [CleanSentence("doc-2", "sent-2", "Cùng  một câu.")],
        "test": [],
    }

    with pytest.raises(ValueError, match="appears in both"):
        assert_no_sentence_leakage(splits)


def test_duplicate_sentence_id_is_rejected() -> None:
    records = [
        CleanSentence("doc-1", "sent-1", "Câu thứ nhất."),
        CleanSentence("doc-2", "sent-1", "Câu thứ hai."),
    ]

    with pytest.raises(ValueError, match="duplicate sentence_id"):
        assert_unique_sentence_ids(records)

