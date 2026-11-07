import pytest

from src.data.quality import (
    assert_no_sentence_leakage,
    assert_unique_sentence_ids,
    deduplicate_near_sentences,
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


def test_near_duplicate_deduplication_keeps_first_record() -> None:
    records = [
        CleanSentence("doc-1", "sent-1", "Đây là một câu gần như giống nhau hoàn toàn."),
        CleanSentence("doc-2", "sent-2", "Đây là một câu gần như giống nhau hoàn toàn!"),
        CleanSentence("doc-3", "sent-3", "Nội dung khác biệt được giữ lại."),
    ]

    unique, report = deduplicate_near_sentences(records, threshold=0.95)

    assert [record.sentence_id for record in unique] == ["sent-1", "sent-3"]
    assert report["records_removed"] == 1
    assert report["comparisons"] == 1


def test_near_duplicate_deduplication_fails_when_limit_is_exhausted() -> None:
    records = [
        CleanSentence("doc-1", "sent-1", "Cùng tiền tố nhưng nội dung thứ nhất."),
        CleanSentence("doc-2", "sent-2", "Cùng tiền tố nhưng nội dung thứ hai."),
        CleanSentence("doc-3", "sent-3", "Cùng tiền tố nhưng nội dung thứ ba."),
    ]

    with pytest.raises(RuntimeError, match="comparison limit"):
        deduplicate_near_sentences(
            records,
            threshold=1.0,
            maximum_comparisons=1,
        )

