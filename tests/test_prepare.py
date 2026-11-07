from src.data.prepare import prepare_dataset
from src.data.schema import CleanSentence


def _corpus() -> list[CleanSentence]:
    records = [
        CleanSentence(
            document_id=f"doc-{index}",
            sentence_id=f"sent-{index}",
            text=f"Đây là nội dung tài liệu số {index}.",
            source="synthetic-test",
        )
        for index in range(10)
    ]
    records.append(
        CleanSentence(
            document_id="duplicate-doc",
            sentence_id="duplicate-sent",
            text="Đây  là nội dung tài liệu số 0.",
            source="synthetic-test",
        )
    )
    return records


def test_prepare_dataset_is_reproducible_and_train_only() -> None:
    first = prepare_dataset(
        _corpus(), seed=42, sentence_error_probability=1.0
    )
    second = prepare_dataset(
        _corpus(), seed=42, sentence_error_probability=1.0
    )

    assert first == second
    assert first.report["exact_duplicates_removed"] == 1
    assert sum(len(records) for records in first.splits.values()) == 10
    assert all(
        len(record.clean_tokens) == len(record.noisy_tokens) == len(record.labels)
        for records in first.splits.values()
        for record in records
    )

    train_tokens = {
        token for record in first.splits["train"] for token in record.clean_tokens
    }
    learned_tokens = set(first.vocabulary.token_to_id) - {"<PAD>", "<UNK>"}
    assert learned_tokens == train_tokens
    assert first.report["splits"]["train"]["oov_count"] == 0

