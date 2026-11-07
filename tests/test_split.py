import pytest

from src.data.split import assert_no_document_leakage, split_by_document


def _records() -> list[dict[str, object]]:
    return [
        {"document_id": f"doc-{index}", "text": f"Câu số {index}."}
        for index in range(10)
    ] + [{"document_id": "doc-0", "text": "Câu thứ hai cùng tài liệu."}]


def test_split_is_deterministic_and_keeps_documents_together() -> None:
    first = split_by_document(_records(), seed=42)
    second = split_by_document(_records(), seed=42)

    assert first == second
    assert {
        name: len({record["document_id"] for record in records})
        for name, records in first.items()
    } == {
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    assert_no_document_leakage(first)


def test_split_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        split_by_document(_records(), train_ratio=0.8, validation_ratio=0.2, test_ratio=0.2)


def test_leakage_check_detects_reused_document() -> None:
    splits = {
        "train": [{"document_id": "shared"}],
        "validation": [{"document_id": "shared"}],
        "test": [],
    }

    with pytest.raises(ValueError, match="appears in both"):
        assert_no_document_leakage(splits)

