import pytest

from src.data.noise import NoiseEvent
from src.data.schema import CleanSentence, PreparedSentence


def test_clean_sentence_validates_required_fields() -> None:
    record = CleanSentence.from_mapping(
        {"document_id": "doc-1", "sentence_id": "sent-1", "text": "Xin chào."}
    )

    assert record.text == "Xin chào."

    with pytest.raises(ValueError, match="sentence_id"):
        CleanSentence.from_mapping({"document_id": "doc-1", "text": "Thiếu mã câu."})


def test_prepared_sentence_builds_aligned_error_labels() -> None:
    clean = CleanSentence("doc-1", "sent-1", "Tôi học.")
    event = NoiseEvent(1, "học", "họ", "delete")

    prepared = PreparedSentence.create(
        clean,
        ["Tôi", "học", "."],
        ["Tôi", "họ", "."],
        [event],
    )

    assert prepared.labels == (0, 1, 0)
    assert prepared.to_dict()["noise_events"][0]["operation"] == "delete"

