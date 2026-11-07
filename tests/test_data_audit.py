from src.data.audit import audit_clean_corpus
from src.data.schema import CleanSentence


def test_data_audit_reports_quality_and_near_duplicates() -> None:
    records = [
        CleanSentence("doc-1", "sent-1", "Tiếng Việt rất đẹp.", "test"),
        CleanSentence("doc-2", "sent-2", "Tiếng Việt rất đẹp.", "test"),
        CleanSentence("doc-3", "sent-3", "Tiếng Việt rất đẹp!", "test"),
        CleanSentence("doc-4", "sent-4", "Xem https://example.com", "test"),
    ]

    report = audit_clean_corpus(records, near_duplicate_threshold=0.85)

    assert report["documents"] == 4
    assert report["sentences"] == 4
    assert report["non_nfc_sentences"] == 1
    assert report["exact_duplicate_records"] == 1
    assert report["sentences_with_url"] == 1
    assert report["near_duplicates"]["pairs_found"] >= 1


def test_data_audit_handles_empty_corpus() -> None:
    report = audit_clean_corpus([])

    assert report["sentences"] == 0
    assert report["token_length"]["p95"] == 0.0
    assert report["near_duplicates"]["comparisons"] == 0

