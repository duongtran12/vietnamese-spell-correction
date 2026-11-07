import unicodedata

import pytest

from src.data.normalize import normalize_text, tokenize_with_spans


def test_normalize_text_converts_decomposed_vietnamese_to_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "Tiếng Việt")

    normalized = normalize_text(decomposed)

    assert normalized == "Tiếng Việt"
    assert unicodedata.is_normalized("NFC", normalized)


def test_token_spans_map_back_to_normalized_text() -> None:
    text = "Tôi học NLP, bạn nhé!"

    spans = tokenize_with_spans(text)

    assert [token.text for token in spans] == ["Tôi", "học", "NLP", ",", "bạn", "nhé", "!"]
    assert all(text[token.start : token.end] == token.text for token in spans)


def test_normalize_text_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]

