"""Unicode normalization and lightweight tokenization utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class TokenSpan:
    """A token and its character offsets in the normalized text."""

    text: str
    start: int
    end: int


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text to Unicode NFC without changing its content."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return unicodedata.normalize("NFC", text)


def tokenize_with_spans(text: str) -> list[TokenSpan]:
    """Tokenize words and punctuation while preserving source offsets."""

    normalized = normalize_text(text)
    return [
        TokenSpan(match.group(), match.start(), match.end())
        for match in TOKEN_PATTERN.finditer(normalized)
    ]


def tokenize(text: str) -> list[str]:
    """Return token strings using the same rules as ``tokenize_with_spans``."""

    return [token.text for token in tokenize_with_spans(text)]

