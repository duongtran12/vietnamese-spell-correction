"""Validated records used by the data preparation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.data.noise import NoiseEvent


def _required_string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class CleanSentence:
    """One clean sentence and the document it belongs to."""

    document_id: str
    sentence_id: str
    text: str
    source: str | None = None
    document_title: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CleanSentence":
        source = data.get("source")
        if source is not None and not isinstance(source, str):
            raise ValueError("source must be a string or null")
        document_title = data.get("document_title")
        if document_title is not None and not isinstance(document_title, str):
            raise ValueError("document_title must be a string or null")
        return cls(
            document_id=_required_string(data, "document_id"),
            sentence_id=_required_string(data, "sentence_id"),
            text=_required_string(data, "text"),
            source=source.strip() if source else None,
            document_title=document_title.strip() if document_title else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedSentence:
    """A clean/noisy aligned sentence ready for detector training."""

    document_id: str
    sentence_id: str
    source: str | None
    document_title: str | None
    clean_text: str
    clean_tokens: tuple[str, ...]
    noisy_tokens: tuple[str, ...]
    labels: tuple[int, ...]
    noise_events: tuple[NoiseEvent, ...]

    @classmethod
    def create(
        cls,
        clean: CleanSentence,
        clean_tokens: Sequence[str],
        noisy_tokens: Sequence[str],
        noise_events: Sequence[NoiseEvent],
    ) -> "PreparedSentence":
        if len(clean_tokens) != len(noisy_tokens):
            raise ValueError("clean and noisy token sequences must have equal length")

        error_indices = {event.token_index for event in noise_events}
        if any(index < 0 or index >= len(clean_tokens) for index in error_indices):
            raise ValueError("noise event index is outside the token sequence")

        labels = tuple(
            1 if index in error_indices else 0 for index in range(len(clean_tokens))
        )
        return cls(
            document_id=clean.document_id,
            sentence_id=clean.sentence_id,
            source=clean.source,
            document_title=clean.document_title,
            clean_text=clean.text,
            clean_tokens=tuple(clean_tokens),
            noisy_tokens=tuple(noisy_tokens),
            labels=labels,
            noise_events=tuple(noise_events),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "sentence_id": self.sentence_id,
            "source": self.source,
            "document_title": self.document_title,
            "clean_text": self.clean_text,
            "clean_tokens": list(self.clean_tokens),
            "noisy_tokens": list(self.noisy_tokens),
            "labels": list(self.labels),
            "noise_events": [asdict(event) for event in self.noise_events],
        }

