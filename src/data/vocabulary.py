"""Deterministic train-only token vocabulary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    frequencies: dict[str, int]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        unknown_id = self.token_to_id[UNK_TOKEN]
        return [self.token_to_id.get(token, unknown_id) for token in tokens]

    def to_dict(self) -> dict[str, Any]:
        return {
            "special_tokens": {"pad": PAD_TOKEN, "unk": UNK_TOKEN},
            "token_to_id": self.token_to_id,
            "frequencies": self.frequencies,
        }


def build_vocabulary(
    token_sequences: Iterable[Sequence[str]],
    *,
    min_frequency: int = 1,
    max_size: int | None = None,
) -> Vocabulary:
    """Build a reproducible vocabulary ordered by frequency then token."""

    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    if max_size is not None and max_size < 2:
        raise ValueError("max_size must leave room for PAD and UNK")

    counts = Counter(token for sequence in token_sequences for token in sequence)
    candidates = [
        (token, frequency)
        for token, frequency in counts.items()
        if frequency >= min_frequency and token not in {PAD_TOKEN, UNK_TOKEN}
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if max_size is not None:
        candidates = candidates[: max_size - 2]

    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    token_to_id.update(
        {token: index for index, (token, _) in enumerate(candidates, start=2)}
    )
    frequencies = {token: frequency for token, frequency in candidates}
    return Vocabulary(token_to_id=token_to_id, frequencies=frequencies)


def oov_statistics(tokens: Sequence[str], vocabulary: Vocabulary) -> dict[str, float | int]:
    """Count tokens that are absent from a fitted vocabulary."""

    oov_count = sum(token not in vocabulary.token_to_id for token in tokens)
    token_count = len(tokens)
    return {
        "oov_count": oov_count,
        "oov_rate": oov_count / token_count if token_count else 0.0,
    }

