"""Train-only dictionary and edit-distance candidate generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

from src.correction.edit_distance import (
    damerau_levenshtein_distance,
    levenshtein_distance,
)


DistanceFunction = Callable[[str, str], int]
DISTANCE_FUNCTIONS: dict[str, DistanceFunction] = {
    "levenshtein": levenshtein_distance,
    "damerau_levenshtein": damerau_levenshtein_distance,
}


@dataclass(frozen=True)
class Candidate:
    token: str
    distance: int
    frequency: int
    edit_score: float
    frequency_score: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def build_training_dictionary(
    token_sequences: Iterable[Sequence[str]],
) -> dict[str, int]:
    """Count alphabetic tokens from clean training sentences only."""

    counts = Counter(
        token
        for sequence in token_sequences
        for token in sequence
        if token.isalpha()
    )
    return dict(sorted(counts.items()))


class CandidateGenerator:
    """Generate candidates from a fixed frequency dictionary."""

    def __init__(
        self,
        frequencies: Mapping[str, int],
        *,
        distance: str = "damerau_levenshtein",
        short_token_max_length: int = 4,
        short_token_max_distance: int = 1,
        long_token_max_distance: int = 2,
    ) -> None:
        if distance not in DISTANCE_FUNCTIONS:
            raise ValueError(f"unsupported distance function: {distance}")
        if short_token_max_length < 1:
            raise ValueError("short_token_max_length must be positive")
        if short_token_max_distance < 0 or long_token_max_distance < 0:
            raise ValueError("maximum distances must be non-negative")
        if any(frequency < 1 for frequency in frequencies.values()):
            raise ValueError("dictionary frequencies must be positive")

        self.frequencies = dict(frequencies)
        self.distance_function = DISTANCE_FUNCTIONS[distance]
        self.short_token_max_length = short_token_max_length
        self.short_token_max_distance = short_token_max_distance
        self.long_token_max_distance = long_token_max_distance
        self._tokens_by_length: dict[int, list[str]] = defaultdict(list)
        for token in sorted(self.frequencies):
            self._tokens_by_length[len(token)].append(token)

    def max_distance_for(self, token: str) -> int:
        if len(token) <= self.short_token_max_length:
            return self.short_token_max_distance
        return self.long_token_max_distance

    def generate(self, token: str, *, top_k: int = 5) -> list[Candidate]:
        """Return baseline candidates ranked by distance then frequency."""

        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        max_distance = self.max_distance_for(token)
        candidates: list[Candidate] = []
        minimum_length = max(1, len(token) - max_distance)
        maximum_length = len(token) + max_distance
        for length in range(minimum_length, maximum_length + 1):
            for candidate_token in self._tokens_by_length.get(length, []):
                distance = self.distance_function(token, candidate_token)
                if distance > max_distance:
                    continue
                frequency = self.frequencies[candidate_token]
                candidates.append(
                    Candidate(
                        token=candidate_token,
                        distance=distance,
                        frequency=frequency,
                        edit_score=-distance / max(len(token), len(candidate_token), 1),
                        frequency_score=math.log1p(frequency),
                    )
                )

        candidates.sort(
            key=lambda candidate: (
                candidate.distance,
                -candidate.frequency,
                candidate.token,
            )
        )
        return candidates[:top_k]

