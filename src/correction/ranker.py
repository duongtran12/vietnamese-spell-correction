"""Context-aware ranking of edit-distance candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from src.correction.candidates import Candidate
from src.models.trigram_lm import TrigramLanguageModel


@dataclass(frozen=True)
class RankedCandidate:
    token: str
    distance: int
    frequency: int
    edit_score: float
    frequency_score: float
    context_log_probability: float
    total_score: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


class CandidateRanker:
    """Combine context, edit, and train-frequency scores."""

    def __init__(
        self,
        language_model: TrigramLanguageModel,
        *,
        context_weight: float = 1.0,
        edit_weight: float = 1.0,
        frequency_weight: float = 0.1,
    ) -> None:
        self.language_model = language_model
        self.context_weight = context_weight
        self.edit_weight = edit_weight
        self.frequency_weight = frequency_weight

    def rank(
        self,
        tokens: Sequence[str],
        token_index: int,
        candidates: Sequence[Candidate],
        *,
        top_k: int = 3,
    ) -> list[RankedCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be positive")

        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            context_score = self.language_model.replacement_log_probability(
                tokens, token_index, candidate.token
            )
            total_score = (
                self.context_weight * context_score
                + self.edit_weight * candidate.edit_score
                + self.frequency_weight * candidate.frequency_score
            )
            ranked.append(
                RankedCandidate(
                    token=candidate.token,
                    distance=candidate.distance,
                    frequency=candidate.frequency,
                    edit_score=candidate.edit_score,
                    frequency_score=candidate.frequency_score,
                    context_log_probability=context_score,
                    total_score=total_score,
                )
            )

        ranked.sort(
            key=lambda candidate: (
                -candidate.total_score,
                candidate.distance,
                -candidate.frequency,
                candidate.token,
            )
        )
        return ranked[:top_k]

