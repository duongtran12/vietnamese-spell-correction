"""Interpolated unigram, bigram, and trigram language model."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Mapping, Sequence


START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<UNK>"


class TrigramLanguageModel:
    """A train-only count language model with add-k smoothing."""

    def __init__(
        self,
        *,
        add_k: float = 0.1,
        lambdas: tuple[float, float, float] = (0.1, 0.3, 0.6),
    ) -> None:
        if add_k <= 0:
            raise ValueError("add_k must be positive")
        if len(lambdas) != 3 or any(weight < 0 for weight in lambdas):
            raise ValueError("lambdas must contain three non-negative weights")
        if not math.isclose(sum(lambdas), 1.0):
            raise ValueError("lambdas must sum to 1")

        self.add_k = add_k
        self.lambdas = lambdas
        self.vocabulary: set[str] = {END_TOKEN, UNK_TOKEN}
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: Counter[tuple[str, str]] = Counter()
        self.trigram_counts: Counter[tuple[str, str, str]] = Counter()
        self.bigram_context_counts: Counter[str] = Counter()
        self.trigram_context_counts: Counter[tuple[str, str]] = Counter()
        self.total_targets = 0
        self.is_fitted = False

    def fit(self, token_sequences: Iterable[Sequence[str]]) -> "TrigramLanguageModel":
        """Fit all counts from clean training token sequences."""

        self.vocabulary = {END_TOKEN, UNK_TOKEN}
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.bigram_context_counts.clear()
        self.trigram_context_counts.clear()
        self.total_targets = 0

        sequences = [list(tokens) for tokens in token_sequences]
        self.vocabulary.update(token for tokens in sequences for token in tokens)
        for tokens in sequences:
            padded = [START_TOKEN, START_TOKEN, *tokens, END_TOKEN]
            for position in range(2, len(padded)):
                history_2 = padded[position - 2]
                history_1 = padded[position - 1]
                target = padded[position]
                self.unigram_counts[target] += 1
                self.bigram_counts[(history_1, target)] += 1
                self.trigram_counts[(history_2, history_1, target)] += 1
                self.bigram_context_counts[history_1] += 1
                self.trigram_context_counts[(history_2, history_1)] += 1
                self.total_targets += 1

        self.is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("language model must be fitted before scoring")

    def _map_token(self, token: str) -> str:
        if token in {START_TOKEN, END_TOKEN}:
            return token
        return token if token in self.vocabulary else UNK_TOKEN

    def probability(self, target: str, history_2: str, history_1: str) -> float:
        """Return interpolated smoothed probability for one target token."""

        self._require_fitted()
        target = self._map_token(target)
        history_2 = self._map_token(history_2)
        history_1 = self._map_token(history_1)
        vocabulary_size = len(self.vocabulary)

        unigram = (self.unigram_counts[target] + self.add_k) / (
            self.total_targets + self.add_k * vocabulary_size
        )
        bigram = (self.bigram_counts[(history_1, target)] + self.add_k) / (
            self.bigram_context_counts[history_1] + self.add_k * vocabulary_size
        )
        trigram = (
            self.trigram_counts[(history_2, history_1, target)] + self.add_k
        ) / (
            self.trigram_context_counts[(history_2, history_1)]
            + self.add_k * vocabulary_size
        )
        lambda_unigram, lambda_bigram, lambda_trigram = self.lambdas
        return (
            lambda_unigram * unigram
            + lambda_bigram * bigram
            + lambda_trigram * trigram
        )

    def sentence_log_probability(self, tokens: Sequence[str]) -> float:
        """Score a complete sentence, including its end token."""

        self._require_fitted()
        padded = [START_TOKEN, START_TOKEN, *tokens, END_TOKEN]
        return sum(
            math.log(
                self.probability(
                    padded[position],
                    padded[position - 2],
                    padded[position - 1],
                )
            )
            for position in range(2, len(padded))
        )

    def replacement_log_probability(
        self,
        tokens: Sequence[str],
        token_index: int,
        candidate: str,
    ) -> float:
        """Score only predictions affected by replacing one token."""

        self._require_fitted()
        if token_index < 0 or token_index >= len(tokens):
            raise IndexError("token_index is outside the sentence")

        replaced = list(tokens)
        replaced[token_index] = candidate
        padded = [START_TOKEN, START_TOKEN, *replaced, END_TOKEN]
        replacement_position = token_index + 2
        affected_positions = range(
            replacement_position,
            min(replacement_position + 3, len(padded)),
        )
        return sum(
            math.log(
                self.probability(
                    padded[position],
                    padded[position - 2],
                    padded[position - 1],
                )
            )
            for position in affected_positions
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fitted model without external pretrained artifacts."""

        self._require_fitted()
        return {
            "add_k": self.add_k,
            "lambdas": list(self.lambdas),
            "vocabulary": sorted(self.vocabulary),
            "total_targets": self.total_targets,
            "unigram_counts": dict(self.unigram_counts),
            "bigram_counts": [
                {"tokens": list(tokens), "count": count}
                for tokens, count in sorted(self.bigram_counts.items())
            ],
            "trigram_counts": [
                {"tokens": list(tokens), "count": count}
                for tokens, count in sorted(self.trigram_counts.items())
            ],
            "bigram_context_counts": dict(self.bigram_context_counts),
            "trigram_context_counts": [
                {"tokens": list(tokens), "count": count}
                for tokens, count in sorted(self.trigram_context_counts.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrigramLanguageModel":
        """Restore a model previously produced by ``to_dict``."""

        model = cls(
            add_k=float(data["add_k"]),
            lambdas=tuple(float(value) for value in data["lambdas"]),  # type: ignore[arg-type]
        )
        model.vocabulary = set(data["vocabulary"])
        model.total_targets = int(data["total_targets"])
        model.unigram_counts.update(data["unigram_counts"])
        model.bigram_counts.update(
            {tuple(item["tokens"]): int(item["count"]) for item in data["bigram_counts"]}
        )
        model.trigram_counts.update(
            {tuple(item["tokens"]): int(item["count"]) for item in data["trigram_counts"]}
        )
        model.bigram_context_counts.update(data["bigram_context_counts"])
        model.trigram_context_counts.update(
            {
                tuple(item["tokens"]): int(item["count"])
                for item in data["trigram_context_counts"]
            }
        )
        model.is_fitted = True
        return model

