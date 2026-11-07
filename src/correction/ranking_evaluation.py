"""Comparison of edit-only and context-aware candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.correction.candidates import CandidateGenerator
from src.correction.ranker import CandidateRanker


@dataclass(frozen=True)
class RankingExample:
    tokens: tuple[str, ...]
    token_index: int
    clean_token: str
    error_type: str


def _accuracy_at_k(
    rankings: Sequence[Sequence[str]], targets: Sequence[str], k: int
) -> float:
    if not targets:
        return 0.0
    return sum(
        target in ranking[:k]
        for ranking, target in zip(rankings, targets, strict=True)
    ) / len(targets)


def evaluate_ranker(
    generator: CandidateGenerator,
    ranker: CandidateRanker,
    examples: Iterable[RankingExample],
    *,
    candidate_top_k: int = 5,
    accuracy_k_values: Sequence[int] = (1, 3),
) -> dict[str, Any]:
    """Evaluate the context ranker against edit-distance ordering."""

    examples = list(examples)
    if candidate_top_k < 1:
        raise ValueError("candidate_top_k must be positive")
    if not accuracy_k_values or any(k < 1 for k in accuracy_k_values):
        raise ValueError("accuracy_k_values must contain positive integers")

    baseline_rankings: list[list[str]] = []
    context_rankings: list[list[str]] = []
    for example in examples:
        candidates = generator.generate(
            example.tokens[example.token_index], top_k=candidate_top_k
        )
        baseline_rankings.append([candidate.token for candidate in candidates])
        ranked = ranker.rank(
            example.tokens,
            example.token_index,
            candidates,
            top_k=candidate_top_k,
        )
        context_rankings.append([candidate.token for candidate in ranked])

    targets = [example.clean_token for example in examples]
    in_vocabulary_indices = [
        index for index, target in enumerate(targets) if target in generator.frequencies
    ]
    metrics: dict[str, Any] = {
        "examples": len(examples),
        "in_vocabulary_examples": len(in_vocabulary_indices),
        "target_oov_rate": (
            (len(examples) - len(in_vocabulary_indices)) / len(examples)
            if examples
            else 0.0
        ),
        "candidate_recall_at_k": _accuracy_at_k(
            baseline_rankings, targets, candidate_top_k
        ),
        "baseline_accuracy_at_k": {
            str(k): _accuracy_at_k(baseline_rankings, targets, k)
            for k in sorted(set(accuracy_k_values))
        },
        "context_accuracy_at_k": {
            str(k): _accuracy_at_k(context_rankings, targets, k)
            for k in sorted(set(accuracy_k_values))
        },
    }
    metrics["accuracy_at_1_delta"] = (
        metrics["context_accuracy_at_k"].get("1", 0.0)
        - metrics["baseline_accuracy_at_k"].get("1", 0.0)
    )
    return metrics

