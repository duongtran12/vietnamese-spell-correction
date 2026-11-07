"""Candidate-generator evaluation metrics and latency measurement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Iterable, Sequence

from src.correction.candidates import CandidateGenerator


@dataclass(frozen=True)
class CandidateExample:
    noisy_token: str
    clean_token: str
    error_type: str


def _recall_at_k(rankings: Sequence[Sequence[str]], targets: Sequence[str], k: int) -> float:
    if not targets:
        return 0.0
    hits = sum(target in ranking[:k] for ranking, target in zip(rankings, targets, strict=True))
    return hits / len(targets)


def evaluate_candidate_generator(
    generator: CandidateGenerator,
    examples: Iterable[CandidateExample],
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Measure candidate recall, coverage, list size, and latency."""

    examples = list(examples)
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    maximum_k = max(k_values)

    rankings: list[list[str]] = []
    latencies_ms: list[float] = []
    candidates_per_example: list[int] = []
    for example in examples:
        started = perf_counter()
        candidates = generator.generate(example.noisy_token, top_k=maximum_k)
        latencies_ms.append((perf_counter() - started) * 1000)
        rankings.append([candidate.token for candidate in candidates])
        candidates_per_example.append(len(candidates))

    targets = [example.clean_token for example in examples]
    in_vocabulary_indices = [
        index
        for index, target in enumerate(targets)
        if target in generator.frequencies
    ]
    in_vocabulary_rankings = [rankings[index] for index in in_vocabulary_indices]
    in_vocabulary_targets = [targets[index] for index in in_vocabulary_indices]
    sorted_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    metrics: dict[str, Any] = {
        "examples": len(examples),
        "recall_at_k": {
            str(k): _recall_at_k(rankings, targets, k) for k in sorted(set(k_values))
        },
        "recall_at_k_in_vocabulary": {
            str(k): _recall_at_k(
                in_vocabulary_rankings, in_vocabulary_targets, k
            )
            for k in sorted(set(k_values))
        },
        "in_vocabulary_examples": len(in_vocabulary_indices),
        "target_oov_count": len(examples) - len(in_vocabulary_indices),
        "target_oov_rate": (
            (len(examples) - len(in_vocabulary_indices)) / len(examples)
            if examples
            else 0.0
        ),
        "no_candidate_rate": (
            sum(count == 0 for count in candidates_per_example) / len(examples)
            if examples
            else 0.0
        ),
        "average_candidates": (
            sum(candidates_per_example) / len(examples) if examples else 0.0
        ),
        "latency_ms": {
            "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
            "p95": sorted_latencies[p95_index] if sorted_latencies else 0.0,
        },
    }

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped_indices[example.error_type].append(index)
    metrics["recall_at_5_by_error_type"] = {
        error_type: _recall_at_k(
            [rankings[index] for index in indices],
            [targets[index] for index in indices],
            min(5, maximum_k),
        )
        for error_type, indices in sorted(grouped_indices.items())
    }
    return metrics

