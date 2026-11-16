"""Structured end-to-end error analysis for spelling correction."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Protocol, Sequence

from src.inference import CorrectionResult


class CorrectionPipeline(Protocol):
    def correct_tokens(self, tokens: Sequence[str]) -> CorrectionResult: ...


CATEGORIES = (
    "correct_correction",
    "wrong_correction",
    "detected_kept",
    "missed_detection",
    "false_correction",
    "false_positive_kept",
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _example(
    record: Mapping[str, Any],
    *,
    token_index: int,
    operation: str | None,
    noisy_token: str,
    clean_token: str,
    applied_token: str,
    decision: Any | None,
) -> dict[str, Any]:
    noisy_tokens = [str(token) for token in record["noisy_tokens"]]
    window_start = max(0, token_index - 4)
    window_end = min(len(noisy_tokens), token_index + 5)
    return {
        "sentence_id": str(record.get("sentence_id", "")),
        "document_title": record.get("document_title"),
        "token_index": token_index,
        "operation": operation,
        "noisy_token": noisy_token,
        "clean_token": clean_token,
        "applied_token": applied_token,
        "detector_confidence": (
            float(decision.detector_confidence) if decision is not None else None
        ),
        "suggestions": (
            [suggestion.token for suggestion in decision.suggestions]
            if decision is not None
            else []
        ),
        "context": " ".join(noisy_tokens[window_start:window_end]),
    }


def analyze_errors(
    pipeline: CorrectionPipeline,
    records: Iterable[Mapping[str, Any]],
    *,
    max_examples_per_category: int = 10,
) -> dict[str, Any]:
    """Classify detector and correction outcomes with representative examples."""

    if max_examples_per_category < 1:
        raise ValueError("max_examples_per_category must be positive")
    counts = {category: 0 for category in CATEGORIES}
    examples: dict[str, list[dict[str, Any]]] = {
        category: [] for category in CATEGORIES
    }
    operation_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "detected": 0,
            "top3_hit": 0,
            "corrected": 0,
            "wrong_correction": 0,
            "detected_kept": 0,
            "missed_detection": 0,
        }
    )
    true_positive = false_positive = false_negative = true_negative = 0
    sentence_count = 0

    for record in records:
        noisy_tokens = [str(token) for token in record["noisy_tokens"]]
        clean_tokens = [str(token) for token in record["clean_tokens"]]
        labels = [int(label) for label in record["labels"]]
        if not (len(noisy_tokens) == len(clean_tokens) == len(labels)):
            raise ValueError("analysis tokens and labels must be aligned")
        operations = {
            int(event["token_index"]): str(event["operation"])
            for event in record.get("noise_events", [])
        }
        result = pipeline.correct_tokens(noisy_tokens)
        decisions = {decision.token_index: decision for decision in result.decisions}

        for token_index, (noisy_token, clean_token, label, applied_token) in enumerate(
            zip(
                noisy_tokens,
                clean_tokens,
                labels,
                result.corrected_tokens,
                strict=True,
            )
        ):
            decision = decisions.get(token_index)
            detected = decision is not None
            if label == 1:
                if detected:
                    true_positive += 1
                else:
                    false_negative += 1
                operation = operations.get(token_index, "unknown")
                operation_data = operation_counts[operation]
                operation_data["total"] += 1
                operation_data["detected"] += detected
                if decision is not None and clean_token in {
                    suggestion.token for suggestion in decision.suggestions[:3]
                }:
                    operation_data["top3_hit"] += 1

                if applied_token == clean_token:
                    category = "correct_correction"
                    operation_data["corrected"] += 1
                elif not detected:
                    category = "missed_detection"
                    operation_data["missed_detection"] += 1
                elif applied_token == noisy_token:
                    category = "detected_kept"
                    operation_data["detected_kept"] += 1
                else:
                    category = "wrong_correction"
                    operation_data["wrong_correction"] += 1
            else:
                operation = None
                if detected:
                    false_positive += 1
                else:
                    true_negative += 1
                if applied_token != noisy_token:
                    category = "false_correction"
                elif detected:
                    category = "false_positive_kept"
                else:
                    continue

            counts[category] += 1
            if len(examples[category]) < max_examples_per_category:
                examples[category].append(
                    _example(
                        record,
                        token_index=token_index,
                        operation=operation,
                        noisy_token=noisy_token,
                        clean_token=clean_token,
                        applied_token=applied_token,
                        decision=decision,
                    )
                )
        sentence_count += 1

    operations_report: dict[str, dict[str, int | float]] = {}
    for operation, data in sorted(operation_counts.items()):
        total = data["total"]
        operations_report[operation] = {
            **data,
            "detection_recall": _safe_ratio(data["detected"], total),
            "top3_recall": _safe_ratio(data["top3_hit"], total),
            "correction_recall": _safe_ratio(data["corrected"], total),
        }

    return {
        "sentences": sentence_count,
        "detector_confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "outcome_counts": counts,
        "by_error_type": operations_report,
        "examples": examples,
    }
