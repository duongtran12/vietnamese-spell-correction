"""End-to-end spelling detection and correction metrics."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Sequence

from src.inference import CorrectionResult


class CorrectionPipeline(Protocol):
    def correct_tokens(self, tokens: Sequence[str]) -> CorrectionResult: ...


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_end_to_end(
    pipeline: CorrectionPipeline,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate detection, applied corrections, Top-3, and false corrections."""

    sentence_count = 0
    sentence_matches = 0
    total_error_tokens = 0
    total_correct_tokens = 0
    detected_true_errors = 0
    detected_false_errors = 0
    missed_errors = 0
    changed_tokens = 0
    correct_changes = 0
    false_corrections = 0
    top_three_hits = 0

    for record in records:
        noisy_tokens = [str(token) for token in record["noisy_tokens"]]
        clean_tokens = [str(token) for token in record["clean_tokens"]]
        labels = [int(label) for label in record["labels"]]
        if not (len(noisy_tokens) == len(clean_tokens) == len(labels)):
            raise ValueError("evaluation tokens and labels must be aligned")

        result = pipeline.correct_tokens(noisy_tokens)
        decisions = {decision.token_index: decision for decision in result.decisions}
        detected_indices = set(decisions)
        error_indices = {index for index, label in enumerate(labels) if label == 1}
        correct_indices = set(range(len(labels))) - error_indices

        total_error_tokens += len(error_indices)
        total_correct_tokens += len(correct_indices)
        detected_true_errors += len(detected_indices & error_indices)
        detected_false_errors += len(detected_indices & correct_indices)
        missed_errors += len(error_indices - detected_indices)

        for index, (noisy_token, clean_token, corrected_token) in enumerate(
            zip(noisy_tokens, clean_tokens, result.corrected_tokens, strict=True)
        ):
            if corrected_token != noisy_token:
                changed_tokens += 1
                if corrected_token == clean_token and index in error_indices:
                    correct_changes += 1
                if index in correct_indices:
                    false_corrections += 1

        for index in error_indices:
            decision = decisions.get(index)
            if decision and clean_tokens[index] in {
                suggestion.token for suggestion in decision.suggestions[:3]
            }:
                top_three_hits += 1

        sentence_count += 1
        sentence_matches += list(result.corrected_tokens) == clean_tokens

    detection_precision = _safe_ratio(
        detected_true_errors, detected_true_errors + detected_false_errors
    )
    detection_recall = _safe_ratio(detected_true_errors, total_error_tokens)
    correction_precision = _safe_ratio(correct_changes, changed_tokens)
    correction_recall = _safe_ratio(correct_changes, total_error_tokens)
    return {
        "sentences": sentence_count,
        "error_tokens": total_error_tokens,
        "detection": {
            "true_positive": detected_true_errors,
            "false_positive": detected_false_errors,
            "false_negative": missed_errors,
            "precision": detection_precision,
            "recall": detection_recall,
            "f1": _f1(detection_precision, detection_recall),
        },
        "correction": {
            "correct_changes": correct_changes,
            "all_changes": changed_tokens,
            "precision": correction_precision,
            "recall": correction_recall,
            "f1": _f1(correction_precision, correction_recall),
            "accuracy_at_1": correction_recall,
            "accuracy_at_3": _safe_ratio(top_three_hits, total_error_tokens),
        },
        "sentence_accuracy": _safe_ratio(sentence_matches, sentence_count),
        "false_correction_rate": _safe_ratio(false_corrections, total_correct_tokens),
    }

