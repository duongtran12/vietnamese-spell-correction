"""Presentation helpers shared by the local Streamlit demo and tests."""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence

from src.data.normalize import normalize_text, tokenize_with_spans
from src.inference import TokenDecision


def render_highlighted_text(
    text: str,
    decisions: Sequence[TokenDecision],
) -> str:
    """Return safe HTML with detector decisions highlighted by token span."""

    normalized = normalize_text(text)
    spans = tokenize_with_spans(normalized)
    by_index = {decision.token_index: decision for decision in decisions}
    pieces: list[str] = []
    cursor = 0
    for token_index, span in enumerate(spans):
        pieces.append(html.escape(normalized[cursor : span.start]))
        token_html = html.escape(normalized[span.start : span.end])
        decision = by_index.get(token_index)
        if decision is None:
            pieces.append(token_html)
        else:
            background = "#fecaca" if decision.changed else "#fde68a"
            title = html.escape(
                f"Confidence: {decision.detector_confidence:.3f}; "
                f"applied: {decision.applied_token}",
                quote=True,
            )
            pieces.append(
                f'<mark title="{title}" style="background:{background};'
                f'padding:0.1rem 0.2rem;border-radius:0.25rem">{token_html}</mark>'
            )
        cursor = span.end
    pieces.append(html.escape(normalized[cursor:]))
    return "".join(pieces)


def apply_token_choices(text: str, choices: Mapping[int, str]) -> str:
    """Replace selected token indices while preserving original whitespace."""

    normalized = normalize_text(text)
    spans = tokenize_with_spans(normalized)
    pieces: list[str] = []
    cursor = 0
    for token_index, span in enumerate(spans):
        pieces.append(normalized[cursor : span.start])
        pieces.append(choices.get(token_index, span.text))
        cursor = span.end
    pieces.append(normalized[cursor:])
    return "".join(pieces)


def decision_rows(decisions: Sequence[TokenDecision]) -> list[dict[str, Any]]:
    """Build compact rows for the detector-decision table."""

    return [
        {
            "Vị trí": decision.token_index,
            "Token": decision.original_token,
            "Confidence": round(decision.detector_confidence, 4),
            "Top-3": ", ".join(
                suggestion.token for suggestion in decision.suggestions[:3]
            ),
            "Áp dụng": decision.applied_token,
            "Trạng thái": "Đã sửa" if decision.changed else "Giữ nguyên",
        }
        for decision in decisions
    ]


def suggestion_rows(decision: TokenDecision) -> list[dict[str, Any]]:
    """Expose ranker score components for one detector decision."""

    return [
        {
            "Ứng viên": suggestion.token,
            "Tổng điểm": round(suggestion.total_score, 4),
            "Ngữ cảnh": round(suggestion.context_log_probability, 4),
            "Edit score": round(suggestion.edit_score, 4),
            "Tần suất": suggestion.frequency,
            "Khoảng cách": suggestion.distance,
        }
        for suggestion in decision.suggestions
    ]
