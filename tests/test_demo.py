from dataclasses import replace

from src.demo import (
    apply_token_choices,
    decision_rows,
    render_highlighted_text,
    suggestion_rows,
)
from src.inference import TokenDecision
from tests.test_inference import _pipeline


def _decision() -> TokenDecision:
    result = _pipeline().correct_text("tôi đi họ .")
    return result.decisions[0]


def test_render_highlighted_text_marks_decisions_and_escapes_html() -> None:
    decision = _decision()

    rendered = render_highlighted_text("tôi đi họ <script>", [decision])

    assert "<mark" in rendered
    assert "họ" in rendered
    assert "&lt;script&gt;" in rendered


def test_apply_token_choices_preserves_spacing_and_punctuation() -> None:
    corrected = apply_token_choices("tôi  đi họ!", {2: "học"})

    assert corrected == "tôi  đi học!"


def test_demo_rows_include_decision_and_score_details() -> None:
    decision = _decision()
    decision = replace(decision, detector_confidence=0.98765)

    summary = decision_rows([decision])
    details = suggestion_rows(decision)

    assert summary[0]["Confidence"] == 0.9877
    assert summary[0]["Token"] == "họ"
    assert details[0]["Ứng viên"] == "học"
    assert "Tổng điểm" in details[0]
