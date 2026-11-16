from scripts.analyze_errors import plot_confusion_matrix, plot_outcomes, render_markdown
from src.evaluation.error_analysis import analyze_errors
from tests.test_inference import _pipeline


def _records() -> list[dict]:
    return [
        {
            "sentence_id": "s1",
            "noisy_tokens": ["tôi", "đi", "họ", "."],
            "clean_tokens": ["tôi", "đi", "học", "."],
            "labels": [0, 0, 1, 0],
            "noise_events": [
                {
                    "token_index": 2,
                    "clean_token": "học",
                    "noisy_token": "họ",
                    "operation": "delete",
                }
            ],
        },
        {
            "sentence_id": "s2",
            "noisy_tokens": ["tôi", "đi", "học", "."],
            "clean_tokens": ["tôi", "đi", "học", "."],
            "labels": [0, 0, 0, 0],
            "noise_events": [],
        },
    ]


def test_error_analysis_classifies_corrections_and_detector_counts() -> None:
    report = analyze_errors(_pipeline(), _records())

    assert report["outcome_counts"]["correct_correction"] == 1
    assert report["outcome_counts"]["false_correction"] == 0
    assert report["detector_confusion_matrix"] == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 7,
    }
    assert report["by_error_type"]["delete"]["correction_recall"] == 1.0


def test_error_analysis_outputs_markdown_and_figures(tmp_path) -> None:
    report = analyze_errors(_pipeline(), _records())
    confusion_path = tmp_path / "confusion.png"
    outcomes_path = tmp_path / "outcomes.png"

    markdown = render_markdown(report)
    plot_confusion_matrix(report["detector_confusion_matrix"], confusion_path)
    plot_outcomes(report["outcome_counts"], outcomes_path)

    assert "Correct corrections: 1" in markdown
    assert confusion_path.stat().st_size > 0
    assert outcomes_path.stat().st_size > 0
