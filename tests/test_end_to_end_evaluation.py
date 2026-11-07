from src.evaluation.end_to_end import evaluate_end_to_end
from tests.test_inference import _pipeline


def test_end_to_end_metrics_cover_detection_correction_and_sentences() -> None:
    records = [
        {
            "noisy_tokens": ["tôi", "đi", "họ", "."],
            "clean_tokens": ["tôi", "đi", "học", "."],
            "labels": [0, 0, 1, 0],
        },
        {
            "noisy_tokens": ["tôi", "đi", "học", "."],
            "clean_tokens": ["tôi", "đi", "học", "."],
            "labels": [0, 0, 0, 0],
        },
    ]

    metrics = evaluate_end_to_end(_pipeline(), records)

    assert metrics["detection"]["f1"] == 1.0
    assert metrics["correction"]["accuracy_at_1"] == 1.0
    assert metrics["correction"]["accuracy_at_3"] == 1.0
    assert metrics["sentence_accuracy"] == 1.0
    assert metrics["false_correction_rate"] == 0.0

