from scripts.tune_inference import select_best_result


def _result(f1: float, false_rate: float, sentence_accuracy: float) -> dict:
    return {
        "detector_threshold": 0.5,
        "minimum_rank_margin": 0.5,
        "metrics": {
            "correction": {"f1": f1},
            "false_correction_rate": false_rate,
            "sentence_accuracy": sentence_accuracy,
        },
    }


def test_select_best_result_respects_false_correction_cap() -> None:
    unsafe = _result(0.8, 0.03, 0.7)
    safe = _result(0.6, 0.01, 0.6)

    best = select_best_result(
        [unsafe, safe],
        max_false_correction_rate=0.02,
    )

    assert best is safe


def test_select_best_result_falls_back_when_no_result_is_under_cap() -> None:
    lower_f1 = _result(0.5, 0.03, 0.7)
    higher_f1 = _result(0.6, 0.04, 0.6)

    best = select_best_result(
        [lower_f1, higher_f1],
        max_false_correction_rate=0.02,
    )

    assert best is higher_f1
