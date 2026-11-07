import math

import pytest

from src.models.trigram_lm import TrigramLanguageModel


def _model() -> TrigramLanguageModel:
    return TrigramLanguageModel().fit(
        [
            ["tôi", "đi", "học"],
            ["tôi", "đi", "học"],
            ["tôi", "đi", "làm"],
            ["bạn", "đang", "học"],
        ]
    )


def test_trigram_probability_is_smoothed_and_context_sensitive() -> None:
    model = _model()

    known = model.probability("học", "tôi", "đi")
    less_likely = model.probability("làm", "tôi", "đi")
    unseen = model.probability("không-có", "tôi", "đi")

    assert known > less_likely > unseen > 0.0


def test_replacement_score_prefers_observed_context() -> None:
    model = _model()
    tokens = ["tôi", "đi", "hco"]

    assert model.replacement_log_probability(tokens, 2, "học") > model.replacement_log_probability(
        tokens, 2, "làm"
    )
    assert math.isfinite(model.sentence_log_probability(tokens))


def test_language_model_round_trip_preserves_scores() -> None:
    model = _model()
    restored = TrigramLanguageModel.from_dict(model.to_dict())

    assert restored.sentence_log_probability(["tôi", "đi", "học"]) == pytest.approx(
        model.sentence_log_probability(["tôi", "đi", "học"])
    )


def test_language_model_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        TrigramLanguageModel().sentence_log_probability(["tôi"])

