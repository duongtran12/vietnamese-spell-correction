import pytest

from src.data.vocabulary import PAD_TOKEN, UNK_TOKEN, build_vocabulary, oov_statistics


def test_vocabulary_is_deterministic_and_frequency_ordered() -> None:
    vocabulary = build_vocabulary([["tôi", "học"], ["tôi", "NLP"]])

    assert vocabulary.token_to_id == {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        "tôi": 2,
        "NLP": 3,
        "học": 4,
    }
    assert vocabulary.encode(["tôi", "mới"]) == [2, 1]
    assert oov_statistics(["tôi", "mới"], vocabulary) == {
        "oov_count": 1,
        "oov_rate": 0.5,
    }


def test_vocabulary_validates_size_limit() -> None:
    with pytest.raises(ValueError, match="PAD and UNK"):
        build_vocabulary([["tôi"]], max_size=1)

