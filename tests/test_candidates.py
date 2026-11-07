import pytest

from src.correction.candidates import CandidateGenerator, build_training_dictionary


def test_training_dictionary_counts_only_alphabetic_tokens() -> None:
    frequencies = build_training_dictionary(
        [["Tôi", "học", "."], ["học", "NLP", "2026"]]
    )

    assert frequencies == {"NLP": 1, "Tôi": 1, "học": 2}


def test_candidate_generator_handles_transposition_and_frequency_ties() -> None:
    generator = CandidateGenerator({"hoc": 5, "học": 10, "họp": 20})

    transposed = generator.generate("hco", top_k=3)
    tied = generator.generate("họ", top_k=3)

    assert transposed[0].token == "hoc"
    assert transposed[0].distance == 1
    assert [candidate.token for candidate in tied[:2]] == ["họp", "học"]


def test_candidate_generator_uses_length_based_distance_limit() -> None:
    generator = CandidateGenerator({"nghiên": 2, "nghiêng": 4})

    assert generator.max_distance_for("học") == 1
    assert generator.max_distance_for("nghiên") == 2
    assert generator.generate("nghiê", top_k=5)[0].token == "nghiên"


def test_candidate_generator_validates_top_k() -> None:
    generator = CandidateGenerator({"học": 1})

    with pytest.raises(ValueError, match="positive"):
        generator.generate("họ", top_k=0)

