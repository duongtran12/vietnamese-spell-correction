import pytest

from src.data.noise import generate_noisy_tokens


def test_noise_is_reproducible_and_records_metadata() -> None:
    tokens = ["Tôi", "đang", "học", "tiếng", "Việt", "."]

    first = generate_noisy_tokens(tokens, seed=7, sentence_error_probability=1.0)
    second = generate_noisy_tokens(tokens, seed=7, sentence_error_probability=1.0)

    assert first == second
    assert len(first.events) == 1
    event = first.events[0]
    assert first.tokens[event.token_index] == event.noisy_token
    assert tokens[event.token_index] == event.clean_token
    assert event.clean_token != event.noisy_token
    assert tokens == ["Tôi", "đang", "học", "tiếng", "Việt", "."]


def test_noise_can_leave_a_clean_sentence_unchanged() -> None:
    tokens = ["Câu", "đúng", "."]

    result = generate_noisy_tokens(tokens, seed=1, sentence_error_probability=0.0)

    assert result.tokens == tokens
    assert result.events == []


def test_noise_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        generate_noisy_tokens(["xin", "chào"], seed=1, sentence_error_probability=1.1)

