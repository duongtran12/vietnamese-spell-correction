"""Reproducible synthetic spelling-noise generation at token level."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


VIETNAMESE_LETTERS = "aăâbcdđeêghiklmnoôơpqrstuưvxy"
DIACRITIC_GROUPS = (
    "aàáảãạ",
    "ăằắẳẵặ",
    "âầấẩẫậ",
    "eèéẻẽẹ",
    "êềếểễệ",
    "iìíỉĩị",
    "oòóỏõọ",
    "ôồốổỗộ",
    "ơờớởỡợ",
    "uùúủũụ",
    "ưừứửữự",
    "yỳýỷỹỵ",
)


def _build_diacritic_choices() -> dict[str, str]:
    choices: dict[str, str] = {}
    for group in DIACRITIC_GROUPS:
        for character in group:
            choices[character] = group.replace(character, "")
            choices[character.upper()] = group.upper().replace(character.upper(), "")
    return choices


DIACRITIC_CHOICES = _build_diacritic_choices()


@dataclass(frozen=True)
class NoiseEvent:
    """Metadata describing one token corruption."""

    token_index: int
    clean_token: str
    noisy_token: str
    operation: str


@dataclass(frozen=True)
class NoisyTokens:
    """A corrupted token sequence and its traceable noise events."""

    tokens: list[str]
    events: list[NoiseEvent]


def _different_letter(current: str, rng: random.Random) -> str:
    alphabet = VIETNAMESE_LETTERS.upper() if current.isupper() else VIETNAMESE_LETTERS
    choices = alphabet.replace(current, "")
    return rng.choice(choices)


def _apply_operation(token: str, operation: str, rng: random.Random) -> str:
    if operation == "delete":
        index = rng.randrange(len(token))
        return token[:index] + token[index + 1 :]

    if operation == "insert":
        index = rng.randrange(len(token) + 1)
        character = rng.choice(VIETNAMESE_LETTERS)
        return token[:index] + character + token[index:]

    if operation == "substitute":
        index = rng.randrange(len(token))
        character = _different_letter(token[index], rng)
        return token[:index] + character + token[index + 1 :]

    if operation == "transpose":
        candidates = [
            index for index in range(len(token) - 1) if token[index] != token[index + 1]
        ]
        if candidates:
            index = rng.choice(candidates)
            return (
                token[:index]
                + token[index + 1]
                + token[index]
                + token[index + 2 :]
            )
        return _apply_operation(token, "substitute", rng)

    if operation == "diacritic":
        candidates = [
            index for index, character in enumerate(token) if character in DIACRITIC_CHOICES
        ]
        if candidates:
            index = rng.choice(candidates)
            replacement = rng.choice(DIACRITIC_CHOICES[token[index]])
            return token[:index] + replacement + token[index + 1 :]
        return _apply_operation(token, "substitute", rng)

    raise ValueError(f"unsupported noise operation: {operation}")


def generate_noisy_tokens(
    tokens: Sequence[str],
    *,
    seed: int,
    sentence_error_probability: float = 0.6,
    max_errors: int = 1,
    operations: Sequence[str] = (
        "delete",
        "insert",
        "substitute",
        "transpose",
        "diacritic",
    ),
) -> NoisyTokens:
    """Corrupt eligible word tokens and retain full error metadata."""

    if not 0.0 <= sentence_error_probability <= 1.0:
        raise ValueError("sentence_error_probability must be between 0 and 1")
    if max_errors < 0:
        raise ValueError("max_errors must be non-negative")
    if not operations:
        raise ValueError("at least one noise operation is required")

    noisy_tokens = list(tokens)
    rng = random.Random(seed)
    eligible = [
        index
        for index, token in enumerate(tokens)
        if len(token) >= 2 and all(character.isalpha() for character in token)
    ]
    if not eligible or max_errors == 0 or rng.random() >= sentence_error_probability:
        return NoisyTokens(noisy_tokens, [])

    error_count = min(max_errors, len(eligible))
    selected_indices = sorted(rng.sample(eligible, error_count))
    events: list[NoiseEvent] = []
    for token_index in selected_indices:
        clean_token = tokens[token_index]
        operation = rng.choice(list(operations))
        noisy_token = _apply_operation(clean_token, operation, rng)
        noisy_tokens[token_index] = noisy_token
        events.append(
            NoiseEvent(token_index, clean_token, noisy_token, operation)
        )

    return NoisyTokens(noisy_tokens, events)

