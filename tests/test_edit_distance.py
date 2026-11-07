import pytest

from src.correction.edit_distance import (
    damerau_levenshtein_distance,
    levenshtein_distance,
)


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("", "học", 3),
        ("học", "học", 0),
        ("hoc", "học", 1),
        ("tiêng", "tiếng", 1),
        ("ngôn", "ngônn", 1),
    ],
)
def test_levenshtein_distance_for_vietnamese_tokens(
    source: str, target: str, expected: int
) -> None:
    assert levenshtein_distance(source, target) == expected
    assert levenshtein_distance(target, source) == expected


def test_damerau_distance_counts_adjacent_transposition_once() -> None:
    assert levenshtein_distance("hco", "hoc") == 2
    assert damerau_levenshtein_distance("hco", "hoc") == 1


def test_edit_distance_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        levenshtein_distance("học", None)  # type: ignore[arg-type]

