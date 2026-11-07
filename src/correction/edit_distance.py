"""Character edit-distance algorithms implemented from scratch."""

from __future__ import annotations


def _validate_tokens(source: str, target: str) -> None:
    if not isinstance(source, str) or not isinstance(target, str):
        raise TypeError("source and target must be strings")


def levenshtein_distance(source: str, target: str) -> int:
    """Return insertion, deletion, and substitution distance."""

    _validate_tokens(source, target)
    if len(source) < len(target):
        source, target = target, source
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for source_index, source_character in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_character in enumerate(target, start=1):
            insertion = current[target_index - 1] + 1
            deletion = previous[target_index] + 1
            substitution = previous[target_index - 1] + (
                source_character != target_character
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def damerau_levenshtein_distance(source: str, target: str) -> int:
    """Return unrestricted Damerau–Levenshtein distance.

    In addition to insertion, deletion, and substitution, this metric counts
    transposition of adjacent characters as one edit.
    """

    _validate_tokens(source, target)
    source_length = len(source)
    target_length = len(target)
    if not source:
        return target_length
    if not target:
        return source_length

    maximum_distance = source_length + target_length
    matrix = [
        [0] * (target_length + 2) for _ in range(source_length + 2)
    ]
    matrix[0][0] = maximum_distance
    for source_index in range(source_length + 1):
        matrix[source_index + 1][0] = maximum_distance
        matrix[source_index + 1][1] = source_index
    for target_index in range(target_length + 1):
        matrix[0][target_index + 1] = maximum_distance
        matrix[1][target_index + 1] = target_index

    last_row: dict[str, int] = {}
    for source_index in range(1, source_length + 1):
        last_matching_column = 0
        for target_index in range(1, target_length + 1):
            matching_row = last_row.get(target[target_index - 1], 0)
            matching_column = last_matching_column
            substitution_cost = 1
            if source[source_index - 1] == target[target_index - 1]:
                substitution_cost = 0
                last_matching_column = target_index

            matrix[source_index + 1][target_index + 1] = min(
                matrix[source_index][target_index] + substitution_cost,
                matrix[source_index + 1][target_index] + 1,
                matrix[source_index][target_index + 1] + 1,
                matrix[matching_row][matching_column]
                + (source_index - matching_row - 1)
                + 1
                + (target_index - matching_column - 1),
            )
        last_row[source[source_index - 1]] = source_index

    return matrix[source_length + 1][target_length + 1]

