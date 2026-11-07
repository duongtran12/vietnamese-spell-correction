"""UTF-8 JSON and JSONL input/output helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping

from src.data.schema import CleanSentence


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON objects from a UTF-8 JSONL file."""

    input_path = Path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {input_path}:{line_number}: {error}"
                ) from error
            if not isinstance(data, dict):
                raise ValueError(
                    f"invalid record at {input_path}:{line_number}: "
                    "record must be a JSON object"
                )
            records.append(data)
    return records


def read_clean_jsonl(path: str | Path) -> list[CleanSentence]:
    """Read and validate clean sentence records from a JSONL file."""

    records: list[CleanSentence] = []
    for index, data in enumerate(read_jsonl(path), start=1):
        try:
            records.append(CleanSentence.from_mapping(data))
        except ValueError as error:
            raise ValueError(f"invalid record at {path}:{index}: {error}") from error
    return records



def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write JSON objects atomically as UTF-8 JSONL."""

    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    _atomic_write(Path(path), content)


def write_json(path: str | Path, data: Any) -> None:
    """Write a JSON value atomically as formatted UTF-8."""

    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    _atomic_write(Path(path), content)

