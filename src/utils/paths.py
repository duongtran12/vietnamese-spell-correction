"""Stable project-relative path labels for portable experiment reports."""

from __future__ import annotations

from pathlib import Path


def portable_project_path(path: str | Path) -> str:
    """Remove machine-specific prefixes while preserving project provenance."""

    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        pass

    parts = candidate.parts
    lowered = [part.casefold() for part in parts]
    for anchor in ("data", "artifacts", "reports", "configs"):
        if anchor in lowered:
            index = len(lowered) - 1 - lowered[::-1].index(anchor)
            return str(Path(*parts[index:]))
    return candidate.name
