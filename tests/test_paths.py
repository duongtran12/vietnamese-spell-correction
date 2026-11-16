from pathlib import Path

from src.utils.paths import portable_project_path


def test_portable_project_path_keeps_data_suffix() -> None:
    path = Path("D:/some/machine/project/data/processed/run/validation.jsonl")

    result = portable_project_path(path)

    assert result == str(Path("data/processed/run/validation.jsonl"))


def test_portable_project_path_falls_back_to_filename() -> None:
    result = portable_project_path(Path("D:/external/location/model.bin"))

    assert result == "model.bin"
