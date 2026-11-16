from pathlib import Path

from scripts.reproduce_validation import build_commands


def test_reproduction_commands_never_reference_test_split() -> None:
    commands = build_commands(
        data_dir=Path("data/processed/experiment"),
        artifact_dir=Path("artifacts"),
        report_dir=Path("reports"),
        experiment_name="experiment",
        device="cpu",
        include_tuning=True,
    )

    flattened = " ".join(argument for command in commands for argument in command)

    assert "test.jsonl" not in flattened
    assert "validation.jsonl" in flattened
    assert "scripts.tune_inference" in flattened


def test_reproduction_can_skip_tuning() -> None:
    commands = build_commands(
        data_dir=Path("data"),
        artifact_dir=Path("artifacts"),
        report_dir=Path("reports"),
        experiment_name="experiment",
        device="cpu",
        include_tuning=False,
    )

    modules = [command[2] for command in commands]

    assert "scripts.tune_inference" not in modules
    assert "scripts.evaluate_end_to_end" in modules
