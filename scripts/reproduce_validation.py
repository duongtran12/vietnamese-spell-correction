"""Reproduce frozen validation reports without reading the test split."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-name", default="viwiki-20261101-smoke-dedup"
    )
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--include-tuning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_commands(
    *,
    data_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
    experiment_name: str,
    device: str,
    include_tuning: bool,
) -> list[list[str]]:
    """Build the deterministic validation-only command sequence."""

    python = sys.executable
    train = data_dir / "train.jsonl"
    validation = data_dir / "validation.jsonl"
    vocabulary = data_dir / "vocabulary.json"
    detector = artifact_dir / f"{experiment_name}-detector.pt"
    trigram = artifact_dir / f"{experiment_name}-trigram.json"
    figures = report_dir / "figures"
    commands: list[list[str]] = [
        [
            python,
            "-m",
            "scripts.evaluate_detector",
            "--checkpoint",
            str(detector),
            "--data",
            str(validation),
            "--output",
            str(report_dir / f"{experiment_name}-detector-validation.json"),
            "--batch-size",
            "32",
            "--device",
            device,
        ],
        [
            python,
            "-m",
            "scripts.evaluate_candidates",
            "--train",
            str(train),
            "--evaluation",
            str(validation),
            "--config",
            "configs/ranking.yaml",
            "--output",
            str(report_dir / f"{experiment_name}-candidates.json"),
        ],
        [
            python,
            "-m",
            "scripts.evaluate_ranker",
            "--train",
            str(train),
            "--evaluation",
            str(validation),
            "--config",
            "configs/ranking.yaml",
            "--output",
            str(report_dir / f"{experiment_name}-ranker.json"),
        ],
    ]
    if include_tuning:
        commands.append(
            [
                python,
                "-m",
                "scripts.tune_inference",
                "--detector",
                str(detector),
                "--vocabulary",
                str(vocabulary),
                "--trigram",
                str(trigram),
                "--data",
                str(validation),
                "--thresholds",
                "0.5,0.6,0.7,0.8,0.9",
                "--rank-margins",
                "0.25,0.5,0.75,1.0",
                "--max-false-correction-rate",
                "0.02",
                "--output",
                str(report_dir / f"{experiment_name}-inference-tuning.json"),
                "--device",
                device,
            ]
        )
    commands.extend(
        [
            [
                python,
                "-m",
                "scripts.evaluate_end_to_end",
                "--detector",
                str(detector),
                "--vocabulary",
                str(vocabulary),
                "--trigram",
                str(trigram),
                "--data",
                str(validation),
                "--output",
                str(report_dir / f"{experiment_name}-end-to-end-validation.json"),
                "--device",
                device,
            ],
            [
                python,
                "-m",
                "scripts.run_ablation",
                "--train",
                str(train),
                "--evaluation",
                str(validation),
                "--output",
                str(report_dir / f"{experiment_name}-ablation.json"),
                "--figure",
                str(figures / f"{experiment_name}-ranker-ablation.png"),
                "--error-figure",
                str(figures / f"{experiment_name}-error-types.png"),
            ],
            [
                python,
                "-m",
                "scripts.analyze_errors",
                "--detector",
                str(detector),
                "--vocabulary",
                str(vocabulary),
                "--trigram",
                str(trigram),
                "--data",
                str(validation),
                "--output",
                str(report_dir / f"{experiment_name}-error-analysis.json"),
                "--markdown",
                str(report_dir / "VIWIKI_SMOKE_DEDUP_ERROR_ANALYSIS.md"),
                "--confusion-figure",
                str(figures / f"{experiment_name}-detector-confusion.png"),
                "--outcome-figure",
                str(figures / f"{experiment_name}-correction-outcomes.png"),
                "--device",
                device,
            ],
        ]
    )
    return commands


def _validate_inputs(
    data_dir: Path,
    artifact_dir: Path,
    experiment_name: str,
) -> None:
    required = [
        data_dir / "train.jsonl",
        data_dir / "validation.jsonl",
        data_dir / "vocabulary.json",
        artifact_dir / f"{experiment_name}-detector.pt",
        artifact_dir / f"{experiment_name}-trigram.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing validation inputs:\n{formatted}")


def main() -> None:
    args = _parse_args()
    _validate_inputs(args.data_dir, args.artifact_dir, args.experiment_name)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        experiment_name=args.experiment_name,
        device=args.device,
        include_tuning=args.include_tuning,
    )
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {subprocess.list2cmdline(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    print("Validation reproduction completed. The test split was not read.")


if __name__ == "__main__":
    main()
