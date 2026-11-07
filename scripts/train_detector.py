"""Train the BiLSTM token error detector from random initialization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import DetectorDataset, make_detector_collate_fn
from src.data.io import read_json, read_jsonl, write_json
from src.models.bilstm_detector import BiLSTMDetector, BiLSTMDetectorConfig
from src.training.detector import (
    DetectorTrainingConfig,
    compute_class_weights,
    fit_detector,
)
from src.utils.seed import set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/detector.yaml")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/detector.pt")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/detector_training.json")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("detector config must contain a YAML object")
    return config


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    model_config = config.get("model", {})
    training_values = config.get("training", {})
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    vocabulary_artifact = read_json(args.vocabulary)
    token_to_id = vocabulary_artifact["token_to_id"]
    train_dataset = DetectorDataset(read_jsonl(args.train), token_to_id)
    validation_dataset = DetectorDataset(read_jsonl(args.validation), token_to_id)
    if not train_dataset or not validation_dataset:
        raise ValueError("train and validation datasets must be non-empty")

    pad_id = int(token_to_id["<PAD>"])
    collate_fn = make_detector_collate_fn(pad_id)
    batch_size = int(training_values.get("batch_size", 32))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = BiLSTMDetector(
        BiLSTMDetectorConfig(
            vocab_size=len(token_to_id),
            padding_idx=pad_id,
            embedding_dim=int(model_config.get("embedding_dim", 128)),
            hidden_size=int(model_config.get("hidden_size", 128)),
            num_layers=int(model_config.get("num_layers", 1)),
            dropout=float(model_config.get("dropout", 0.3)),
        )
    )
    training_config = DetectorTrainingConfig(
        epochs=int(training_values.get("epochs", 20)),
        learning_rate=float(training_values.get("learning_rate", 0.001)),
        weight_decay=float(training_values.get("weight_decay", 0.0)),
        gradient_clip=float(training_values.get("gradient_clip", 1.0)),
        early_stopping_patience=int(
            training_values.get("early_stopping_patience", 4)
        ),
        error_threshold=float(training_values.get("error_threshold", 0.5)),
    )
    device = _device(args.device)
    result = fit_detector(
        model,
        train_loader,
        validation_loader,
        class_weights=compute_class_weights(train_dataset),
        vocabulary=token_to_id,
        training_config=training_config,
        checkpoint_path=args.checkpoint,
        device=device,
        provenance={
            "fit_split": "train",
            "train_path": str(args.train),
            "validation_path": str(args.validation),
            "vocabulary_path": str(args.vocabulary),
            "seed": seed,
            "device": str(device),
        },
    )
    write_json(args.report, result)

    print(f"Device: {device}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best validation ERROR-F1: {result['best_validation_f1']:.4f}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()

