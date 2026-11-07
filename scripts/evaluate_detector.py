"""Evaluate a locally trained BiLSTM detector checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import DetectorDataset, LABEL_PAD_ID, make_detector_collate_fn
from src.data.io import read_jsonl, write_json
from src.models.bilstm_detector import BiLSTMDetector, BiLSTMDetectorConfig
from src.training.detector import evaluate_detector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/detector_evaluation.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not checkpoint.get("trained_from_scratch"):
        raise ValueError("checkpoint does not declare train-from-scratch provenance")

    model = BiLSTMDetector(BiLSTMDetectorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    vocabulary = checkpoint["vocabulary"]
    dataset = DetectorDataset(read_jsonl(args.data), vocabulary)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_detector_collate_fn(int(vocabulary["<PAD>"])),
    )
    class_weights = checkpoint["class_weights"].to(device)
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=LABEL_PAD_ID,
    )
    model.to(device)
    threshold = float(checkpoint["training_config"]["error_threshold"])
    metrics = evaluate_detector(
        model,
        loader,
        loss_function,
        device=device,
        error_threshold=threshold,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "threshold": threshold,
        "metrics": metrics,
    }
    write_json(args.output, report)

    print(f"ERROR precision: {metrics['precision']:.4f}")
    print(f"ERROR recall: {metrics['recall']:.4f}")
    print(f"ERROR F1: {metrics['f1']:.4f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

