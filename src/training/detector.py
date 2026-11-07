"""Training, evaluation, metrics, and checkpointing for the BiLSTM detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import DetectorDataset, LABEL_PAD_ID
from src.models.bilstm_detector import BiLSTMDetector


@dataclass(frozen=True)
class DetectorTrainingConfig:
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    early_stopping_patience: int = 4
    error_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if not 0.0 <= self.error_threshold <= 1.0:
            raise ValueError("error_threshold must be between 0 and 1")


def error_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, int | float]:
    """Compute token metrics with ERROR=1 as the positive class."""

    predictions = predictions.reshape(-1).to(torch.long)
    targets = targets.reshape(-1).to(torch.long)
    if predictions.numel() != targets.numel():
        raise ValueError("predictions and targets must have equal size")

    true_positive = int(((predictions == 1) & (targets == 1)).sum().item())
    false_positive = int(((predictions == 1) & (targets == 0)).sum().item())
    false_negative = int(((predictions == 0) & (targets == 1)).sum().item())
    true_negative = int(((predictions == 0) & (targets == 0)).sum().item())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_class_weights(dataset: DetectorDataset) -> torch.Tensor:
    """Return inverse-frequency weights without dividing by an absent class."""

    correct, error = dataset.label_counts()
    total = correct + error
    if total == 0 or correct == 0 or error == 0:
        return torch.ones(2, dtype=torch.float32)
    return torch.tensor(
        [total / (2 * correct), total / (2 * error)], dtype=torch.float32
    )


def _loss_for_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_function: nn.Module,
) -> torch.Tensor:
    return loss_function(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def train_one_epoch(
    model: BiLSTMDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    *,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]

        optimizer.zero_grad(set_to_none=True)
        logits = model(token_ids, lengths)
        loss = _loss_for_batch(logits, labels, loss_function)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / batches if batches else 0.0


@torch.no_grad()
def evaluate_detector(
    model: BiLSTMDetector,
    loader: DataLoader,
    loss_function: nn.Module,
    *,
    device: torch.device,
    error_threshold: float = 0.5,
) -> dict[str, int | float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"]
        logits = model(token_ids, lengths)
        total_loss += float(_loss_for_batch(logits, labels, loss_function).item())
        batches += 1

        valid = labels != LABEL_PAD_ID
        error_probabilities = torch.softmax(logits, dim=-1)[..., 1]
        predictions.append((error_probabilities[valid] >= error_threshold).cpu())
        targets.append(labels[valid].cpu())

    flat_predictions = torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.bool)
    flat_targets = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    metrics = error_class_metrics(flat_predictions, flat_targets)
    metrics["loss"] = total_loss / batches if batches else 0.0
    metrics["tokens"] = int(flat_targets.numel())
    return metrics


def _save_checkpoint_atomic(path: Path, checkpoint: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
            temporary_path = Path(handle.name)
        torch.save(dict(checkpoint), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def fit_detector(
    model: BiLSTMDetector,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    class_weights: torch.Tensor,
    vocabulary: Mapping[str, int],
    training_config: DetectorTrainingConfig,
    checkpoint_path: str | Path,
    device: torch.device,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train from scratch and checkpoint the best validation ERROR-F1."""

    model.to(device)
    class_weights = class_weights.to(device)
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=LABEL_PAD_ID,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, training_config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device=device,
            gradient_clip=training_config.gradient_clip,
        )
        validation_metrics = evaluate_detector(
            model,
            validation_loader,
            loss_function,
            device=device,
            error_threshold=training_config.error_threshold,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation_metrics,
        }
        history.append(epoch_record)

        validation_f1 = float(validation_metrics["f1"])
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint_atomic(
                Path(checkpoint_path),
                {
                    "artifact_type": "bilstm_token_error_detector",
                    "trained_from_scratch": True,
                    "model_config": model.config.to_dict(),
                    "training_config": asdict(training_config),
                    "vocabulary": dict(vocabulary),
                    "class_weights": class_weights.detach().cpu(),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "provenance": dict(provenance or {}),
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                break

    return {
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "epochs_completed": len(history),
        "history": history,
    }

