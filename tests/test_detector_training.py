from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DetectorDataset, make_detector_collate_fn
from src.models.bilstm_detector import BiLSTMDetector, BiLSTMDetectorConfig
from src.training.detector import (
    DetectorTrainingConfig,
    compute_class_weights,
    error_class_metrics,
    fit_detector,
)


def test_error_class_metrics_use_error_as_positive_class() -> None:
    metrics = error_class_metrics(
        torch.tensor([1, 1, 0, 0]),
        torch.tensor([1, 0, 1, 0]),
    )

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_detector_training_smoke_saves_train_from_scratch_checkpoint(
    tmp_path: Path,
) -> None:
    torch.manual_seed(2)
    vocabulary = {"<PAD>": 0, "<UNK>": 1, "tôi": 2, "học": 3, "họ": 4, ".": 5}
    records = [
        {"noisy_tokens": ["tôi", "học", "."], "labels": [0, 0, 0]},
        {"noisy_tokens": ["tôi", "họ", "."], "labels": [0, 1, 0]},
        {"noisy_tokens": ["họ", "học", "."], "labels": [1, 0, 0]},
        {"noisy_tokens": ["tôi", "họ"], "labels": [0, 1]},
    ]
    dataset = DetectorDataset(records, vocabulary)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=make_detector_collate_fn(0),
    )
    model = BiLSTMDetector(
        BiLSTMDetectorConfig(
            vocab_size=len(vocabulary),
            embedding_dim=4,
            hidden_size=4,
            dropout=0.0,
        )
    )
    checkpoint_path = tmp_path / "detector.pt"

    result = fit_detector(
        model,
        loader,
        loader,
        class_weights=compute_class_weights(dataset),
        vocabulary=vocabulary,
        training_config=DetectorTrainingConfig(
            epochs=2,
            learning_rate=0.01,
            early_stopping_patience=2,
        ),
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
        provenance={"fit_split": "train", "seed": 2},
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert result["epochs_completed"] == 2
    assert checkpoint["trained_from_scratch"] is True
    assert checkpoint["provenance"]["fit_split"] == "train"
    assert checkpoint["model_config"]["vocab_size"] == len(vocabulary)

