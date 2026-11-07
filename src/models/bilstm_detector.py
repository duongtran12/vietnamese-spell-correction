"""BiLSTM token error detector with randomly initialized embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class BiLSTMDetectorConfig:
    vocab_size: int
    padding_idx: int = 0
    embedding_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    num_classes: int = 2

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class BiLSTMDetector(nn.Module):
    """Classify every input token as CORRECT or ERROR."""

    def __init__(self, config: BiLSTMDetectorConfig) -> None:
        super().__init__()
        if config.vocab_size < 2:
            raise ValueError("vocab_size must include at least PAD and UNK")
        if config.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.padding_idx,
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.bilstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.input_dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.bilstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=token_ids.shape[1],
        )
        return self.classifier(self.output_dropout(output))

