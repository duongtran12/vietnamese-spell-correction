"""PyTorch dataset and padded batches for token error detection."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


LABEL_PAD_ID = -100


class DetectorDataset(Dataset[dict[str, torch.Tensor]]):
    """Convert processed noisy tokens and labels to vocabulary IDs."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        token_to_id: Mapping[str, int],
    ) -> None:
        if "<UNK>" not in token_to_id or "<PAD>" not in token_to_id:
            raise ValueError("vocabulary must contain <PAD> and <UNK>")

        self.token_to_id = dict(token_to_id)
        self.examples: list[tuple[list[int], list[int]]] = []
        unknown_id = self.token_to_id["<UNK>"]
        for record in records:
            tokens = record.get("noisy_tokens")
            labels = record.get("labels")
            if not isinstance(tokens, list) or not isinstance(labels, list):
                raise ValueError("each record must contain noisy_tokens and labels lists")
            if not tokens or len(tokens) != len(labels):
                raise ValueError("tokens and labels must be non-empty and aligned")
            if any(label not in (0, 1) for label in labels):
                raise ValueError("detector labels must be 0 or 1")

            token_ids = [self.token_to_id.get(str(token), unknown_id) for token in tokens]
            self.examples.append((token_ids, [int(label) for label in labels]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        token_ids, labels = self.examples[index]
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def label_counts(self) -> tuple[int, int]:
        correct = sum(label == 0 for _, labels in self.examples for label in labels)
        error = sum(label == 1 for _, labels in self.examples for label in labels)
        return correct, error


def collate_detector_batch(
    examples: Sequence[dict[str, torch.Tensor]],
    *,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch and expose lengths plus a boolean attention mask."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    token_ids = [example["token_ids"] for example in examples]
    labels = [example["labels"] for example in examples]
    lengths = torch.tensor([len(sequence) for sequence in token_ids], dtype=torch.long)
    padded_tokens = pad_sequence(token_ids, batch_first=True, padding_value=pad_id)
    padded_labels = pad_sequence(
        labels, batch_first=True, padding_value=LABEL_PAD_ID
    )
    positions = torch.arange(padded_tokens.shape[1]).unsqueeze(0)
    attention_mask = positions < lengths.unsqueeze(1)
    return {
        "token_ids": padded_tokens,
        "labels": padded_labels,
        "lengths": lengths,
        "attention_mask": attention_mask,
    }


def make_detector_collate_fn(pad_id: int):
    """Return a picklable collate function bound to one vocabulary."""

    return partial(collate_detector_batch, pad_id=pad_id)

