import torch

from src.data.dataset import DetectorDataset, LABEL_PAD_ID, collate_detector_batch


def test_detector_dataset_and_collate_preserve_alignment() -> None:
    vocabulary = {"<PAD>": 0, "<UNK>": 1, "tôi": 2, "học": 3}
    dataset = DetectorDataset(
        [
            {"noisy_tokens": ["tôi", "học"], "labels": [0, 1]},
            {"noisy_tokens": ["mới"], "labels": [0]},
        ],
        vocabulary,
    )

    batch = collate_detector_batch([dataset[0], dataset[1]], pad_id=0)

    assert batch["token_ids"].tolist() == [[2, 3], [1, 0]]
    assert batch["labels"].tolist() == [[0, 1], [0, LABEL_PAD_ID]]
    assert batch["lengths"].tolist() == [2, 1]
    assert torch.equal(
        batch["attention_mask"], torch.tensor([[True, True], [True, False]])
    )
    assert dataset.label_counts() == (2, 1)

