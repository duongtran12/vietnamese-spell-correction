import torch

from src.models.bilstm_detector import BiLSTMDetector, BiLSTMDetectorConfig


def test_bilstm_detector_returns_token_logits() -> None:
    torch.manual_seed(1)
    model = BiLSTMDetector(
        BiLSTMDetectorConfig(
            vocab_size=8,
            embedding_dim=6,
            hidden_size=5,
            dropout=0.0,
        )
    )
    token_ids = torch.tensor([[2, 3, 4], [5, 6, 0]])
    lengths = torch.tensor([3, 2])

    logits = model(token_ids, lengths)

    assert logits.shape == (2, 3, 2)
    assert model.embedding.weight.requires_grad

