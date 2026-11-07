import torch
from torch import nn

from src.correction.candidates import CandidateGenerator
from src.correction.ranker import CandidateRanker
from src.inference import SpellingCorrectionPipeline
from src.models.trigram_lm import TrigramLanguageModel


class TokenIdDetector(nn.Module):
    def __init__(self, error_token_id: int) -> None:
        super().__init__()
        self.error_token_id = error_token_id

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        is_error = token_ids == self.error_token_id
        logits = torch.zeros((*token_ids.shape, 2), device=token_ids.device)
        logits[..., 0] = torch.where(is_error, -5.0, 5.0)
        logits[..., 1] = torch.where(is_error, 5.0, -5.0)
        return logits


def _pipeline(minimum_rank_margin: float = 0.0) -> SpellingCorrectionPipeline:
    vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1,
        "tôi": 2,
        "đi": 3,
        "họ": 4,
        "học": 5,
        ".": 6,
    }
    language_model = TrigramLanguageModel().fit(
        [["tôi", "đi", "học", "."], ["tôi", "đi", "học", "."], ["cuộc", "họp", "."]]
    )
    return SpellingCorrectionPipeline(
        TokenIdDetector(error_token_id=4),  # type: ignore[arg-type]
        vocabulary,
        CandidateGenerator({"học": 10, "họp": 1}),
        CandidateRanker(language_model),
        detector_threshold=0.5,
        minimum_rank_margin=minimum_rank_margin,
        keep_original_bonus=0.0,
    )


def test_pipeline_corrects_text_and_preserves_original_spacing() -> None:
    result = _pipeline().correct_text("tôi  đi họ.")

    assert result.corrected_text == "tôi  đi học."
    assert result.corrected_tokens == ("tôi", "đi", "học", ".")
    assert len(result.decisions) == 1
    assert result.decisions[0].original_token == "họ"
    assert result.decisions[0].suggestions[0].token == "học"
    assert result.decisions[0].changed is True


def test_pipeline_keeps_token_when_rank_margin_is_too_small() -> None:
    result = _pipeline(minimum_rank_margin=1_000.0).correct_text("tôi đi họ.")

    assert result.corrected_text == "tôi đi họ."
    assert result.decisions[0].changed is False
    assert result.decisions[0].suggestions


def test_pipeline_can_keep_a_correct_oov_token() -> None:
    vocabulary = {"<PAD>": 0, "<UNK>": 1, "tôi": 2, "đi": 3}
    language_model = TrigramLanguageModel().fit([["ta", "đi"], ["tại", "đây"]])
    pipeline = SpellingCorrectionPipeline(
        TokenIdDetector(error_token_id=2),  # type: ignore[arg-type]
        vocabulary,
        CandidateGenerator({"ta": 10, "tại": 8}),
        CandidateRanker(language_model),
        detector_threshold=0.5,
        minimum_rank_margin=0.5,
        keep_original_bonus=2.0,
    )

    result = pipeline.correct_text("tôi đi")

    assert result.corrected_text == "tôi đi"
    assert result.decisions[0].suggestions[0].token == "tôi"
    assert result.decisions[0].changed is False


def test_pipeline_restores_title_case_after_ranking_lowercase_candidate() -> None:
    vocabulary = {"<PAD>": 0, "<UNK>": 1, "Hộm": 2, "nay": 3}
    language_model = TrigramLanguageModel().fit(
        [["hôm", "nay"], ["hôm", "nay"], ["hôm", "qua"]]
    )
    pipeline = SpellingCorrectionPipeline(
        TokenIdDetector(error_token_id=2),  # type: ignore[arg-type]
        vocabulary,
        CandidateGenerator({"hôm": 10, "họp": 1}),
        CandidateRanker(language_model),
        detector_threshold=0.5,
        minimum_rank_margin=0.0,
        keep_original_bonus=0.75,
    )

    result = pipeline.correct_text("Hộm nay")

    assert result.corrected_text == "Hôm nay"
    assert result.decisions[0].suggestions[0].token == "Hôm"

