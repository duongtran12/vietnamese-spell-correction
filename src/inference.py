"""Offline end-to-end Vietnamese spelling correction pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.correction.candidates import CandidateGenerator
from src.correction.ranker import CandidateRanker, RankedCandidate
from src.data.normalize import TokenSpan, normalize_text, tokenize_with_spans
from src.models.bilstm_detector import BiLSTMDetector, BiLSTMDetectorConfig
from src.models.trigram_lm import TrigramLanguageModel


@dataclass(frozen=True)
class TokenDecision:
    token_index: int
    original_token: str
    detector_confidence: float
    suggestions: tuple[RankedCandidate, ...]
    applied_token: str
    rank_margin: float | None
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["suggestions"] = [suggestion.to_dict() for suggestion in self.suggestions]
        return data


@dataclass(frozen=True)
class CorrectionResult:
    original_tokens: tuple[str, ...]
    corrected_tokens: tuple[str, ...]
    decisions: tuple[TokenDecision, ...]
    original_text: str | None = None
    corrected_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "corrected_text": self.corrected_text,
            "original_tokens": list(self.original_tokens),
            "corrected_tokens": list(self.corrected_tokens),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


class SpellingCorrectionPipeline:
    """Run detector, candidate generation, and context ranking offline."""

    def __init__(
        self,
        detector: BiLSTMDetector,
        detector_vocabulary: Mapping[str, int],
        candidate_generator: CandidateGenerator,
        ranker: CandidateRanker,
        *,
        detector_threshold: float = 0.5,
        minimum_rank_margin: float = 0.0,
        candidate_pool_size: int = 5,
        top_k_suggestions: int = 3,
        device: torch.device | None = None,
    ) -> None:
        if not 0.0 <= detector_threshold <= 1.0:
            raise ValueError("detector_threshold must be between 0 and 1")
        if minimum_rank_margin < 0:
            raise ValueError("minimum_rank_margin must be non-negative")
        if candidate_pool_size < 1 or top_k_suggestions < 1:
            raise ValueError("candidate and suggestion sizes must be positive")
        if top_k_suggestions > candidate_pool_size:
            raise ValueError("top_k_suggestions cannot exceed candidate_pool_size")
        if "<UNK>" not in detector_vocabulary:
            raise ValueError("detector vocabulary must contain <UNK>")

        self.detector = detector
        self.detector_vocabulary = dict(detector_vocabulary)
        self.candidate_generator = candidate_generator
        self.ranker = ranker
        self.detector_threshold = detector_threshold
        self.minimum_rank_margin = minimum_rank_margin
        self.candidate_pool_size = candidate_pool_size
        self.top_k_suggestions = top_k_suggestions
        self.device = device or torch.device("cpu")
        self.detector.to(self.device)
        self.detector.eval()

    @torch.no_grad()
    def detector_probabilities(self, tokens: Sequence[str]) -> list[float]:
        if not tokens:
            return []
        unknown_id = self.detector_vocabulary["<UNK>"]
        token_ids = torch.tensor(
            [[self.detector_vocabulary.get(token, unknown_id) for token in tokens]],
            dtype=torch.long,
            device=self.device,
        )
        lengths = torch.tensor([len(tokens)], dtype=torch.long)
        logits = self.detector(token_ids, lengths)
        return torch.softmax(logits, dim=-1)[0, :, 1].cpu().tolist()

    def correct_tokens(self, tokens: Sequence[str]) -> CorrectionResult:
        original_tokens = tuple(tokens)
        corrected_tokens = list(tokens)
        probabilities = self.detector_probabilities(original_tokens)
        decisions: list[TokenDecision] = []

        for token_index, confidence in enumerate(probabilities):
            if confidence < self.detector_threshold:
                continue
            original_token = original_tokens[token_index]
            candidates = self.candidate_generator.generate(
                original_token,
                top_k=self.candidate_pool_size,
            )
            suggestions = self.ranker.rank(
                corrected_tokens,
                token_index,
                candidates,
                top_k=self.top_k_suggestions,
            )

            rank_margin: float | None = None
            applied_token = original_token
            if suggestions:
                if len(suggestions) > 1:
                    rank_margin = suggestions[0].total_score - suggestions[1].total_score
                has_sufficient_margin = (
                    rank_margin is None or rank_margin >= self.minimum_rank_margin
                )
                if has_sufficient_margin:
                    applied_token = suggestions[0].token
                    corrected_tokens[token_index] = applied_token

            decisions.append(
                TokenDecision(
                    token_index=token_index,
                    original_token=original_token,
                    detector_confidence=float(confidence),
                    suggestions=tuple(suggestions),
                    applied_token=applied_token,
                    rank_margin=rank_margin,
                    changed=applied_token != original_token,
                )
            )

        return CorrectionResult(
            original_tokens=original_tokens,
            corrected_tokens=tuple(corrected_tokens),
            decisions=tuple(decisions),
        )

    def correct_text(self, text: str) -> CorrectionResult:
        normalized = normalize_text(text)
        spans = tokenize_with_spans(normalized)
        result = self.correct_tokens([span.text for span in spans])
        corrected_text = _replace_token_spans(
            normalized,
            spans,
            result.corrected_tokens,
        )
        return replace(
            result,
            original_text=normalized,
            corrected_text=corrected_text,
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        detector_checkpoint: str | Path,
        vocabulary_artifact: str | Path,
        trigram_artifact: str | Path,
        detector_threshold: float = 0.5,
        minimum_rank_margin: float = 0.0,
        candidate_pool_size: int = 5,
        top_k_suggestions: int = 3,
        context_weight: float = 1.0,
        edit_weight: float = 1.0,
        frequency_weight: float = 0.1,
        candidate_distance: str = "damerau_levenshtein",
        short_token_max_length: int = 4,
        short_token_max_distance: int = 1,
        long_token_max_distance: int = 2,
        device: torch.device | None = None,
    ) -> "SpellingCorrectionPipeline":
        checkpoint = torch.load(
            detector_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not checkpoint.get("trained_from_scratch"):
            raise ValueError("detector checkpoint lacks train-from-scratch provenance")
        detector = BiLSTMDetector(
            BiLSTMDetectorConfig(**checkpoint["model_config"])
        )
        detector.load_state_dict(checkpoint["model_state_dict"])

        with Path(vocabulary_artifact).open("r", encoding="utf-8") as handle:
            vocabulary_data = json.load(handle)
        if checkpoint["vocabulary"] != vocabulary_data["token_to_id"]:
            raise ValueError(
                "detector checkpoint and vocabulary artifact do not match"
            )
        frequencies = {
            token: int(frequency)
            for token, frequency in vocabulary_data["frequencies"].items()
            if token.isalpha()
        }
        candidate_generator = CandidateGenerator(
            frequencies,
            distance=candidate_distance,
            short_token_max_length=short_token_max_length,
            short_token_max_distance=short_token_max_distance,
            long_token_max_distance=long_token_max_distance,
        )

        with Path(trigram_artifact).open("r", encoding="utf-8") as handle:
            trigram_data = json.load(handle)
        language_model = TrigramLanguageModel.from_dict(trigram_data)
        ranker = CandidateRanker(
            language_model,
            context_weight=context_weight,
            edit_weight=edit_weight,
            frequency_weight=frequency_weight,
        )
        return cls(
            detector,
            checkpoint["vocabulary"],
            candidate_generator,
            ranker,
            detector_threshold=detector_threshold,
            minimum_rank_margin=minimum_rank_margin,
            candidate_pool_size=candidate_pool_size,
            top_k_suggestions=top_k_suggestions,
            device=device,
        )


def _replace_token_spans(
    text: str,
    spans: Sequence[TokenSpan],
    replacements: Sequence[str],
) -> str:
    if len(spans) != len(replacements):
        raise ValueError("token spans and replacements must be aligned")
    pieces: list[str] = []
    cursor = 0
    for span, replacement in zip(spans, replacements, strict=True):
        pieces.append(text[cursor : span.start])
        pieces.append(replacement)
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)

