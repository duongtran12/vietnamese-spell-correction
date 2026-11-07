from src.correction.candidates import Candidate
from src.correction.ranker import CandidateRanker
from src.models.trigram_lm import TrigramLanguageModel


def test_ranker_uses_context_to_break_equal_candidate_scores() -> None:
    language_model = TrigramLanguageModel().fit(
        [["tôi", "đi", "học"], ["tôi", "đi", "học"], ["cuộc", "họp", "bắt", "đầu"]]
    )
    ranker = CandidateRanker(language_model)
    candidates = [
        Candidate("họp", 1, 2, -1 / 3, 1.0),
        Candidate("học", 1, 2, -1 / 3, 1.0),
    ]

    ranked = ranker.rank(["tôi", "đi", "họ"], 2, candidates)

    assert ranked[0].token == "học"
    assert ranked[0].context_log_probability > ranked[1].context_log_probability


def test_ranker_returns_explainable_score_components() -> None:
    language_model = TrigramLanguageModel().fit([["tôi", "học"]])
    ranker = CandidateRanker(language_model)
    candidate = Candidate("học", 1, 3, -0.3, 1.4)

    result = ranker.rank(["tôi", "họ"], 1, [candidate])[0].to_dict()

    assert set(result) == {
        "token",
        "distance",
        "frequency",
        "edit_score",
        "frequency_score",
        "context_log_probability",
        "total_score",
    }

