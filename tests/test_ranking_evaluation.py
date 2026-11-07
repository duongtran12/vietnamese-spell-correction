from src.correction.candidates import CandidateGenerator
from src.correction.ranker import CandidateRanker
from src.correction.ranking_evaluation import RankingExample, evaluate_ranker
from src.models.trigram_lm import TrigramLanguageModel


def test_context_ranking_improves_ambiguous_top_one_baseline() -> None:
    training = [
        ["tôi", "đi", "học"],
        ["tôi", "đi", "học"],
        ["cuộc", "họp", "bắt", "đầu"],
        ["cuộc", "họp", "kết", "thúc"],
        ["cuộc", "họp", "quan", "trọng"],
    ]
    generator = CandidateGenerator({"học": 2, "họp": 3})
    ranker = CandidateRanker(TrigramLanguageModel().fit(training))
    examples = [RankingExample(("tôi", "đi", "họ"), 2, "học", "delete")]

    metrics = evaluate_ranker(generator, ranker, examples)

    assert metrics["candidate_recall_at_k"] == 1.0
    assert metrics["baseline_accuracy_at_k"]["1"] == 0.0
    assert metrics["context_accuracy_at_k"]["1"] == 1.0
    assert metrics["accuracy_at_1_delta"] == 1.0

