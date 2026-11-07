from src.correction.candidates import CandidateGenerator
from src.correction.evaluation import CandidateExample, evaluate_candidate_generator


def test_candidate_evaluation_reports_recall_and_coverage() -> None:
    generator = CandidateGenerator({"hoc": 10, "học": 8, "họp": 2})
    examples = [
        CandidateExample("hco", "hoc", "transpose"),
        CandidateExample("họcc", "học", "insert"),
        CandidateExample("xyz", "không-có", "substitute"),
    ]

    metrics = evaluate_candidate_generator(generator, examples)

    assert metrics["examples"] == 3
    assert metrics["recall_at_k"]["1"] == 2 / 3
    assert metrics["recall_at_k"]["5"] == 2 / 3
    assert metrics["recall_at_k_in_vocabulary"]["1"] == 1.0
    assert metrics["in_vocabulary_examples"] == 2
    assert metrics["target_oov_count"] == 1
    assert metrics["target_oov_rate"] == 1 / 3
    assert metrics["no_candidate_rate"] == 1 / 3
    assert metrics["recall_at_5_by_error_type"]["transpose"] == 1.0
    assert metrics["latency_ms"]["mean"] >= 0.0

