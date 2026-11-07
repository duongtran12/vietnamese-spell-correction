from src.data.noise import generate_noisy_tokens
from src.data.normalize import normalize_text, tokenize
from src.data.split import assert_no_document_leakage, split_by_document


def test_data_pipeline_smoke() -> None:
    documents = [
        {
            "document_id": f"doc-{index}",
            "text": normalize_text(f"Tôi đang học xử lý ngôn ngữ tự nhiên số {index}."),
        }
        for index in range(10)
    ]

    splits = split_by_document(documents, seed=42)
    assert_no_document_leakage(splits)

    clean_tokens = tokenize(str(splits["train"][0]["text"]))
    noisy = generate_noisy_tokens(
        clean_tokens,
        seed=42,
        sentence_error_probability=1.0,
    )

    assert len(noisy.events) == 1
    assert noisy.tokens != clean_tokens
    assert len(noisy.tokens) == len(clean_tokens)

