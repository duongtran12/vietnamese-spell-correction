"""Offline Streamlit demo for Vietnamese spelling correction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import torch
import yaml

from src.demo import (
    apply_token_choices,
    decision_rows,
    render_highlighted_text,
    suggestion_rows,
)
from src.inference import SpellingCorrectionPipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR = ROOT / "artifacts/viwiki-20261101-smoke-dedup-detector.pt"
DEFAULT_VOCABULARY = (
    ROOT / "data/processed/viwiki-20261101-smoke-dedup/vocabulary.json"
)
DEFAULT_TRIGRAM = ROOT / "artifacts/viwiki-20261101-smoke-dedup-trigram.json"
DEFAULT_EXAMPLE = "Hộm nay tôi đi hộc xử lý ngôn ngử tự nhiên."


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config phải chứa YAML object: {path}")
    return config


@st.cache_resource(show_spinner="Đang nạp mô hình cục bộ...")
def load_pipeline(
    detector_path: str,
    vocabulary_path: str,
    trigram_path: str,
) -> SpellingCorrectionPipeline:
    """Load local artifacts once for the current Streamlit process."""

    inference_config = _load_yaml(ROOT / "configs/inference.yaml")
    ranking_root = _load_yaml(ROOT / "configs/ranking.yaml")
    ranking_config = ranking_root.get("ranker", {})
    candidate_config = ranking_root.get("candidates", {})
    return SpellingCorrectionPipeline.from_artifacts(
        detector_checkpoint=detector_path,
        vocabulary_artifact=vocabulary_path,
        trigram_artifact=trigram_path,
        detector_threshold=float(inference_config.get("detector_threshold", 0.5)),
        minimum_rank_margin=float(inference_config.get("minimum_rank_margin", 0.0)),
        diacritic_rank_margin=float(
            inference_config.get("diacritic_rank_margin", 0.5)
        ),
        title_case_diacritic_rank_margin=float(
            inference_config.get("title_case_diacritic_rank_margin", 0.5)
        ),
        keep_original_bonus=float(inference_config.get("keep_original_bonus", 0.75)),
        candidate_pool_size=int(inference_config.get("candidate_pool_size", 5)),
        top_k_suggestions=int(inference_config.get("top_k_suggestions", 3)),
        context_weight=float(ranking_config.get("context_weight", 1.0)),
        edit_weight=float(ranking_config.get("edit_weight", 1.0)),
        frequency_weight=float(ranking_config.get("frequency_weight", 0.1)),
        candidate_distance=str(
            candidate_config.get("distance", "damerau_levenshtein")
        ),
        short_token_max_length=int(candidate_config.get("short_token_max_length", 4)),
        short_token_max_distance=int(
            candidate_config.get("short_token_max_distance", 1)
        ),
        long_token_max_distance=int(candidate_config.get("long_token_max_distance", 2)),
        device=torch.device("cpu"),
    )


def _unique_options(original: str, suggestions: list[str]) -> list[str]:
    options: list[str] = []
    for token in [original, *suggestions]:
        if token not in options:
            options.append(token)
    return options


def main() -> None:
    st.set_page_config(
        page_title="Vietnamese Spell Correction",
        page_icon="✍️",
        layout="wide",
    )
    st.title("✍️ Vietnamese Spell Correction")
    st.caption(
        "BiLSTM detector + Damerau–Levenshtein candidates + Trigram ranker — "
        "chạy hoàn toàn cục bộ."
    )

    with st.sidebar:
        st.header("Artifact cục bộ")
        detector_path = st.text_input("BiLSTM checkpoint", str(DEFAULT_DETECTOR))
        vocabulary_path = st.text_input("Vocabulary", str(DEFAULT_VOCABULARY))
        trigram_path = st.text_input("Trigram model", str(DEFAULT_TRIGRAM))
        st.divider()
        st.metric("Validation Correction F1", "47,79%")
        st.metric("Sentence Accuracy", "55,56%")
        st.metric("False Correction Rate", "0,58%")
        st.caption("Kết quả smoke validation; chưa phải kết quả test cuối.")

    example = st.selectbox(
        "Câu ví dụ",
        [
            DEFAULT_EXAMPLE,
            "Hôm qua chúng tôi đả hoàn thành bài tập.",
            "Hôm nay tôi đi học xử lý ngôn ngữ tự nhiên.",
        ],
    )
    text = st.text_area("Nhập câu tiếng Việt", value=example, height=110)

    if st.button("Kiểm tra chính tả", type="primary", width="stretch"):
        artifact_paths = [
            Path(detector_path),
            Path(vocabulary_path),
            Path(trigram_path),
        ]
        missing = [str(path) for path in artifact_paths if not path.is_file()]
        if missing:
            st.error("Không tìm thấy artifact:\n\n" + "\n".join(missing))
            st.stop()
        if not text.strip():
            st.warning("Hãy nhập ít nhất một từ tiếng Việt.")
            st.stop()
        try:
            pipeline = load_pipeline(detector_path, vocabulary_path, trigram_path)
            st.session_state["correction_result"] = pipeline.correct_text(text)
        except Exception as error:
            st.exception(error)
            st.stop()

    result = st.session_state.get("correction_result")
    if result is None:
        st.info("Nhập câu hoặc chọn ví dụ, sau đó nhấn **Kiểm tra chính tả**.")
        return

    st.subheader("Kết quả")
    left, right = st.columns(2)
    with left:
        st.markdown("**Câu đầu vào — đỏ: đã sửa, vàng: nghi sai nhưng giữ nguyên**")
        st.markdown(
            '<div style="font-size:1.15rem;line-height:2;padding:0.8rem;'
            'border:1px solid #d1d5db;border-radius:0.5rem">'
            + render_highlighted_text(result.original_text or "", result.decisions)
            + "</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**Câu sửa tự động Top-1**")
        st.success(result.corrected_text or "")

    if not result.decisions:
        st.success("Mô hình không phát hiện token nào vượt detector threshold.")
        return

    st.subheader("Token bị nghi sai")
    st.dataframe(decision_rows(result.decisions), width="stretch", hide_index=True)

    st.subheader("Chọn phương án thủ công")
    choices: dict[int, str] = {}
    for decision in result.decisions:
        suggestion_tokens = [item.token for item in decision.suggestions]
        options = _unique_options(decision.original_token, suggestion_tokens)
        default_token = decision.applied_token
        default_index = options.index(default_token) if default_token in options else 0
        choices[decision.token_index] = st.selectbox(
            f"Token #{decision.token_index}: {decision.original_token} "
            f"(confidence {decision.detector_confidence:.3f})",
            options,
            index=default_index,
            key=f"choice-{decision.token_index}-{decision.original_token}",
        )
        with st.expander(f"Chi tiết điểm — {decision.original_token}"):
            st.dataframe(
                suggestion_rows(decision),
                width="stretch",
                hide_index=True,
            )

    custom_text = apply_token_choices(result.original_text or "", choices)
    st.markdown("**Câu sau khi áp dụng lựa chọn của người dùng**")
    st.success(custom_text)

    with st.expander("Giới hạn của mô hình"):
        st.markdown(
            "- Mô hình hiện được huấn luyện trên một shard Wikipedia nhỏ.\n"
            "- Từ đúng không có trong từ điển train có thể không được đề xuất.\n"
            "- Detector còn false positive; hệ thống dùng keep-original và rank margin "
            "để hạn chế sửa nhầm.\n"
            "- Kết quả hiện tại là validation smoke experiment, chưa phải test cuối."
        )


if __name__ == "__main__":
    main()
