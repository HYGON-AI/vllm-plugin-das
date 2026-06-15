from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vllm_config_with_hf_config_patch_recomputes_hf_text_config() -> None:
    source = (
        ROOT / "vllm_hcu/patches/vllm__config__vllm.patch.py"
    ).read_text(encoding="utf-8")

    assert "model_config.hf_config = hf_config" in source
    assert "model_config.hf_text_config = model_config.hf_config.get_text_config()" in source


def test_qwen3_vl_moe_patch_syncs_nested_tie_before_language_model_build() -> None:
    source = (
        ROOT / "vllm_hcu/patches/vllm__model_executor__models__qwen3_vl_moe.patch.py"
    ).read_text(encoding="utf-8")

    assert "config.text_config.tie_word_embeddings = config.tie_word_embeddings" in source
    assert "vllm_config=vllm_config.with_hf_config(config.text_config)" in source
