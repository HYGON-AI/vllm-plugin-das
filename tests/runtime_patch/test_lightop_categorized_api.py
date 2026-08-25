from importlib import import_module

import pytest


REQUIRED_EXPORTS = {
    "lightop.activation": {
        "fuse_silu_mul_fp8_quant",
        "fuse_silu_mul_fp8_quant_ep",
        "fuse_silu_mul_per_token_quant",
        "fuse_silu_mul_quant",
        "fuse_silu_mul_quant_ep",
        "silu_and_mul_opt",
    },
    "lightop.attention": {
        "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32",
        "get_paged_mqa_logits_metadata",
        "mqa_logits",
        "paged_mqa_logits",
        "split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant",
        "top_k_per_row_decode",
        "top_k_per_row_prefill",
    },
    "lightop.gemm_ops": {
        "hipblaslt_w8a8_channelwise_gemm",
        "m_grouped_w8a8_gemm_nt_contig_asm",
        "m_grouped_w8a8_gemm_nt_masked",
    },
    "lightop.moe": {
        "ep_gather",
        "ep_scatter",
        "fused_experts_impl_fp8_marlin",
        "fused_experts_impl_int8_marlin",
        "moe_align_block_size_out",
        "moe_fused_gate",
    },
    "lightop.norm": {
        "fused_add_rms_norm",
        "gemma_fused_add_rmsnorm",
        "gemma_rmsnorm",
        "rms_norm_dynamic_per_token_quant",
        "rmsnorm_forward_autograd",
    },
    "lightop.quant": {"per_token_quant_fp8", "per_token_quant_int8"},
    "lightop.tensor": {"ds_cat"},
}


@pytest.mark.parametrize("module_name", sorted(REQUIRED_EXPORTS))
def test_categorized_lightop_exports(module_name: str) -> None:
    pytest.importorskip("lightop")
    module = import_module(module_name)
    missing = sorted(
        name for name in REQUIRED_EXPORTS[module_name] if not hasattr(module, name)
    )
    assert not missing, f"{module_name} is missing required exports: {missing}"
