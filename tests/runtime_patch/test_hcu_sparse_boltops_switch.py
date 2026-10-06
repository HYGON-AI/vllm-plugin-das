import pytest


def test_sparse_mla_boltops_switch_is_independent_of_custom_ops(monkeypatch):
    from boltops.mla import flash_mla_sparse_fwd as boltops_sparse_mla_fwd

    from vllm_hcu.v1.attention.ops import flashmla

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_SPARSE_MLA_BOLTOPS", "1")
    flashmla._resolve_sparse_mla_fwd.cache_clear()
    try:
        assert flashmla._resolve_sparse_mla_fwd() is boltops_sparse_mla_fwd
    finally:
        flashmla._resolve_sparse_mla_fwd.cache_clear()


@pytest.mark.parametrize("disable_via", ["boltops", "custom_ops_master"])
def test_boltops_sparse_mla_rejects_fp8_kv(monkeypatch, disable_via):
    import torch

    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
        HcuFlashMLASparseBackend,
    )

    monkeypatch.setenv(
        "VLLM_HCU_USE_CUSTOM_OPS", "0" if disable_via == "custom_ops_master" else "1"
    )
    monkeypatch.setenv(
        "VLLM_HCU_SPARSE_MLA_BOLTOPS", "1" if disable_via == "boltops" else "0"
    )
    reason = HcuFlashMLASparseBackend.supports_combination(
        head_size=512,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8_ds_mla",
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
        device_capability=None,
    )
    assert reason is not None and "BoltOPs" in reason
