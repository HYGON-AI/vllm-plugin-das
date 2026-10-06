def test_sparse_mla_backend_follows_custom_ops_master(monkeypatch):
    from boltops.mla import flash_mla_sparse_fwd as boltops_sparse_mla_fwd

    from vllm_hcu.v1.attention.ops import flashmla

    # The legacy child flag must not override the master switch.
    monkeypatch.setenv("VLLM_HCU_SPARSE_MLA_BOLTOPS", "1")
    flashmla._resolve_sparse_mla_fwd.cache_clear()
    try:
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
        assert (
            flashmla._resolve_sparse_mla_fwd() is flashmla._native_flash_mla_sparse_fwd
        )
        flashmla._resolve_sparse_mla_fwd.cache_clear()
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
        assert flashmla._resolve_sparse_mla_fwd() is boltops_sparse_mla_fwd
        flashmla._resolve_sparse_mla_fwd.cache_clear()
        monkeypatch.delenv("VLLM_HCU_SPARSE_MLA_BOLTOPS")
        assert flashmla._resolve_sparse_mla_fwd() is boltops_sparse_mla_fwd
    finally:
        flashmla._resolve_sparse_mla_fwd.cache_clear()


def test_boltops_sparse_mla_rejects_fp8_kv(monkeypatch):
    import torch

    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
        HcuFlashMLASparseBackend,
    )

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.delenv("VLLM_HCU_SPARSE_MLA_BOLTOPS", raising=False)
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
