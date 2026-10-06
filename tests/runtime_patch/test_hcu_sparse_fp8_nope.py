import torch
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl

from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
    HcuFlashMLASparseImpl,
)


def test_nope_query_is_zero_padded_for_ds_fp8_mla(monkeypatch):
    observed = {}

    def capture_forward(self, q, kv_cache, metadata, layer):
        observed["q"] = q.clone()
        return q, None

    monkeypatch.setattr(FlashMLASparseImpl, "forward_mqa", capture_forward)
    impl = object.__new__(HcuFlashMLASparseImpl)
    impl.q_concat_buffer = torch.zeros((1, 64, 576), dtype=torch.bfloat16)
    impl.dcp_world_size = 1
    q_nope = torch.ones((1, 64, 512), dtype=torch.bfloat16)
    q_pe = torch.empty((1, 64, 0), dtype=torch.bfloat16)

    impl.forward_mqa((q_nope, q_pe), torch.empty(0), None, None)

    assert observed["q"].shape == (1, 64, 576)
    assert torch.all(observed["q"][..., :512] == 1)
    assert torch.all(observed["q"][..., 512:] == 0)


def test_nope_fp8_padding_only_applies_to_512_wide_query(monkeypatch):
    def init_with_other_head_size(self, *args, **kwargs):
        self.kv_cache_dtype = "fp8_ds_mla"
        self.qk_rope_head_dim = 0
        self.q_concat_buffer = torch.zeros((1, 64, 384), dtype=torch.bfloat16)

    monkeypatch.setattr(FlashMLASparseImpl, "__init__", init_with_other_head_size)
    impl = HcuFlashMLASparseImpl()
    assert not impl._fp8_nope
    assert impl.q_concat_buffer.shape[-1] == 384
