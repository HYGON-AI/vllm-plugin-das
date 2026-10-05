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
