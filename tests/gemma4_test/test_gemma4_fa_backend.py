"""CPU dispatch contracts; these tests do not validate native FA numerics."""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_switches(monkeypatch):
    for name in ("VLLM_HCU_GEMMA4_USE_FA", "HCU_FLASH_ATTN_Q5", "HCU_DEBUG_ATTN_Q_LEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def implementation(monkeypatch):
    from vllm_hcu.models import gemma4 as module

    calls = []

    def flash_forward(**kwargs):
        calls.append(("flash", kwargs))
        kwargs["out"].fill_(3)

    monkeypatch.setattr(module, "_gemma4_flash_attention", flash_forward)
    monkeypatch.setattr(module.envs, "VLLM_BATCH_INVARIANT", False)

    def initialize(self, heads=16, kv_heads=2):
        self.num_heads = heads
        self.num_kv_heads = kv_heads
        self.head_size = 512
        self.attn_type = module.AttentionType.DECODER
        self.sliding_window = (-1, -1)
        self._kv_quant_mode = module.KVQuantMode.NONE
        self.scale = 1.0
        self.logits_soft_cap = 0.0
        self.alibi_slopes = self.sinks = None
        self.use_alibi_sqrt = False
        self.chunk_lookback = -1

    def fallback(
        self,
        layer,
        query,
        key,
        value,
        cache,
        metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        calls.append(("triton", metadata))
        if output_block_scale is not None:
            raise NotImplementedError("original fallback")
        return output.fill_(-9)

    monkeypatch.setattr(module.TritonAttentionImpl, "__init__", initialize)
    monkeypatch.setattr(module.TritonAttentionImpl, "forward", fallback)
    return module, calls


def inputs(lengths, heads=16, kv_heads=2, padding=0, block=32):
    total = sum(lengths)
    q = torch.zeros((total + padding, heads, 512), dtype=torch.bfloat16)
    cache = torch.zeros((2, 2, block, kv_heads, 512), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        max_query_len=max(lengths),
        num_actual_tokens=total,
        use_cascade=False,
        mm_prefix_range=None,
        mm_prefix_range_tensor=None,
        query_start_loc=torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0),
        seq_lens=torch.full((len(lengths),), 64, dtype=torch.int32),
        max_seq_len=64,
        block_table=torch.zeros((len(lengths), 2), dtype=torch.int32),
    )
    return q, cache, metadata, torch.full_like(q, 7)


@pytest.mark.parametrize("batch", [1, 2, 20, 22, 24])
@pytest.mark.parametrize("q_len", [5, 17])
@pytest.mark.parametrize("block", [32, 64])
def test_flash_uses_all_requests_and_preserves_output_contract(
    implementation, batch, q_len, block
):
    module, calls = implementation
    impl = module.Gemma4FAAttentionImpl()
    q, cache, metadata, output = inputs([q_len] * batch, padding=3, block=block)
    result = impl.forward(None, q, q, q, cache, metadata, output)
    assert result is output
    assert calls[-1][0] == "flash"
    kwargs = calls[-1][1]
    assert kwargs["cu_seqlens_q"] is metadata.query_start_loc
    assert kwargs["block_table"] is metadata.block_table
    assert kwargs["q"].shape[0] == batch * q_len
    assert kwargs["softmax_scale"] == 1.0
    assert kwargs["k"].shape[1] == block
    assert torch.all(output[: batch * q_len] == 3)
    assert torch.all(output[batch * q_len :] == 7)


def test_unsupported_cache_block_falls_back(implementation):
    module, calls = implementation
    impl = module.Gemma4FAAttentionImpl()
    q, cache, metadata, output = inputs([5], block=128)
    impl.forward(None, q, q, q, cache, metadata, output)
    assert calls[-1][0] == "triton"


def test_mixed_queries_and_non_tp2_heads_do_not_force_batch_one(implementation):
    module, calls = implementation
    impl = module.Gemma4FAAttentionImpl(heads=8, kv_heads=1)
    q, cache, metadata, output = inputs([5, 1, 4, 0], heads=8, kv_heads=1)
    impl.forward(None, q, q, q, cache, metadata, output)
    assert calls[-1][0] == "flash"


@pytest.mark.parametrize("q_len", [1, 4, 16])
def test_other_query_lengths_retain_original_forward(implementation, q_len):
    module, calls = implementation
    impl = module.Gemma4FAAttentionImpl()
    q, cache, metadata, output = inputs([q_len])
    assert impl.forward(None, q, q, q, cache, metadata, output) is output
    assert calls[-1][0] == "triton"


@pytest.mark.parametrize(
    "field",
    ["mm_prefix_range", "mm_prefix_range_tensor", "rswa_prefix_lens", "rswa_window"],
)
def test_unrepresented_mask_semantics_fall_back(implementation, field):
    module, calls = implementation
    impl = module.Gemma4FAAttentionImpl()
    q, cache, metadata, output = inputs([5])
    setattr(metadata, field, object())
    impl.forward(None, q, q, q, cache, metadata, output)
    assert calls[-1][0] == "triton"


def test_profiling_metadata_none_uses_original_forward(implementation):
    module, calls = implementation
    impl = module.Gemma4FAAttentionImpl()
    q, cache, _, output = inputs([5])
    impl.forward(None, q, q, q, cache, None, output)
    assert calls[-1][0] == "triton"
