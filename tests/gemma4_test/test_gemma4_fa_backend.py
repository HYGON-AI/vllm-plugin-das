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


@pytest.mark.parametrize("q_len", [5, 17])
@pytest.mark.parametrize("ranges", [{}, {0: [], 1: []}])
def test_empty_prefix_lm_ranges_admit_text_without_reading_device_mask(
    implementation, q_len, ranges
):
    module, calls = implementation
    impl = module.Gemma4FAAttentionImpl()
    q, cache, metadata, output = inputs([q_len, q_len])
    metadata.mm_prefix_range = ranges
    # Meta tensors cannot be copied to the host or inspected for mask values.
    metadata.mm_prefix_range_tensor = torch.empty((2, 1, 2), device="meta")
    impl.forward(None, q, q, q, cache, metadata, output)
    assert calls[-1][0] == "flash"
    assert metadata.mm_prefix_range is ranges
    assert metadata.mm_prefix_range_tensor.device.type == "meta"


@pytest.mark.parametrize("q_len", [5, 17])
@pytest.mark.parametrize("ranges", [{0: [(1, 4)]}, {0: [], 1: [(1, 4)]}])
def test_real_bidirectional_prefix_ranges_remain_on_triton(
    implementation, q_len, ranges
):
    module, calls = implementation
    impl = module.Gemma4FAAttentionImpl()
    q, cache, metadata, output = inputs([q_len, q_len])
    metadata.mm_prefix_range = ranges
    metadata.mm_prefix_range_tensor = torch.empty((2, 1, 2), device="meta")
    impl.forward(None, q, q, q, cache, metadata, output)
    assert calls[-1][0] == "triton"


@pytest.mark.parametrize("shape", [(0,), (2, 1, 2)])
def test_tensor_only_prefix_mask_is_not_assumed_empty(implementation, shape):
    module, calls = implementation
    impl = module.Gemma4FAAttentionImpl()
    q, cache, metadata, output = inputs([5])
    metadata.mm_prefix_range_tensor = torch.empty(shape, device="meta")
    impl.forward(None, q, q, q, cache, metadata, output)
    assert calls[-1][0] == "triton"


@pytest.mark.parametrize("sharing", [False, True])
def test_outer_attention_updates_kv_before_fa_and_preserves_sharing(
    implementation, monkeypatch, sharing
):
    from vllm_hcu.model_executor.layers.attention_forward_runtime import (
        attention_forward,
    )

    module, calls = implementation
    # This is inherited from vLLM 0.25.1; writing KV inside forward would
    # duplicate the outer update and break KV-sharing ownership.
    assert module.Gemma4FAAttentionBackend.forward_includes_kv_cache_update is False
    assert callable(module.TritonAttentionImpl.do_kv_cache_update)
    impl = module.Gemma4FAAttentionImpl()
    q, cache, metadata, _ = inputs([5])
    key = torch.full((5, 2, 512), 2, dtype=torch.bfloat16)
    value = torch.full_like(key, 6)
    layer = SimpleNamespace(
        calculate_kv_scales=False,
        query_quant=None,
        num_heads=16,
        num_kv_heads=2,
        head_size=512,
        head_size_v=512,
        layer_name="gemma4.full.attn",
        impl=impl,
        attn_backend=module.Gemma4FAAttentionBackend,
        kv_sharing_target_layer_name="target" if sharing else None,
    )
    events = []

    def update(layer, k, v, kv, slots):
        events.append("update")
        kv[0, 0, :5].copy_(k)
        kv[0, 1, :5].copy_(v)

    def flash(**kwargs):
        events.append("flash")
        assert torch.equal(kwargs["k"][0, :5], key)
        assert torch.equal(kwargs["v"][0, :5], value)
        kwargs["out"].fill_(3)

    def compute(query, k, v, output, layer_name, **kwargs):
        return impl.forward(layer, query, k, v, cache, metadata, output)

    monkeypatch.setattr(impl, "do_kv_cache_update", update)
    monkeypatch.setattr(impl, "_flash_attention", flash)
    if sharing:
        # The source layer has already populated the shared target cache.
        cache[0, 0, :5].copy_(key)
        cache[0, 1, :5].copy_(value)
    upstream = SimpleNamespace(
        _resolve_layer_name=lambda name: name,
        get_attention_context=lambda name: (None, layer, cache, torch.arange(5)),
        unified_attention_with_output=compute,
    )
    output = attention_forward(upstream, layer, q, key, value)
    assert events == (["flash"] if sharing else ["update", "flash"])
    assert torch.all(output == 3)
    assert not calls  # Neither the original Triton forward nor its FA stub ran.
