# SPDX-License-Identifier: Apache-2.0
"""Live gfx938 tests using the vLLM packed-cache writer and FP32 reference."""

import pytest
import torch

from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla_decode as runtime

pytestmark = pytest.mark.hcu


def make_cache(page, blocks=4):
    from vllm.models.deepseek_v41.common.ops.cache_utils import quantize_and_insert_k_cache

    stride = ((page * 584 + 575) // 576) * 576
    storage = torch.zeros(blocks, stride, device="cuda", dtype=torch.uint8)
    kv = torch.randn(blocks * page, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    slots = torch.arange(blocks * page, device="cuda", dtype=torch.int64)
    quantize_and_insert_k_cache(kv, storage, slots, block_size=page, use_fnuz=False)
    torch.cuda.synchronize()
    # Independent page-major decoding, NOT a synthetic interleaved token view.
    data = storage[:, :page * 576].reshape(blocks, page, 576)
    fp8 = data[..., :448].contiguous().view(torch.float8_e4m3fn).float().reshape(blocks, page, 7, 64)
    exponents = storage[:, page * 576:page * 584].reshape(blocks, page, 8)[..., :7].float() - 127
    nope = (fp8 * torch.exp2(exponents)[..., None]).reshape(blocks * page, 448)
    rope = data[..., 448:].contiguous().view(torch.bfloat16).float().reshape(blocks * page, 64)
    decoded = torch.cat((nope, rope), -1)
    cache = storage.as_strided((blocks, page, 584), (stride, 584, 1))
    return cache, decoded


def reference(q, decoded, indices, sink):
    valid = (indices >= 0) & (indices < decoded.shape[0])
    values = decoded[indices.clamp(0, decoded.shape[0] - 1).long()]
    values = torch.where(valid[..., None], values, 0)
    scores = torch.einsum("thd,tkd->thk", q.float(), values) * 512**-0.5
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
    # Softmax(+inf) is NaN in torch; the attention sink contract is zero mass
    # for all real KV entries on these heads, not NaN.
    sink_scores = torch.where(torch.isposinf(sink), 0, sink)[None, :, None].expand(q.shape[0], -1, 1)
    weights = torch.cat((scores, sink_scores), -1).softmax(-1)[..., :-1]
    weights = torch.where(torch.isposinf(sink)[None, :, None], 0, weights).nan_to_num()
    return torch.einsum("thk,tkd->thd", weights, values)


@pytest.fixture
def live(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires DCU")
    if not runtime._supported_device(torch.empty(1, device="cuda")):
        pytest.skip("requires gfx938")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE", "1")
    if runtime._decode_bindings() is None:
        pytest.skip("native FlashMLA unavailable")


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("page", [None, 32, 64])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("swa_page", [32, 64])
def test_actual_writer_three_cache_types(live, heads, page, batch, swa_page):
    torch.manual_seed(1234)
    q = torch.randn(batch, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    main_cache, main_values = make_cache(swa_page)
    swa_indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(batch, 1)
    swa_lens = torch.full((batch,), 97, device="cuda", dtype=torch.int32)
    if batch == 3:
        swa_lens[0] = 0
        swa_lens[1] = 1
    sink = torch.linspace(-2, 2, heads, device="cuda")
    sink[0] = torch.inf
    if batch == 1:
        sink[1] = -torch.inf
    out_storage = torch.full((batch * 2, heads, 512), 7, device="cuda", dtype=torch.bfloat16)
    args = dict(
        q=q, swa_cache=main_cache, kv_cache=None, swa_indices=swa_indices,
        swa_lens=swa_lens, scale=512**-0.5, attn_sink=sink, output=out_storage[::2],
        head_dim=512, nope_head_dim=448, rope_head_dim=64,
        scheduler=runtime._decode_bindings()[2]()[0],
    )
    columns = torch.arange(128, device="cuda")
    ref_indices = torch.where(columns[None, :] < swa_lens[:, None], swa_indices, -1)
    values = main_values
    if page is not None:
        extra_cache, extra_values = make_cache(page)
        # Nonidentity block table and scattered holes exercise physical mapping.
        local = torch.arange(128, device="cuda", dtype=torch.int32).repeat(batch, 1)
        local[:, 1::3] = -1
        local[:, -3:] = torch.tensor([0, 0, -1], device="cuda", dtype=torch.int32)
        table = torch.stack([
            torch.tensor([3, 1, 2, 0], device="cuda", dtype=torch.int32).roll(row)
            for row in range(batch)
        ])
        valid = torch.ones(batch, device="cuda", dtype=torch.bool)
        if batch == 3:
            valid[0] = False
        args.update(
            kv_cache=extra_cache, topk_indices=local, block_table=table,
            token_to_req_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
            is_valid_token=valid,
        )
        physical = table.gather(1, local.clamp_min(0).long() // page) * page + local.clamp_min(0) % page
        physical = torch.where((local >= 0) & valid[:, None], physical + main_values.shape[0], -1)
        ref_indices = torch.cat((ref_indices, physical), -1)
        values = torch.cat((values, extra_values))
    original = swa_indices.clone()
    assert runtime.try_flashmla_decode(**args), "must not benchmark/test fallback"
    torch.cuda.synchronize()
    expected = reference(q, values, ref_indices, sink)
    assert torch.isfinite(args["output"]).all()
    torch.testing.assert_close(args["output"].float(), expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(swa_indices, original)
    assert torch.all(out_storage[1::2] == 7)
    # A changed step and changed lengths must use a new native plan.
    args["swa_lens"] = torch.full_like(swa_lens, 17)
    assert runtime.try_flashmla_decode(**args)
    torch.cuda.synchronize()
    ref_indices[:, :128] = torch.where(columns[None, :] < 17, swa_indices, -1)
    torch.testing.assert_close(args["output"].float(), reference(q, values, ref_indices, sink), atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("swa_page", [32, 64])
def test_optional_sink(live, swa_page):
    cache, values = make_cache(swa_page)
    q = torch.randn(1, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    indices = torch.arange(128, device="cuda", dtype=torch.int32).view(1, -1)
    args = dict(
        q=q, swa_cache=cache, kv_cache=None, swa_indices=indices,
        swa_lens=torch.tensor([128], device="cuda", dtype=torch.int32),
        scheduler=object(), scale=512**-0.5, attn_sink=None, output=torch.empty_like(q),
        head_dim=512, nope_head_dim=448, rope_head_dim=64,
    )
    assert runtime.try_flashmla_decode(**args)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        args["output"].float(),
        reference(q, values, indices, torch.full((64,), -torch.inf, device="cuda")),
        atol=2e-3, rtol=2e-2,
    )
