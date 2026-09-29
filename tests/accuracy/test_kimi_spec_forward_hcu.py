"""Synthetic K3 TP8-local KDA forward vs FP32 recurrence, no model service."""
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.hcu


def run(drafts, dim_first, mixed, graph, prefill=False):
    from vllm.model_executor.layers.mamba import mamba_utils
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata, compute_causal_conv1d_metadata
    from vllm_hcu.models.kimi_k3.amd import kimi_gdn_linear_attn as model
    torch.manual_seed(137)
    h, d, width = 12, 128, 1536
    length, blocks = drafts + 1, 32
    non_spec_length = 17 if prefill else int(mixed)
    tokens = 2 * length + non_spec_length
    original_layout = mamba_utils.is_conv_state_dim_first
    original_model_layout = model.is_conv_state_dim_first
    try:
        mamba_utils.is_conv_state_dim_first = lambda: dim_first
        model.is_conv_state_dim_first = lambda: dim_first
        conv_shape, rec_shape = model.KimiGatedDeltaNetAttention.get_state_shape(
            SimpleNamespace(
                tp_size=8,
                num_heads=96,
                head_dim=d,
                conv_size=4,
                num_spec=drafts,
            )
        )
    finally:
        mamba_utils.is_conv_state_dim_first = original_layout
        model.is_conv_state_dim_first = original_model_layout
    conv_slab = torch.zeros(blocks, 2, *conv_shape, dtype=torch.bfloat16, device="cuda")
    conv_cache = conv_slab[:, 0]
    conv = conv_cache if dim_first else conv_cache.transpose(-1, -2)
    rec_slab = torch.zeros(blocks, 2, *rec_shape, dtype=torch.float32, device="cuda")
    rec = rec_slab[:, 0]
    layer = SimpleNamespace(prefix="layer", local_projection_size=width,
        local_num_heads=h, head_dim=d, gate_lower_bound=-5.,
        conv1d=SimpleNamespace(weight=torch.randn(3 * width, 1, 4, device="cuda") * .1, bias=None),
        A_log=torch.zeros(h, device="cuda"), dt_bias=torch.zeros(width, device="cuda"),
        o_norm=model.FusedRMSNormGated(d, eps=1e-6, activation="sigmoid").cuda(),
        kv_cache=(conv_cache, rec))
    layer.o_norm.weight.fill_(1)
    packed = torch.empty(tokens, 5 * width + h, device="cuda", dtype=torch.bfloat16)
    qkv = packed[:, :3 * width]
    g1 = packed[:, 3 * width:4 * width].view(1, tokens, h, d)
    g2 = packed[:, 4 * width:5 * width].view(tokens, h, d)
    beta = packed[:, 5 * width:].view(1, tokens, h)
    out = torch.empty(1, tokens, h, d, device="cuda", dtype=torch.bfloat16)
    spec_ids = torch.tensor([[2 + 2 * j for j in range(length)],
                            [13 + 2 * j for j in range(length)],
                            [25] * length], dtype=torch.int32, device="cuda")
    accepted = torch.ones(3, dtype=torch.int32, device="cuda")
    spec_tokens = list(range(length)) + list(range(length + non_spec_length, tokens))
    starts_cpu = torch.tensor([0, non_spec_length], dtype=torch.int32)
    nums, ptr, offsets = compute_causal_conv1d_metadata(starts_cpu, device=torch.device("cuda"))
    metadata = GDNAttentionMetadata(num_prefills=int(prefill), num_prefill_tokens=non_spec_length if prefill else 0,
        num_decodes=int(mixed and not prefill), num_decode_tokens=int(mixed and not prefill), num_spec_decodes=3,
        num_spec_decode_tokens=2 * length, num_actual_tokens=tokens,
        spec_sequence_masks=torch.tensor([True, True, True, False] if mixed else [True] * 3, device="cuda"),
        spec_token_indx=torch.tensor(spec_tokens, device="cuda"),
        non_spec_token_indx=torch.arange(length, length + non_spec_length, dtype=torch.int64, device="cuda"),
        spec_state_indices_tensor=spec_ids,
        spec_query_start_loc=torch.tensor([0, length, 2 * length, 2 * length], device="cuda", dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([31], device="cuda", dtype=torch.int32),
        non_spec_query_start_loc=starts_cpu.cuda(),
        has_initial_state=torch.tensor([True], device="cuda"),
        nums_dict=nums, batch_ptr=ptr, token_chunk_offset_ptr=offsets,
        num_accepted_tokens=accepted)
    original_context = model.get_forward_context
    try:
        mamba_utils.is_conv_state_dim_first = lambda: dim_first
        model.is_conv_state_dim_first = lambda: dim_first
        model.get_forward_context = lambda: SimpleNamespace(
            attn_metadata={"layer": metadata}
        )

        def launch():
            model.KimiGatedDeltaNetAttention._forward(
                layer, qkv, g1, g2, beta, out
            )

        packed.zero_()
        launch()
        captured = None
        if graph:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                launch()
        errors = {"output": 0., "recurrent": 0.}
        for keep in range(1, length + 1):
            conv.copy_(torch.randn_like(conv) * .1)
            rec.copy_(torch.randn_like(rec) * .01)
            accepted.copy_(torch.tensor([keep, length + 1 - keep, 1], device="cuda"))
            for _ in range(2):
                packed.copy_(torch.randn_like(packed) * .1)
                before = packed.clone()
                expected_conv, expected_rec = conv.clone(), rec.clone()
                expected_out = torch.empty_like(out)
                sequences = [(spec_tokens[:length], spec_ids[0].tolist(), keep),
                             (spec_tokens[length:], spec_ids[1].tolist(), length + 1 - keep)]
                if mixed:
                    sequences.append((list(range(length, length + non_spec_length)), [31], None))
                for positions, state_ids, count in sequences:
                    base = state_ids[0]
                    offset = count - 1 if count is not None else 0
                    history = expected_conv[base, :, offset:offset + 3].float()
                    raw = before[positions, :3 * width].T.float()
                    full = torch.cat((history, raw), 1)
                    weights = layer.conv1d.weight[:, 0]
                    convolved = F.silu(sum(full[:, j:j + len(positions)] * weights[:, j:j + 1]
                                           for j in range(4))).T.to(torch.bfloat16)
                    if count is not None:
                        expected_conv[base].copy_(full[:, 1:])
                    else:
                        # Plain decode updates the canonical width-1 history.
                        expected_conv[base, :, :3].copy_(full[:, -3:])
                    state = expected_rec[state_ids[offset]].clone()
                    for step, position in enumerate(positions):
                        q, k, v = [part.reshape(h, d).float() for part in convolved[step].chunk(3)]
                        q /= (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                        k /= (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                        gate = -5. * torch.sigmoid(g1[0, position].float())
                        state *= gate.exp().unsqueeze(-2)
                        delta = (v - (state * k.unsqueeze(-2)).sum(-1)) * torch.sigmoid(beta[0, position].float()).unsqueeze(-1)
                        state += delta.unsqueeze(-1) * k.unsqueeze(-2)
                        value = (state * (q / d ** .5).unsqueeze(-2)).sum(-1).to(torch.bfloat16).float()
                        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                        value *= torch.sigmoid(g2[position].float())
                        expected_out[0, position] = value.to(torch.bfloat16)
                        expected_rec[state_ids[step] if count is not None else base] = state
                captured.replay() if graph else launch()
                torch.cuda.synchronize()
                assert torch.isfinite(out).all() and torch.isfinite(rec).all()
                torch.testing.assert_close(out, expected_out, atol=.01, rtol=.03)
                torch.testing.assert_close(rec, expected_rec, atol=3e-5, rtol=.02)
                torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
                torch.testing.assert_close(packed[:, 3 * width:], before[:, 3 * width:], atol=0, rtol=0)
                assert torch.count_nonzero(conv_slab[:, 1]) == 0
                assert torch.count_nonzero(rec_slab[:, 1]) == 0
                errors["output"] = max(errors["output"], (out.float() - expected_out.float()).abs().max().item())
                errors["recurrent"] = max(errors["recurrent"], (rec - expected_rec).abs().max().item())
        return dict(drafts=drafts, dim_first=dim_first, mixed=mixed, graph=graph, prefill=prefill,
                    checks=2 * length, max_abs=errors, conv_exact=True, guards_exact=True)
    finally:
        model.get_forward_context = original_context
        model.is_conv_state_dim_first = original_model_layout
        mamba_utils.is_conv_state_dim_first = original_layout


@pytest.mark.parametrize("graph,mixed,prefill", [
    (False, False, False), (False, True, False), (True, False, False),
    (True, True, False), (False, True, True),
])
@pytest.mark.parametrize("drafts", [1, 4])
@pytest.mark.parametrize("dim_first", [False, True])
def test_full_kda_spec_state_against_fp32(drafts, dim_first, mixed, graph, prefill):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from vllm.config import VllmConfig, set_current_vllm_config
    with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
        print(json.dumps(run(drafts, dim_first, mixed, graph, prefill)), flush=True)
