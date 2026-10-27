# SPDX-License-Identifier: Apache-2.0
"""Compare BoltOps outputs and exported history against the original KDA.

Run in the deployed 0.28 container with VLLM_PLUGINS=__disabled__.
Both initial and checkpoint states are deliberately asymmetric.
"""
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers import kimi_k3_kda as kda

pytestmark = pytest.mark.hcu


def inputs(lengths, initial=None):
    total = sum(lengths)
    shape = (1, total, 2, 128)
    cu_cpu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()))
    ci = torch.tensor([(i, c) for i, n in enumerate(lengths)
                       for c in range((n + 63) // 64)], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0] + list(torch.tensor([(n + 63) // 64 for n in lengths])
                                    .cumsum(0).tolist()), device="cuda")
    return dict(q=torch.randn(shape, device="cuda", dtype=torch.bfloat16),
                k=torch.randn(shape, device="cuda", dtype=torch.bfloat16),
                v=torch.randn(shape, device="cuda", dtype=torch.bfloat16),
                raw_g=torch.randn(shape, device="cuda", dtype=torch.bfloat16),
                raw_beta=torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16),
                A_log=torch.randn(2, device="cuda"),
                g_bias=torch.randn(256, device="cuda"), lower_bound=-5.0,
                initial_state=(initial if initial is not None else torch.randn(
                    len(lengths), 2, 128, 128, device="cuda") * 0.1),
                cu_seqlens=cu_cpu.to(device="cuda", dtype=torch.int32),
                chunk_indices=ci, chunk_offsets=offsets,
                output_final_state=True, use_qk_l2norm_in_kernel=True)


def run_original(kwargs):
    from vllm.models.kimi_k3.amd.ops.third_party.kda import chunk_kda_with_fused_gate
    return chunk_kda_with_fused_gate(**{
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in kwargs.items()})


def run_flash_kda(kwargs):
    from flash_kda import fwd
    q, k, v = (kwargs[name] for name in ("q", "k", "v"))
    initial = kwargs["initial_state"]
    output, final = torch.empty_like(v), torch.empty_like(initial)
    fwd(q.contiguous().clone(), k.contiguous().clone(),
        v.clone(memory_format=torch.contiguous_format),
        kwargs["raw_g"].contiguous().clone(), kwargs["raw_beta"].contiguous().clone(),
        float(q.shape[-1] ** -0.5), output, kwargs["A_log"].contiguous().clone(),
        kwargs["g_bias"].view(q.shape[-2], q.shape[-1]).contiguous().clone(),
        float(kwargs["lower_bound"]), initial_state=initial.contiguous().clone(),
        final_state=final,
        cu_seqlens=kwargs["cu_seqlens"].to(torch.int64).contiguous())
    return output, final


def compare(actual, expected, label):
    delta = (actual.float() - expected.float()).abs()
    print(label, "max_abs", delta.max().item(), "rms", delta.square().mean().sqrt().item())
    # BF16 state snapshots introduce a rounding at the cached boundary.
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.03)


@pytest.mark.parametrize("lengths", [[4096], [8192], [8175], [4096, 8175]])
@torch.inference_mode()
def test_output_final_checkpoint_and_prefix_reuse(lengths):
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    torch.manual_seed(20261026)
    kwargs = inputs(lengths)
    boundaries = [0] + list(torch.tensor(lengths).cumsum(0).tolist())
    _, _, hi, ce, pos = kda.checkpoint_plan(
        boundaries, lengths, num_decodes=0, block_size=1536, hash_size=1536)
    slots = torch.arange(1, len(lengths) + 1, device="cuda")
    recurrent = torch.zeros(len(lengths) + 2, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
    metadata = SimpleNamespace(nums_dict={kda.CHECKPOINT_KEY: kda.Checkpoints(
        slots, torch.tensor(hi, device="cuda"), tuple(ce), tuple(pos))})
    token = kda.ACTIVE_LAYER.set((SimpleNamespace(kv_cache=(None, recurrent)), metadata))
    try:
        out, final = kda.chunk_kda(**kwargs)
    finally:
        kda.ACTIVE_LAYER.reset(token)
    # Simulate the layer's ordinary final-state store into reserved slots;
    # checkpoint publication must run afterward and retain its own state.
    recurrent[slots] = final.to(recurrent.dtype)
    kda.finalize_checkpoints(metadata)
    ref_out, ref_final = run_original(kwargs)
    compare(out, ref_out, "output")
    compare(final, ref_final, "final")
    flash_out, flash_final = run_flash_kda(kwargs)
    compare(out, flash_out, "boltops-flash-output")
    compare(final, flash_final, "boltops-flash-final")
    for row, checkpoint in enumerate(pos):
        begin, end = boundaries[row:row + 2]
        prefix = dict(kwargs)
        for name in ("q", "k", "v", "raw_g", "raw_beta"):
            prefix[name] = kwargs[name][:, begin:begin + checkpoint].contiguous()
        prefix.update(initial_state=kwargs["initial_state"][row:row + 1],
                      cu_seqlens=torch.tensor([0, checkpoint], device="cuda", dtype=torch.int32),
                      chunk_indices=torch.tensor([(0, c) for c in range(checkpoint // 64)],
                                                 device="cuda", dtype=torch.int32),
                      chunk_offsets=torch.tensor([0, checkpoint // 64], device="cuda"))
        _, prefix_final = run_original(prefix)
        compare(recurrent[row + 1], prefix_final[0], "checkpoint")
        tail = dict(prefix)
        for name in ("q", "k", "v", "raw_g", "raw_beta"):
            tail[name] = kwargs[name][:, begin + checkpoint:end].contiguous()
        tail_len = end - begin - checkpoint
        tail.update(initial_state=recurrent[row + 1:row + 2].float(),
                    cu_seqlens=torch.tensor([0, tail_len], device="cuda", dtype=torch.int32),
                    chunk_indices=torch.tensor([(0, c) for c in range((tail_len + 63) // 64)],
                                               device="cuda", dtype=torch.int32),
                    chunk_offsets=torch.tensor([0, (tail_len + 63) // 64], device="cuda"))
        resumed_out, resumed_final = kda.chunk_kda(**tail)
        compare(resumed_out, ref_out[:, begin + checkpoint:end], "prefix-hit-output")
        compare(resumed_final[0], ref_final[row], "prefix-hit-final")
