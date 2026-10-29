# SPDX-License-Identifier: Apache-2.0
"""Offline KDA cache replay using the deployed forward and V2 pre-copy kernels.

No engine, HTTP server, model weights or process group is created. Inputs are
fixed projected tensors, not token IDs. This isolates conv + KDA + gated norm
from projections, MLA, the scheduler and the block allocator.

Set KDA_REPRO_DIR to retain JSON reports and per-case profiler traces.
"""
from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers import kimi_k3_kda as adapter
from vllm_hcu.model_executor.layers.kimi_k3_prefill import LENGTHS_KEY

pytestmark = pytest.mark.hcu

HEADS, DIM, WINDOW, BLOCK = 2, 128, 3, 1536


def _tensor(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype, device="cuda")


def _pool(padded, dim_first):
    channels = 3 * HEADS * DIM
    conv_size, ssm_size = channels * WINDOW, HEADS * DIM * DIM
    # Put both state types in one page, as in a hybrid vLLM allocation.
    page_bytes = conv_size * 2 + ssm_size * 4
    if padded:
        page_bytes = math.ceil(page_bytes * 1.0868 / 256) * 256
    backing = torch.full((8, page_bytes), 0xA5, device="cuda", dtype=torch.uint8)
    conv_backing = backing.view(torch.bfloat16)
    recurrent_backing = backing.view(torch.float32)
    if dim_first:
        conv = conv_backing.as_strided((8, channels, WINDOW),
                                       (page_bytes // 2, WINDOW, 1))
    else:
        conv = conv_backing.as_strided((8, WINDOW, channels),
                                       (page_bytes // 2, channels, 1))
    recurrent = recurrent_backing.as_strided((8, HEADS, DIM, DIM),
                                             (page_bytes // 4, DIM * DIM, DIM, 1),
                                             conv_size // 2)
    conv.zero_()
    recurrent.zero_()
    return conv, recurrent


def _metadata(length, start, active_slot, checkpoint=False):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    cu_cpu = torch.tensor([0, length], dtype=torch.int32)
    cu = cu_cpu.to("cuda")
    nums, batch, offsets = compute_causal_conv1d_metadata(cu_cpu, device=cu.device)
    nums[LENGTHS_KEY] = [length]
    chunks = math.ceil(length / 64)
    m = GDNAttentionMetadata(
        num_prefills=1, num_prefill_tokens=length, num_decodes=0,
        num_decode_tokens=0, num_spec_decodes=0, num_spec_decode_tokens=0,
        num_actual_tokens=length, has_initial_state=_tensor([start > 0], torch.bool),
        non_spec_query_start_loc=cu,
        non_spec_state_indices_tensor=_tensor([active_slot]),
        prefill_query_start_loc=cu,
        prefill_state_indices=_tensor([active_slot]),
        prefill_has_initial_state=_tensor([start > 0], torch.bool),
        chunk_indices=_tensor([(0, c) for c in range(chunks)]),
        chunk_offsets=_tensor([0, chunks], torch.int64), nums_dict=nums,
        batch_ptr=batch, token_chunk_offset_ptr=offsets,
    )
    if checkpoint:
        rows, cols, hi, ends, positions = adapter.checkpoint_plan(
            [0, length], [start + length], num_decodes=0,
            block_size=BLOCK, hash_size=BLOCK)
        assert rows == [0]
        m.nums_dict[adapter.CHECKPOINT_KEY] = adapter.Checkpoints(
            _tensor([1], torch.int64), _tensor(hi, torch.int64),
            tuple(ends), tuple(positions))
    return m


def _precopy(caches, block_table, prefix, length, dim_first):
    """Execute the same GPU state-advance and state-copy kernels as V2."""
    from vllm.v1.worker.mamba_utils import (
        precopy_mamba_align_fused_kernel, preprocess_mamba_align_fused_kernel,
    )

    conv, recurrent = caches
    state_idx = _tensor([(prefix - 1) // BLOCK])
    src_col, bias, mapping = _tensor([-1]), _tensor([0]), _tensor([0])
    accepted = _tensor([1])
    preprocess_mamba_align_fused_kernel[(1,)](
        mapping, state_idx, _tensor([prefix]), _tensor([0, length - prefix]),
        accepted, src_col, bias, 1, BLOCK_SIZE=256, MAMBA_BLOCK_SIZE=BLOCK)
    assert state_idx.item() == math.ceil(length / BLOCK) - 1
    precopy_mamba_align_fused_kernel[(1, 2, 1)](
        state_idx, src_col, bias,
        _tensor([block_table.data_ptr()], torch.int64), block_table.stride(0),
        _tensor([s.data_ptr() for s in caches], torch.int64),
        _tensor([s.stride(0) * s.element_size() for s in caches], torch.int64),
        _tensor([s.element_size() for s in caches]),
        _tensor([1 if dim_first else conv.stride(1), recurrent[0].numel()]),
        _tensor([WINDOW, 0]), _tensor([0, 0]),
        _tensor([conv.shape[1] if dim_first else 0, 0]),
        _tensor([conv.stride(1) * conv.element_size() if dim_first else 0, 0],
                torch.int64), mapping, 1, COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=dim_first, HAS_IDX_MAPPING=True, TEMPORAL_TILES=1)


def _slice(inputs, begin, end):
    return {name: (value[begin:end] if name in ("mixed_qkv", "g2")
                   else value[:, begin:end]).clone()
            for name, value in inputs.items()}


def _difference(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return dict(max_abs=delta.max().item(), rms=delta.square().mean().sqrt().item(),
                exact=torch.equal(actual, expected))


@pytest.mark.parametrize("length", [4096, 8192, 8175])
@pytest.mark.parametrize("padded", [False, True], ids=["dense", "padded"])
@pytest.mark.parametrize("conv_backend", ["1", "0"], ids=["hcu_conv", "triton_conv"])
@pytest.mark.parametrize("gate_bias", [0.0, -6.0], ids=["short_memory", "long_memory"])
@torch.inference_mode()
def test_deployed_layer_prefix_replay(monkeypatch, length, padded, conv_backend, gate_bias):
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    import vllm.models.kimi_k3.amd.kda as target
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm_hcu.patch.worker.op_opt.patch_kimi_k3_kda_reference_paths import (
        apply_to_module,
    )

    for name, value in {
        "VLLM_KIMI_KDA_PREFILL_BACKEND": "boltops",
        "VLLM_KIMI_KDA_CACHE_STRATEGY": "extra_buffer",
        "VLLM_HCU_USE_CUSTOM_OPS": "1",
        "VLLM_KIMI_CONV1D_BACKEND": conv_backend,
    }.items():
        monkeypatch.setenv(name, value)
    apply_to_module(target)
    context = SimpleNamespace(attn_metadata={})
    monkeypatch.setattr(target, "get_forward_context", lambda: context)
    torch.manual_seed(20261026 + length)
    dim_first = is_conv_state_dim_first()
    channels = 3 * HEADS * DIM
    norm = SimpleNamespace(weight=torch.ones(DIM, device="cuda", dtype=torch.bfloat16),
                           bias=None, eps=1e-6, activation="sigmoid")
    # Use the actual gated norm kernel without a compiled model/config object.
    common = dict(
        prefix="offline.kda", head_dim=DIM, local_num_heads=HEADS,
        local_projection_size=HEADS * DIM, gate_lower_bound=-5.0,
        decode_conv1d_weight=None, decode_norm_weight=None,
        conv1d=SimpleNamespace(weight=torch.randn(channels, 1, WINDOW + 1,
                              device="cuda", dtype=torch.bfloat16) * 0.1, bias=None),
        A_log=torch.randn(HEADS, device="cuda") * 0.1,
        dt_bias=torch.randn(HEADS * DIM, device="cuda") * 0.1 + gate_bias,
        o_norm=lambda x, g: target.FusedRMSNormGated.forward_cuda(norm, x, g),
    )
    probe = dict(
        mixed_qkv=torch.randn(length, channels, device="cuda", dtype=torch.bfloat16),
        g1=torch.randn(1, length, HEADS, DIM, device="cuda", dtype=torch.bfloat16),
        g2=torch.randn(length, HEADS, DIM, device="cuda", dtype=torch.bfloat16),
        beta=torch.randn(1, length, HEADS, device="cuda", dtype=torch.bfloat16),
    )
    prefix = (length - 1) // BLOCK * BLOCK
    producer = _slice(probe, 0, length)
    for name, value in producer.items():
        tail = value[prefix:] if name in ("mixed_qkv", "g2") else value[:, prefix:]
        tail.copy_(torch.randn_like(tail))

    def forward(inputs, caches, metadata):
        context.attn_metadata = {common["prefix"]: metadata}
        layer = SimpleNamespace(**common, kv_cache=caches)
        output = torch.empty(1, inputs["mixed_qkv"].shape[0], HEADS, DIM,
                             device="cuda", dtype=torch.bfloat16)
        target.KimiK3DeltaAttention._forward(layer, **inputs, core_attn_out=output)
        return output

    name = f"{length}_{'padded' if padded else 'dense'}_{conv_backend}_{gate_bias:g}"
    output_dir = os.environ.get("KDA_REPRO_DIR")
    profiler = nullcontext()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_stack=True)
    report = dict(length=length, prefix=prefix, conv_backend=conv_backend,
                  padded=padded, gate_bias=gate_bias,
                  conv_dtype="bfloat16", recurrent_dtype="float32",
                  conv_layout="DS" if dim_first else "SD",
                  repeated_consumers=[], failures=[])
    failures = report["failures"]

    def check_close(actual, expected, label, exact=False):
        stats = _difference(actual, expected)
        report[label] = stats
        try:
            torch.testing.assert_close(actual.float(), expected.float(),
                                       atol=0 if exact else 0.02,
                                       rtol=0 if exact else 0.03)
        except AssertionError:
            failures.append(label)

    with profiler:
        with torch.profiler.record_function("cold_probe_full"):
            cold_cache = _pool(padded, dim_first)
            cold = forward(_slice(probe, 0, length), cold_cache,
                           _metadata(length, 0, 2))
            cold_final = cold_cache[1][2].clone()
        with torch.profiler.record_function("prefix_reference"):
            prefix_cache = _pool(padded, dim_first)
            forward(_slice(probe, 0, prefix), prefix_cache,
                    _metadata(prefix, 0, 2))
        with torch.profiler.record_function("producer_full_export_checkpoint"):
            caches = _pool(padded, dim_first)
            forward(producer, caches, _metadata(length, 0, 2, checkpoint=True))
            snapshots = [state[1].clone() for state in caches]
            for i, label in enumerate(("conv_checkpoint", "ssm_checkpoint")):
                check_close(snapshots[i], prefix_cache[i][2], label, exact=i == 0)
        checkpoint_col, active_col = prefix // BLOCK - 1, math.ceil(length / BLOCK) - 1
        previous = None
        for repeat in range(3):
            with torch.profiler.record_function(f"cached_probe_repeat_{repeat}"):
                destination = 3 + repeat
                table = _tensor([[0] * (active_col + 1)])
                table[0, checkpoint_col] = 1
                table[0, active_col] = destination
                _precopy(caches, table, prefix, length, dim_first)
                for i in range(2):
                    check_close(caches[i][destination], snapshots[i],
                                f"precopy_{repeat}_{i}", exact=True)
                resumed = forward(_slice(probe, prefix, length), caches,
                                  _metadata(length - prefix, prefix, destination))
                check_close(resumed, cold[:, prefix:], f"output_{repeat}")
                check_close(caches[1][destination], cold_final, f"final_{repeat}")
                for i in range(2):
                    check_close(caches[i][1], snapshots[i],
                                f"immutable_checkpoint_{repeat}_{i}", exact=True)
                if previous is not None:
                    check_close(resumed, previous, f"repeat_exact_{repeat}", exact=True)
                previous = resumed.clone()
                delta = (resumed.float() - cold[:, prefix:].float()).abs()
                differs = (resumed != cold[:, prefix:]).any(dim=-1).any(dim=-1)[0]
                different_positions = torch.nonzero(differs).flatten()
                report["repeated_consumers"].append(dict(
                    repeat=repeat,
                    first_resumed_position_one_based=prefix + 1,
                    first_different_position_one_based=(
                        prefix + 1 + different_positions[0].item()
                        if different_positions.numel() else None),
                    first_position_max_abs=delta[:, 0].max().item(),
                    last_position_max_abs=delta[:, -1].max().item(),
                    output=_difference(resumed, cold[:, prefix:]),
                ))
        with torch.profiler.record_function("control_restore_fp32_prefix_final"):
            # The exported per-chunk h is BF16 even when the committed state
            # pool is FP32. A truncated-prefix final state retains FP32.
            # Keep conv, inputs, copies and kernels unchanged for this control.
            caches[1][1].copy_(prefix_cache[1][2])
            table[0, active_col] = 6
            _precopy(caches, table, prefix, length, dim_first)
            control = forward(_slice(probe, prefix, length), caches,
                              _metadata(length - prefix, prefix, 6))
            check_close(control, cold[:, prefix:],
                        "fp32_reference_checkpoint_output", exact=True)
            check_close(caches[1][6], cold_final,
                        "fp32_reference_checkpoint_final", exact=True)
    if output_dir:
        path = Path(output_dir)
        profiler.export_chrome_trace(str(path / f"{name}.trace.json.gz"))
        (path / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    assert not failures, f"offline KDA replay failed: {failures}"


@torch.inference_mode()
def test_align_baseline_vs_extra_buffer(monkeypatch):
    """Compare the pre-optimization split forward with one-pass checkpointing.

    The baseline processes the prefix and suffix in two forwards, as the old
    align scheduler did. The optimized path processes the full prompt once,
    exports its internal checkpoint, restores it, and replays the suffix. Since
    both paths use the same kernels and checkpoint boundary, module outputs and
    saved/restored states are expected to match exactly. Reports are written
    before assertions so a precision regression still leaves a profiler trace.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    import vllm.models.kimi_k3.amd.kda as target
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm_hcu.patch.worker.op_opt.patch_kimi_k3_kda_reference_paths import (
        apply_to_module,
    )

    length, gate_bias, conv_backend, padded = 4096, -6.0, "1", True
    prefix = (length - 1) // BLOCK * BLOCK
    for name, value in {
        "VLLM_KIMI_KDA_PREFILL_BACKEND": "boltops",
        "VLLM_KIMI_KDA_CACHE_STRATEGY": "extra_buffer",
        "VLLM_HCU_USE_CUSTOM_OPS": "1",
        "VLLM_KIMI_CONV1D_BACKEND": conv_backend,
    }.items():
        monkeypatch.setenv(name, value)
    apply_to_module(target)
    context = SimpleNamespace(attn_metadata={})
    monkeypatch.setattr(target, "get_forward_context", lambda: context)
    torch.manual_seed(20261027 + length)
    dim_first = is_conv_state_dim_first()
    channels = 3 * HEADS * DIM
    norm = SimpleNamespace(
        weight=torch.ones(DIM, device="cuda", dtype=torch.bfloat16),
        bias=None,
        eps=1e-6,
        activation="sigmoid",
    )
    common = dict(
        prefix="offline.align_vs_extra_buffer",
        head_dim=DIM,
        local_num_heads=HEADS,
        local_projection_size=HEADS * DIM,
        gate_lower_bound=-5.0,
        decode_conv1d_weight=None,
        decode_norm_weight=None,
        conv1d=SimpleNamespace(
            weight=torch.randn(channels, 1, WINDOW + 1, device="cuda",
                               dtype=torch.bfloat16) * 0.1,
            bias=None,
        ),
        A_log=torch.randn(HEADS, device="cuda") * 0.1,
        dt_bias=torch.randn(HEADS * DIM, device="cuda") * 0.1 + gate_bias,
        o_norm=lambda x, g: target.FusedRMSNormGated.forward_cuda(norm, x, g),
    )
    inputs = dict(
        mixed_qkv=torch.randn(length, channels, device="cuda",
                              dtype=torch.bfloat16),
        g1=torch.randn(1, length, HEADS, DIM, device="cuda",
                       dtype=torch.bfloat16),
        g2=torch.randn(length, HEADS, DIM, device="cuda",
                       dtype=torch.bfloat16),
        beta=torch.randn(1, length, HEADS, device="cuda",
                         dtype=torch.bfloat16),
    )

    def forward(tokens, caches, metadata):
        context.attn_metadata = {common["prefix"]: metadata}
        layer = SimpleNamespace(**common, kv_cache=caches)
        output = torch.empty(1, tokens["mixed_qkv"].shape[0], HEADS, DIM,
                             device="cuda", dtype=torch.bfloat16)
        target.KimiK3DeltaAttention._forward(
            layer, **tokens, core_attn_out=output
        )
        return output

    report = {
        "test": "align_baseline_vs_extra_buffer",
        "length": length,
        "prefix": prefix,
        "suffix": length - prefix,
        "cache_dtype": "bfloat16_conv_float32_recurrent",
        "checkpoint_h_dtype": "float32",
        "conv_backend": conv_backend,
        "padded_pool": padded,
        "comparisons": {},
    }

    def compare(actual, expected):
        return _difference(actual, expected)

    output_dir = os.environ.get("KDA_REPRO_DIR")
    profiler = nullcontext()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        )

    with profiler:
        # Old align behavior: stop at the cache boundary, then continue from
        # that same active recurrent/conv state.
        with torch.profiler.record_function("baseline_align_prefix_forward"):
            baseline_cache = _pool(padded, dim_first)
            baseline_prefix_out = forward(
                _slice(inputs, 0, prefix), baseline_cache,
                _metadata(prefix, 0, 2),
            )
            baseline_checkpoint = [state[2].clone() for state in baseline_cache]
        with torch.profiler.record_function("baseline_align_suffix_forward"):
            baseline_suffix_out = forward(
                _slice(inputs, prefix, length), baseline_cache,
                _metadata(length - prefix, prefix, 2),
            )
            baseline_final = [state[2].clone() for state in baseline_cache]
        baseline_full_out = torch.cat(
            [baseline_prefix_out, baseline_suffix_out], dim=1
        )

        # Optimized behavior: one full model forward, with an internal
        # checkpoint exported at the same prefix boundary.
        with torch.profiler.record_function("optimized_single_full_forward"):
            optimized_cache = _pool(padded, dim_first)
            optimized_full_out = forward(
                _slice(inputs, 0, length), optimized_cache,
                _metadata(length, 0, 2, checkpoint=True),
            )
            optimized_checkpoint = [state[1].clone()
                                    for state in optimized_cache]
            optimized_full_final = [state[2].clone()
                                    for state in optimized_cache]

        report["comparisons"]["one_pass_output_vs_two_pass_output"] = compare(
            optimized_full_out, baseline_full_out
        )
        report["comparisons"]["checkpoint_conv_vs_align_boundary"] = compare(
            optimized_checkpoint[0], baseline_checkpoint[0]
        )
        report["comparisons"]["checkpoint_ssm_vs_align_boundary"] = compare(
            optimized_checkpoint[1], baseline_checkpoint[1]
        )
        report["comparisons"]["one_pass_final_conv_vs_two_pass_final"] = compare(
            optimized_full_final[0], baseline_final[0]
        )
        report["comparisons"]["one_pass_final_ssm_vs_two_pass_final"] = compare(
            optimized_full_final[1], baseline_final[1]
        )

        # Replay the optimized internal checkpoint through the same V2 state
        # copy path, then compare its suffix against old align continuation.
        checkpoint_col = prefix // BLOCK - 1
        active_col = math.ceil(length / BLOCK) - 1
        destination = 3
        table = _tensor([[0] * (active_col + 1)])
        table[0, checkpoint_col] = 1
        table[0, active_col] = destination
        with torch.profiler.record_function("optimized_checkpoint_restore"):
            _precopy(optimized_cache, table, prefix, length, dim_first)
        with torch.profiler.record_function("optimized_cached_suffix_forward"):
            optimized_suffix_out = forward(
                _slice(inputs, prefix, length), optimized_cache,
                _metadata(length - prefix, prefix, destination),
            )
        report["comparisons"]["cached_suffix_vs_align_suffix"] = compare(
            optimized_suffix_out, baseline_suffix_out
        )
        report["comparisons"]["cached_final_conv_vs_align_final"] = compare(
            optimized_cache[0][destination], baseline_final[0]
        )
        report["comparisons"]["cached_final_ssm_vs_align_final"] = compare(
            optimized_cache[1][destination], baseline_final[1]
        )

    if output_dir:
        path = Path(output_dir)
        profiler.export_chrome_trace(
            str(path / "align_baseline_vs_extra_buffer.trace.json.gz")
        )
        (path / "align_baseline_vs_extra_buffer.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    required_exact = tuple(report["comparisons"])
    alignment_failures = [
        name for name in required_exact
        if not report["comparisons"][name]["exact"]
    ]
    report["alignment_pass"] = not alignment_failures
    report["alignment_failures"] = alignment_failures
    if output_dir:
        (Path(output_dir) / "align_baseline_vs_extra_buffer.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(json.dumps(report, sort_keys=True))
    assert not alignment_failures, (
        "optimized module path does not exactly align with the pre-optimization "
        f"align baseline: {alignment_failures}"
    )
