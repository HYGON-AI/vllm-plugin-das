#!/usr/bin/env python3
"""Fixed CPU E2E replay: real config/RoutedExperts/loaders/method/modular glue.

Only hardware admission, EP transport/events, FP8 quantization, physical packing,
scatter/gather kernels and DeepGEMM/LightOp kernels are CPU doubles. No oracle,
factory, global distributed group or method/prepare/expert forward is replaced.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import torch
from hcu_vllm_das.moe import channel_fp8_prepare as prepare
from hcu_vllm_das.moe import channel_fp8_runtime as runtime
from hcu_vllm_das.quantization import dsv41_channel_fp8 as config_module
from hcu_vllm_das.quantization.channel_fp8_moe import HcuDSV41ChannelFP8MoEMethod
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe import all2all_utils
from vllm.model_executor.layers.fused_moe.prepare_finalize import deepep_ht, deepep_ll
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

RESULTS, CALLS, PACKED = [], [], {}
PACK_RANK = 6


def case(name, callback):
    try:
        detail = callback()
        RESULTS.append({"case": name, "status": "passed", "detail": detail})
    except Exception as error:
        traceback.print_exc()
        RESULTS.append({"case": name, "status": "failed", "error": f"{type(error).__name__}: {error}"})


def expect_error(callback, text=None):
    try:
        callback()
    except (AssertionError, RuntimeError, ValueError, TypeError) as error:
        if text:
            assert text.lower() in str(error).lower(), str(error)
        return
    raise AssertionError("invalid Channel-FP8 contract accepted")


def schema():
    return {"quant_method": "fp8", "is_per_channel": True, "weight_block_size": None, "activation_scheme": "dynamic"}


def quantize(values, *args, **kwargs):
    if "per_act_token_quant" in kwargs:
        assert kwargs["per_act_token_quant"] is True and kwargs.get("block_shape") is None
    values = values.float()
    scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
    return (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def fake_pack(weight):
    CALLS.append("pack")
    e, n, k = weight.shape
    shape = (e, k // 64, n // 16, 4, 16, 16) if PACK_RANK == 6 else (e, k // 64, n // 32, 2, 4, 16, 16)
    result = weight.clone().view(shape)
    PACKED[result.data_ptr()] = weight.float().clone()
    return result


def gemm_contiguous(a, b, out, expert_ids):
    CALLS.append("ht_gemm")
    q, scale = a
    w, ws = b
    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert tuple(scale.shape) == (q.shape[0], 1) and scale.is_contiguous()
    assert expert_ids.dtype == torch.int32 and expert_ids.shape == (q.shape[0],)
    raw = PACKED[w.data_ptr()]
    assert tuple(ws.shape) == tuple(raw.shape[:2])
    out.zero_()
    for e in range(raw.shape[0]):
        selected = torch.where(expert_ids == e)[0]
        if selected.numel():
            out[selected] = ((q.float()[selected] * scale[selected]) @ (raw[e] * ws[e, :, None]).T).to(out.dtype)


def gemm_masked(a, b, out, counts, expected_m):
    CALLS.append("ll_gemm")
    q, scale = a
    w, ws = b
    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert tuple(scale.shape) == tuple(q.shape[:2]) and scale.is_contiguous()
    assert counts.dtype == torch.int32 and isinstance(expected_m, int) and expected_m > 0
    raw = PACKED[w.data_ptr()]
    assert tuple(ws.shape) == tuple(raw.shape[:2]) and ws.is_contiguous()
    out.zero_()
    for e, count in enumerate(counts.tolist()):
        if count:
            out[e, :count] = ((q[e, :count].float() * scale[e, :count, None]) @ (raw[e] * ws[e, :, None]).T).to(out.dtype)


def activation(values, *, fp8type, expert_ids=None, tokens_per_expert=None, limit=None):
    assert fp8type == 0
    CALLS.append("ht_activation" if expert_ids is not None else "ll_activation")
    gate, up = values.float().chunk(2, dim=-1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    result = torch.nn.functional.silu(gate) * up
    if expert_ids is not None:
        result[expert_ids < 0] = 0
    else:
        assert tokens_per_expert.dtype == torch.int32
        for e, count in enumerate(tokens_per_expert.tolist()):
            result[e, count:] = 0
    return quantize(result)


def scatter(**kw):
    CALLS.append("ep_scatter")
    assert kw["align_m"] == 256 and kw["block_size"] == kw["recv_x"].shape[1]
    ids, offsets = kw["recv_topk"], []
    start = 0
    for count in kw["num_recv_tokens_per_expert"].tolist():
        offsets.append(start)
        start += (count + 255) // 256 * 256
    kw["output_tensor"].zero_()
    used = [0] * len(offsets)
    for row in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            e = int(ids[row, slot])
            if kw["expert_map"] is not None:
                e = int(kw["expert_map"][e])
            if e < 0:
                kw["output_index"][row, slot] = -1
                continue
            dest = offsets[e] + used[e]
            used[e] += 1
            kw["output_tensor"][dest].copy_(kw["recv_x"][row])
            kw["output_tensor_scale"][dest].copy_(kw["recv_x_scale"][row])
            kw["m_indices"][dest] = e
            kw["output_index"][row, slot] = dest


def gather(**kw):
    CALLS.append("ep_gather")
    out = kw["output_tensor"]
    result = torch.zeros_like(out, dtype=torch.float32)
    for row in range(out.shape[0]):
        for slot in range(kw["input_index"].shape[1]):
            source = int(kw["input_index"][row, slot])
            if source >= 0:
                result[row] += kw["input_tensor"][source].float() * kw["recv_topk_weight"][row, slot]
    out.copy_(result)


EVENT = SimpleNamespace(event=None, current_stream_wait=lambda: None)


class Buffer:
    def capture(self):
        return EVENT

    def get_dispatch_layout(self, **kw):
        ids = kw["topk_idx"]
        counts = torch.bincount(ids.flatten(), minlength=kw["num_experts"]).to(torch.int32)
        return None, None, counts, None, EVENT

    def dispatch(self, **kw):
        CALLS.append("ht_dispatch")
        ids = kw["topk_idx"]
        counts = torch.bincount(ids.flatten(), minlength=2).tolist()
        return kw["x"], ids, kw["topk_weights"], counts, ("ht",), EVENT

    def combine(self, **kw):
        CALLS.append("ht_combine")
        return kw["x"], None, EVENT

    def low_latency_dispatch(self, x, ids, topk_weight, max_tokens, num_experts, *, quant_type, quant_group_size=0, fp8_round_scale=False, **kw):
        CALLS.append("ll_dispatch")
        assert quant_type == 2
        assert quant_group_size == 0 and fp8_round_scale is False
        capacity = max_tokens * 2
        local_experts = num_experts // 2
        values = torch.zeros((local_experts, capacity, x.shape[-1]), dtype=x.dtype)
        counts = torch.zeros(local_experts, dtype=torch.int32)
        for row in range(x.shape[0]):
            for slot in range(ids.shape[1]):
                e = int(ids[row, slot])
                values[e, counts[e]].copy_(x[row])
                counts[e] += 1
        q, scales = quantize(values)
        return (q, scales.squeeze(-1).contiguous()), counts, (ids.clone(),), None, lambda: CALLS.append("ll_recv_hook")

    def low_latency_combine(self, x, ids, topk_weights, handle, out=None, **kw):
        weights = topk_weights
        CALLS.append("ll_combine")
        result = torch.zeros((ids.shape[0], x.shape[-1]), dtype=torch.float32)
        offsets = [0] * x.shape[0]
        for row in range(ids.shape[0]):
            for slot in range(ids.shape[1]):
                e = int(ids[row, slot])
                result[row] += x[e, offsets[e]].float() * weights[row, slot]
                offsets[e] += 1
        if out is not None:
            out.copy_(result)
            result = out
        return result.to(x.dtype), None, lambda: CALLS.append("ll_combine_hook")


def make_parallel(mode):
    return FusedMoEParallelConfig(tp_size=1, pcp_size=1, dp_size=2, ep_size=2,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1,
        use_ep=True, all2all_backend="deepep_high_throughput" if mode == "ht" else "deepep_low_latency", enable_eplb=False)


def make_layer(mode, n=64, k=None):
    k = k or (256 if mode == "ht" else 2048)
    parallel = make_parallel(mode)
    moe = FusedMoEConfig(num_experts=4, experts_per_token=2, hidden_dim=k,
        intermediate_size=n, num_local_experts=2, num_logical_experts=4,
        activation=MoEActivation.SILU, device="cpu", routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel, in_dtype=torch.bfloat16, moe_backend="deep_gemm", max_num_tokens=257)
    manager = ExpertMapManager(257, 2, 4, 0, None, parallel, "linear", False)
    cfg = config_module.HcuDSV41ChannelFP8Config.from_config(schema())
    layer = RoutedExperts("model.layers.0.mlp.experts", torch.bfloat16, moe, cfg, manager)
    assert type(layer.quant_method) is HcuDSV41ChannelFP8MoEMethod
    assert layer.w13_weight.weight_loader.__self__ is layer
    return layer


def weights(seed, n, k):
    g = torch.Generator().manual_seed(seed)
    w1 = (torch.randn((2, 2*n, k), generator=g) * 0.3).to(torch.float8_e4m3fn)
    w2 = (torch.randn((2, k, n), generator=g) * 0.3).to(torch.float8_e4m3fn)
    s1 = torch.rand((2, 2*n, 1), generator=g) * 0.03 + 0.01
    s2 = torch.rand((2, k, 1), generator=g) * 0.03 + 0.01
    return w1, w2, s1, s2


def load(layer, tensors, merged=False):
    w1, w2, s1, s2 = tensors
    n = w2.shape[-1]
    for e in range(2):
        for shard, weight, scale in (("w1", w1[e, :n], s1[e, :n]), ("w3", w1[e, n:], s1[e, n:]), ("w2", w2[e], s2[e])):
            stem = "w2" if shard == "w2" else "w13"
            for suffix, tensor in (("weight", weight), ("weight_scale", scale)):
                param = getattr(layer, f"{stem}_{suffix}")
                assert param.weight_loader(param, tensor, f"{stem}_{suffix}", shard, e, True)
    for actual, expected in zip((layer.w13_weight, layer.w2_weight, layer.w13_weight_scale, layer.w2_weight_scale), tensors):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def reference(x, ids, routing, tensors, limit=None):
    # Independent scalar-token/expert reference, including both FP8 roundings
    # and BF16 GEMM output rounding. No production helper/provider calls.
    w1, w2, s1, s2 = tensors
    result = torch.zeros_like(x, dtype=torch.float32)
    for row in range(x.shape[0]):
        value = x[row].float()
        scale = value.abs().max().clamp_min(1e-12) / 448
        value = (value / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
        for slot in range(ids.shape[1]):
            e = int(ids[row, slot])
            projected = (value @ (w1[e].float() * s1[e]).T).to(torch.bfloat16).float()
            gate, up = projected.chunk(2)
            if limit is not None:
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
            activated = gate / (1 + torch.exp(-gate)) * up
            scale2 = activated.abs().max().clamp_min(1e-12) / 448
            activated = (activated / scale2).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale2
            down = (activated @ (w2[e].float() * s2[e]).T).to(torch.bfloat16).float()
            result[row] += down * routing[row, slot]
    return result.to(x.dtype)


def forward_case(mode, rows, rank=6, n=64):
    global PACK_RANK
    PACK_RANK = rank
    layer = make_layer(mode, n=n)
    tensors = weights(21 + rows, n, layer.hidden_size)
    load(layer, tensors)
    before = CALLS.count("pack")
    layer.quant_method.process_weights_after_loading(layer)
    layer.quant_method.process_weights_after_loading(layer)
    assert CALLS.count("pack") - before == 2
    g = torch.Generator().manual_seed(900 + rows)
    x = (torch.randn((rows, layer.hidden_size), generator=g) * 0.2).to(torch.bfloat16)
    ids = torch.tensor([[0, 1] if row % 3 else [1, 0] for row in range(rows)], dtype=torch.int64).view(rows, 2)
    routing = torch.tensor([[0.3, 0.7]] * rows, dtype=torch.float32).view(rows, 2)
    start = len(CALLS)
    actual = layer.forward_modular(x, routing, ids)
    expected = reference(x, ids, routing, tensors)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=1e-7)
    if rows:
        peak = expected.float().abs().max()
        assert peak > 1e-6, "reference amplitude cannot reject a zero provider"
        assert (actual.float() - expected.float()).abs().max() / peak < 0.02
    calls = CALLS[start:]
    assert calls.count(mode + "_dispatch") == calls.count(mode + "_combine") == 1
    if rows or mode == "ll":
        assert calls.count(mode + "_gemm") == 2 and calls.count(mode + "_activation") == 1
    else:
        assert not any("gemm" in entry for entry in calls)
    return {"mode": mode, "rows": rows, "physical_rank": rank, "shape": list(actual.shape), "max_abs_error": float((actual.float() - expected.float()).abs().max()) if rows else 0, "calls": calls}


def contract_cases():
    global PACK_RANK
    for seed in range(200):
        PACK_RANK = 6 + seed % 2
        tensors = weights(seed, 64 if seed % 3 else 128, 256)
        first = runtime.pack_channel_weights(*tensors, layout="contiguous")
        assert runtime.pack_channel_weights(first.w1, first.w2, first.s1, first.s2, layout="contiguous", prior=first) is first
        expect_error(lambda: runtime.pack_channel_weights(first.w1, first.w2, first.s1, first.s2, layout="masked", prior=first), "layout")
        expect_error(lambda: first.validate(first.w1.clone(), first.w2, first.s1, first.s2), "witness")
        expect_error(lambda: first.validate(first.w1, first.w2, first.s1.clone(), first.s2), "witness")
    return {"seeds": 200, "physical_ranks": [6, 7]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    provider = ModuleType("deepgemm")
    provider.marlin_fp8_contiguous_weight = provider.marlin_fp8_masked_weight = fake_pack
    provider.m_grouped_fp8_gemm_nt_contiguous, provider.m_grouped_fp8_gemm_nt_masked = gemm_contiguous, gemm_masked
    lightop = ModuleType("lightop")
    lightop_activation = ModuleType("lightop.activation")
    lightop_activation.fuse_silu_mul_fp8_quant = lightop_activation.fuse_silu_mul_fp8_quant_ep = activation
    buffer = Buffer()
    manager = SimpleNamespace(world_size=2, dp_world_size=2, rank=0, get_handle=lambda args: buffer)
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {"deepgemm": provider, "lightop": lightop, "lightop.activation": lightop_activation}))
        for module, name, value in ((config_module, "_is_hcu", lambda: True), (config_module, "check_health", lambda: None),
                (all2all_utils, "DeepEPHTPrepareAndFinalize", deepep_ht.DeepEPHTPrepareAndFinalize),
                (all2all_utils, "DeepEPLLPrepareAndFinalize", deepep_ll.DeepEPLLPrepareAndFinalize),
                (all2all_utils, "DEEPEP_QUANT_BLOCK_SHAPE", deepep_ll.DEEPEP_QUANT_BLOCK_SHAPE),
                (runtime, "ep_scatter", scatter), (runtime, "ep_gather", gather),
                (prepare, "get_ep_all2all_manager", lambda *args: manager),
                (prepare, "moe_kernel_quantize_input", quantize)):
            stack.enter_context(patch.object(module, name, value, create=True))
        # DBO/event functions touch GPU streams even without DBO; these are the
        # transport boundary. Receivers, factories and all forward glue stay real.
        for module in (deepep_ht, deepep_ll):
            for name in list(vars(module)):
                if name.startswith("dbo_") or name == "_dbo_record_stream":
                    value = (lambda: 0) if name == "dbo_current_ubatch_id" else ((lambda: False) if name == "dbo_enabled" else (lambda *a, **k: None))
                    stack.enter_context(patch.object(module, name, value))
        case("200_seed_physical_layout_and_full_witness", contract_cases)
        for mode, rows, rank, n in (("ht", 0, 6, 64), ("ht", 1, 7, 128), ("ht", 255, 6, 64), ("ht", 256, 7, 64), ("ht", 257, 6, 64), ("ll", 0, 6, 64), ("ll", 1, 7, 128), ("ll", 16, 6, 64)):
            case(f"public_loader_modular_{mode}_{rows}_rank{rank}", lambda m=mode, r=rows, p=rank, n=n: forward_case(m, r, p, n))
    report = {"level": "CPU numerical E2E with explicit hardware/provider doubles; no GPU/DeepEP accuracy acceptance",
        "doubles": ["platform admission", "EP manager/transport/events", "token quantization kernel", "ep_scatter/ep_gather kernels", "DeepGEMM packing/GEMM", "LightOp activation"],
        "real_chain": ["production imports", "config", "public RoutedExperts construction", "native CT create_weights", "public parameter loader", "owned method", "mk.FusedMoEKernel", "owned prepare receiver", "owned experts", "native finalize", "forward_modular"],
        "cases": RESULTS, "passed": sum(c["status"] == "passed" for c in RESULTS), "total": len(RESULTS), "cuda_initialized": torch.cuda.is_initialized()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
