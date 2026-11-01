"""Bounded source-stage QNT009 native/official INT8 differential microprobe.

Run only with GPU2 exclusively allocated. A native failure ends this process;
the pytest parent must not retry a VMFault or proceed to another GPU case.
"""

# Explicit source phase and device/core limits must precede runtime imports.
# ruff: noqa: E402

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import resource
import statistics
import sys
from dataclasses import replace
from pathlib import Path

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
assert os.environ.get("HCU_ADAPTER_HOST") == "none"
assert not os.environ.get("PYTHONPATH") and not os.environ.get("ROCR_VISIBLE_DEVICES")
assert (
    os.environ.get("CUDA_VISIBLE_DEVICES")
    == os.environ.get("HIP_VISIBLE_DEVICES")
    == "2"
)
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(os.environ["QNT009_EVIDENCE_ROOT"])
assert EVIDENCE.is_relative_to("/dev/shm")
sys.path.insert(0, str(ROOT / "adapter_extension/src"))

import torch

torch.set_num_threads(2)
assert torch.version.hip and torch.cuda.device_count() == 1
torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
torch.cuda.set_per_process_memory_fraction(3.5 * 1024**3 / props.total_memory)
hip = ctypes.CDLL("libamdhip64.so")
pci = ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci, 32, 0) == 0
assert (
    pci.value.decode().lower()
    == Path("/sys/class/drm/renderD130/device").resolve().name.lower()
)

import vllm
from hcu_vllm_das.w4a8 import config as api
from hcu_vllm_das.w4a8 import lifecycle, runtime
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.v1.worker.workspace import init_workspace_manager

init_workspace_manager(torch.device("cuda:0"))


def emit(event, **values):
    row = dict(event=event, **values)
    line = json.dumps(row, sort_keys=True, default=str)
    print(line, flush=True)
    with (EVIDENCE / "records.jsonl").open("a") as stream:
        stream.write(line + "\n")


def make_moe(*, ep=False):
    parallel = replace(FusedMoEParallelConfig.make_no_parallel(), tp_size=4)
    return FusedMoEConfig(
        num_experts=512,
        experts_per_token=10,
        hidden_dim=2560,
        intermediate_size=640,
        num_local_experts=512,
        num_logical_experts=512,
        activation=MoEActivation.SILU,
        device="cuda:0",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        max_num_tokens=512,
    )


def make_layer(moe, method):
    from types import SimpleNamespace

    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = moe
    layer.quant_method = method
    layer.expert_map_manager = SimpleNamespace(expert_map=None)
    layer.register_buffer("_expert_map", None, persistent=False)
    layer.global_num_experts = 512
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    h, i = method.maybe_roundup_sizes(
        2560, 160, torch.bfloat16, moe.moe_parallel_config
    )
    moe.hidden_dim, moe.intermediate_size_per_partition = h, i
    with torch.device("cuda:0"):
        method.create_weights(layer, 512, h, i, torch.bfloat16)
    return layer


def install_weights(layer, raw):
    # Mirrors the already CPU-validated official loader's final rank-local
    # slices. The HCU micro allocates no full checkpoint or distributed ranks.
    i = layer.moe_config.intermediate_size_per_partition
    w13, w2, s13, s2 = raw
    layer.w13_weight[:, :160, :1280].copy_(w13[:, :160])
    layer.w13_weight[:, i : i + 160, :1280].copy_(w13[:, 160:])
    layer.w2_weight[:, :2560, :80].copy_(w2)
    layer.w13_weight_scale[:, :160].copy_(s13[:, :160])
    layer.w13_weight_scale[:, i : i + 160].copy_(s13[:, 160:])
    layer.w2_weight_scale[:, :2560].copy_(s2)


def compare(a, b):
    a, b = a.float(), b.float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    diff = a - b
    nrmse = float(
        diff.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)
    )
    relative = float(diff.abs().max() / b.abs().max().clamp_min(1e-12))
    cosine = float(
        torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    )
    result = dict(
        nrmse=nrmse,
        relative_max_abs=relative,
        cosine=cosine,
        exact=torch.equal(a, b),
        max_abs=float(diff.abs().max()),
    )
    assert nrmse <= 0.035 and relative <= 0.10 and cosine >= 0.999, result
    return result


def cpu_reference(x, ids, routing, raw):
    # Independent FP32 INT8-dynamic oracle with the checkpoint's x16 scale.
    x = x.cpu().float()
    ids, routing = ids.cpu(), routing.cpu()
    result = torch.zeros_like(x)

    def quant(value):
        scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
        return (value / scale).round().clamp(-127, 127), scale

    q, sx = quant(x)
    for token in range(len(x)):
        for route in range(10):
            expert = int(ids[token, route])
            w13, w2 = (
                runtime.unpack(w[expert : expert + 1].cpu())[0].float() for w in raw[:2]
            )
            a = (
                (q[token : token + 1] @ w13.t())
                * sx[token]
                * raw[2][expert].cpu().t()
                * 16
            ).bfloat16()
            gate, up = a.chunk(2, -1)
            a = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16().float()
            aq, sa = quant(a)
            out = (aq @ w2.t()) * sa * raw[3][expert].cpu().t() * 16
            result[token] += out[0] * routing[token, route]
    return result.bfloat16().cuda()


emit(
    "identity",
    torch=torch.__version__,
    vllm=vllm.__version__,
    vllm_path=vllm.__file__,
    plugin_path=api.__file__,
    stage="source inserted explicitly; not integrated wheel",
    source_hashes={
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "adapter_extension/src/hcu_vllm_das/w4a8").glob("*.py")
    },
    pci=pci.value.decode(),
    arch=props.gcnArchName,
    allocator_cap_bytes=int(3.5 * 1024**3),
    core_limit=list(resource.getrlimit(resource.RLIMIT_CORE)),
)
assert vllm.__version__ == "0.32.0+das.a198423b7.dtk2610"
assert Path(vllm.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
target = (
    Path(
        os.environ.get(
            "VLLM_TARGET_ROOT", "/dev/shm/vllm032-upgrade-20261031/target-build/source"
        )
    )
    / "vllm"
)
for relative in (
    "model_executor/layers/fused_moe/routed_experts.py",
    "model_executor/layers/fused_moe/experts/triton_moe.py",
    "model_executor/layers/quantization/compressed_tensors/compressed_tensors.py",
    "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_int8.py",
    "models/qwen4_exp/common/ngram_embedding.py",
):
    expected = (target / relative).read_bytes()
    assert (Path(vllm.__file__).parent / relative).read_bytes() == expected, relative
    emit(
        "official_source_match",
        relative=relative,
        sha256=hashlib.sha256(expected).hexdigest(),
    )
torch.manual_seed(20261101)

config = VllmConfig(
    parallel_config=ParallelConfig(
        worker_cls="hcu_adapter.bridges.vllm_worker.Worker",
        distributed_executor_backend="uni",
    )
)
with set_current_vllm_config(config):
    from types import SimpleNamespace

    config.model_config = SimpleNamespace(quantization="slimquant_w4a8")
    lifecycle.install()
    moe = make_moe()
    plan = runtime.prepare_native(moe)
    emit(
        "native_preflight",
        logical_geometry=[512, 2560, 160, 10],
        padded_geometry=vars(plan.shape) if plan else None,
        all_m_count=len(plan.configs) if plan else 0,
        layout=plan.layout if plan else None,
        first_config=vars(plan.configs[0]) if plan else None,
    )
    assert plan is not None, (
        "Actual geometry has no complete native pre-pack route; not a native pass"
    )
    native_method = api.SlimQuantW4A8MoEMethod(copy.copy(moe), plan)
    fallback_method = api.SlimQuantW4A8MoEMethod(copy.copy(moe), None)
    native = make_layer(native_method.moe, native_method)
    fallback = make_layer(fallback_method.moe, fallback_method)
    raw = (
        torch.randint(-128, 128, (512, 320, 1280), device="cuda", dtype=torch.int8),
        torch.randint(-128, 128, (512, 2560, 80), device="cuda", dtype=torch.int8),
        torch.rand((512, 320, 1), device="cuda") * 0.0005 + 0.0005,
        torch.rand((512, 2560, 1), device="cuda") * 0.0005 + 0.0005,
    )
    install_weights(native, raw)
    install_weights(fallback, raw)
    native_method.process_weights_after_loading(native)
    fallback_method.process_weights_after_loading(fallback)
    emit(
        "prepared",
        native_shapes=[list(native.w13_weight.shape), list(native.w2_weight.shape)],
        fallback_shapes=[
            list(fallback.w13_weight.shape),
            list(fallback.w2_weight.shape),
        ],
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
    )
    rows = []
    cases = (512,) if os.environ.get("QNT009_MAX_BOUNDARY_ONLY") == "1" else (1, 17, 65)
    for m in cases:
        x = (torch.randn((m, 2560), device="cuda") * 0.25).bfloat16()
        ids = ((torch.arange(m * 10, device="cuda").reshape(m, 10) * 13) % 512).int()
        weights = torch.softmax(torch.randn((m, 10), device="cuda"), -1)

        def call(method, layer):
            value = (
                x
                if method.skip_forward_padding
                else torch.nn.functional.pad(
                    x, (0, layer.moe_config.hidden_dim - 2560)
                )
            )
            return method.apply(layer, value, weights, ids, None, None)[:, :2560]

        emit("native_enter", m=m, layout=plan.layout, config=vars(plan.configs[m - 1]))
        out_native = call(native_method, native)
        torch.cuda.synchronize()
        emit("native_exit", m=m)
        out_official = call(fallback_method, fallback)
        torch.cuda.synchronize()
        checks = [dict(stage="eager", **compare(out_native, out_official))]
        if m == 1:
            reference = cpu_reference(x, ids, weights, raw)
            checks += [
                dict(stage="native-vs-cpu-oracle", **compare(out_native, reference)),
                dict(
                    stage="official-vs-cpu-oracle", **compare(out_official, reference)
                ),
            ]
        # Warm up each exact route before capture; keep all addresses fixed.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                call(native_method, native)
                call(fallback_method, fallback)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_native = call(native_method, native)
            replay_official = call(fallback_method, fallback)
        pointers = (x.data_ptr(), ids.data_ptr(), weights.data_ptr())
        for wave in range(4):
            x.copy_((torch.randn_like(x.float()) * (0.1 + wave * 0.07)).bfloat16())
            ids.add_(37).remainder_(512)
            weights.copy_(torch.softmax(torch.randn_like(weights), -1))
            graph.replay()
            torch.cuda.synchronize()
            eager_native = call(native_method, native)
            eager_official = call(fallback_method, fallback)
            checks += [
                dict(
                    stage=f"graph-{wave}-native-exact",
                    exact=torch.equal(replay_native, eager_native),
                ),
                dict(
                    stage=f"graph-{wave}-official-exact",
                    exact=torch.equal(replay_official, eager_official),
                ),
                dict(
                    stage=f"graph-{wave}-pair",
                    **compare(replay_native, replay_official),
                ),
            ]
            assert torch.equal(replay_native, eager_native) and torch.equal(
                replay_official, eager_official
            )
            assert pointers == (x.data_ptr(), ids.data_ptr(), weights.data_ptr())
        row = dict(m=m, checks=checks)
        if os.environ.get("QNT009_EXTRA_EVIDENCE") == "1":
            calls = []

            def observe(frame, event, arg):
                if (
                    event == "c_call"
                    and getattr(arg, "__name__", "") == "moe_c_moe_gemm_marlin_w4a8"
                ):
                    calls.append(
                        dict(name=arg.__name__, module=getattr(arg, "__module__", None))
                    )

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                sys.setprofile(observe)
                try:
                    call(native_method, native)
                    torch.cuda.synchronize()
                finally:
                    sys.setprofile(None)
            kernels = sorted(
                {
                    event.name
                    for event in profile.events()
                    if str(event.device_type).endswith("CUDA")
                }
            )
            profile.export_chrome_trace(str(EVIDENCE / f"w4a8-m{m}-trace.json"))
            assert len(calls) >= 2, "No completed real W4A8 native GEMM binding calls"
            row["native_witness"] = dict(calls=calls, gpu_kernel_names=kernels)
            emit("native_witness", m=m, **row["native_witness"])
            graphs = {}
            for name, method, layer in (
                ("native", native_method, native),
                ("official", fallback_method, fallback),
            ):
                graphs[name] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[name]):
                    call(method, layer)
            timings = []
            for repeat in range(3):
                round_us = {}
                for name, perf_graph in graphs.items():
                    for _ in range(10):
                        perf_graph.replay()
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    for _ in range(100):
                        perf_graph.replay()
                    end.record()
                    end.synchronize()
                    round_us[name] = begin.elapsed_time(end) * 10
                timings.append(round_us)
            row["graph_perf_us"] = timings
            emit(
                "graph_perf",
                m=m,
                rounds=timings,
                median_us={
                    name: statistics.median(t[name] for t in timings) for name in graphs
                },
            )
        rows.append(row)
        emit("case_pass", **row)
    emit(
        "summary",
        cases=rows,
        max_allocated=torch.cuda.max_memory_allocated(),
        max_reserved=torch.cuda.max_memory_reserved(),
        native_solution=plan.layout[0],
        no_model_launched=True,
        ep="not enabled; separate gate",
        int8_ple="pending separate protocol",
    )
    (EVIDENCE / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
