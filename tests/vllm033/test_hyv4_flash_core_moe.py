# SPDX-License-Identifier: Apache-2.0
"""CPU composition tests, NOT native Core/GPU factory execution.

The actual pinned Core HYV4 MoE/FFN classes are source-loaded with test-local
public-module injection. Factory/linear/norm implementations are explicit CPU
spies; the factory signature is read from the actual Core AST, not invented.
Numeric tests use real Torch tensors, per-token CPU FP8 round trips on replicated
projections, and real Gloo all-reduces observed by a communication spy. This does
not validate native expert quantization, accelerator kernels, or model serving.

Set HYV4_CORE033_ROOT if the pinned Core checkout is not a sibling directory.
This file can run alone with pytest --confcutdir=<this file's directory>.
"""

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import distributed as dist, multiprocessing as mp, nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
CORE = Path(os.environ.get("HYV4_CORE033_ROOT") or os.environ["VLLM_TARGET_ROOT"])
ACTIVE = REPO / "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/core_moe.py"
CORE_MOE = CORE / "vllm/models/hy_v4/nvidia/moe.py"
FACTORY = CORE / "vllm/model_executor/layers/fused_moe/layer.py"


def _weight(name, shape):
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 0.6


def _dynamic_fp8(x):
    # A real CPU FP8 round trip, not Core's native quantization method.
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def _config(**overrides):
    values = dict(
        hidden_size=7,
        intermediate_size=8,
        expert_hidden_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        hidden_act="silu",
        route_norm=True,
        router_scaling_factor=1.7,
        routed_expert_hidden_size=5,
        sandwich_norm=False,
        rms_norm_eps=1e-6,
        swiglu_limit=0.45,
        flash_w4a8_backend="disabled",
    )
    return SimpleNamespace(**(values | overrides))


def _signature(path, function):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function
    )
    args = fn.args
    assert not args.vararg and not args.kwarg, "Core signature must be explicit"
    defaults = [inspect.Parameter.empty] * (len(args.args) - len(args.defaults))
    defaults += [ast.literal_eval(d) for d in args.defaults]
    return inspect.Signature(
        [
            inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=d)
            for a, d in zip(args.args, defaults)
        ]
        + [
            inspect.Parameter(
                a.arg,
                inspect.Parameter.KEYWORD_ONLY,
                default=(
                    ast.literal_eval(d) if d is not None else inspect.Parameter.empty
                ),
            )
            for a, d in zip(args.kwonlyargs, args.kw_defaults)
        ]
    )


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class _Harness:
    """Explicit dependency injection; no Core functions/globals are patched."""

    def __init__(self, monkeypatch, tp=1, rank=0, factory_error=False):
        self.tp, self.rank = tp, rank
        self.events, self.factory_calls = [], []
        self.factory_error = factory_error
        self.parallel = SimpleNamespace(
            enable_expert_parallel=False,
            enable_elastic_ep=False,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            enable_dbo=False,
            use_sequence_parallel_moe=False,
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        )
        signature = _signature(FACTORY, "FusedMoEFactory")
        harness = self

        def inject(name, **exports):
            parts = name.split(".")
            for index in range(1, len(parts) + 1):
                part = ".".join(parts[:index])
                if part not in sys.modules or index == len(parts):
                    module = ModuleType(part)
                    module.__path__ = []
                    monkeypatch.setitem(sys.modules, part, module)
            module = sys.modules[name]
            for key, value in exports.items():
                setattr(module, key, value)

        # Force a hermetic test namespace, even if another test imported vllm.
        for name in tuple(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                monkeypatch.delitem(sys.modules, name)

        class ReplicatedLinear(nn.Module):
            def __init__(
                self,
                input_size,
                output_size,
                bias=True,
                quant_config=None,
                prefix="",
                out_dtype=None,
                params_dtype=None,
            ):
                super().__init__()
                assert bias is False
                self.prefix, self.quant = prefix, quant_config is not None
                self.weight = nn.Parameter(_weight(prefix, (output_size, input_size)))

            def forward(self, x):
                harness.events.append((self.prefix, x.detach().clone()))
                return F.linear(_dynamic_fp8(x) if self.quant else x, self.weight), None

        class ColumnLinear(nn.Module):
            def __init__(self, input_size, output_sizes, bias, quant_config, prefix):
                super().__init__()
                assert bias is False and output_sizes[0] == output_sizes[1]
                width = output_sizes[0]
                full = _weight(prefix, (2 * width, input_size)).view(
                    2, width, input_size
                )
                shard = full[:, rank * (width // tp) : (rank + 1) * (width // tp)]
                self.weight = nn.Parameter(shard.reshape(-1, input_size))
                self.prefix = prefix

            def forward(self, x):
                harness.events.append((self.prefix, x.detach().clone()))
                return F.linear(x, self.weight), None

        class RowLinear(nn.Module):
            def __init__(
                self,
                input_size,
                output_size,
                bias,
                quant_config,
                reduce_results,
                prefix,
            ):
                super().__init__()
                self.reduce_results, self.prefix = reduce_results, prefix
                full = _weight(prefix, (output_size, input_size))
                self.weight = nn.Parameter(
                    full[:, rank * (input_size // tp) : (rank + 1) * (input_size // tp)]
                )

            def forward(self, x):
                harness.events.append((self.prefix, x.detach().clone()))
                out = F.linear(x, self.weight)
                if self.reduce_results:
                    out = harness.reduce(out, "shared")
                return out, None

        class SiluAndMul(nn.Module):
            def forward(self, x):
                gate, up = x.chunk(2, -1)
                return F.silu(gate) * up

        class RMSNorm(nn.Module):
            def __init__(self, size, eps):
                super().__init__()
                self.variance_epsilon = eps
                self.weight = nn.Parameter(torch.ones(size))

            def forward(self, x):
                return (
                    x
                    * torch.rsqrt(
                        x.square().mean(-1, keepdim=True) + self.variance_epsilon
                    )
                    * self.weight
                )

        class CPUExperts(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.args = args
                width, size = args["intermediate_size"], args["hidden_size"]
                self.w13 = _weight(
                    args["prefix"] + ".w13", (args["num_experts"], 2, width, size)
                )
                self.w2 = _weight(
                    args["prefix"] + ".w2", (args["num_experts"], size, width)
                )
                self.shared = args.get("shared_experts")

            def forward(self, hidden_states, router_logits):
                args = self.args
                scores = router_logits.sigmoid()
                indices = (
                    (scores + args["e_score_correction_bias"])
                    .topk(args["top_k"], dim=-1)
                    .indices
                )
                weights = scores.gather(1, indices)
                if args["renormalize"]:
                    weights = weights / weights.sum(-1, keepdim=True)
                weights = weights * args["routed_scaling_factor"]
                width = args["intermediate_size"]
                shard = slice(rank * (width // tp), (rank + 1) * (width // tp))
                output = torch.zeros_like(hidden_states)
                for expert in range(args["num_experts"]):
                    gu = F.linear(
                        hidden_states,
                        self.w13[expert, :, shard].reshape(-1, args["hidden_size"]),
                    )
                    gate, up = gu.chunk(2, -1)
                    limit = args["swiglu_limit"]
                    if limit is not None:
                        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
                    part = F.linear(F.silu(gate) * up, self.w2[expert, :, shard])
                    routed_weight = ((indices == expert) * weights).sum(
                        -1, keepdim=True
                    )
                    output = output + part * routed_weight
                harness.events.append(("latent_partial", output.detach().clone()))
                if self.shared is not None:  # Core no-latent's fused shared path.
                    output = output + self.shared(hidden_states)
                return harness.reduce(output, "routed")

        def factory(**kwargs):
            # Reject invented keywords using the *actual* pinned factory API.
            bound = signature.bind(**kwargs)
            bound.apply_defaults()
            self.factory_calls.append(kwargs)
            if self.factory_error:
                raise NotImplementedError("test native factory rejection")
            assert bound.arguments["reduce_results"] is True
            return CPUExperts(bound.arguments)

        inject("transformers", PreTrainedConfig=object)
        inject(
            "vllm.config",
            get_current_vllm_config=lambda: SimpleNamespace(
                parallel_config=self.parallel, lora_config=None
            ),
        )
        # Core constructs this group across TP even when expert parallel is off.
        ep = SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: tp), rank_in_group=rank
        )
        inject(
            "vllm.distributed",
            get_ep_group=lambda: ep,
            get_tensor_model_parallel_world_size=lambda: tp,
        )
        inject(
            "vllm.logger",
            init_logger=lambda _: SimpleNamespace(debug_once=lambda *a, **k: None),
        )
        inject(
            "vllm.model_executor.layers.fused_moe",
            FusedMoEFactory=factory,
            GateLinear=ReplicatedLinear,
        )
        inject(
            "vllm.model_executor.layers.linear",
            ReplicatedLinear=ReplicatedLinear,
            MergedColumnParallelLinear=ColumnLinear,
            RowParallelLinear=RowLinear,
        )
        inject("vllm.model_executor.layers.activation", SiluAndMul=SiluAndMul)
        inject("vllm.model_executor.layers.layernorm", RMSNorm=RMSNorm)
        inject("vllm.model_executor.layers.quantization", QuantizationConfig=object)
        # Explicit config doubles for CPU composition; these do not qualify
        # Core quantized kernels. The actual production admission code runs.
        class Fp8Config:
            is_checkpoint_fp8_serialized = True
            activation_scheme = "dynamic"
            weight_block_size = [128, 128]
            store_dtype = "fp8"
        class CompressedTensorsConfig:
            pass
        self.fp8_config = Fp8Config
        inject("vllm.model_executor.layers.quantization.fp8", Fp8Config=Fp8Config)
        inject("vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
               CompressedTensorsConfig=CompressedTensorsConfig)
        self.quantization = _load(monkeypatch,
            "hcu_vllm_das.models.hy_v4_flash.quantization",
            ACTIVE.with_name("quantization.py"))

        inject("vllm.models.hy_v4.nvidia")
        self.core = _load(monkeypatch, "vllm.models.hy_v4.nvidia.moe", CORE_MOE)
        self.active = _load(monkeypatch, "_flash_active_moe_under_test", ACTIVE)

    def reduce(self, x, branch):
        if self.tp > 1:
            self.events.append(("all_reduce:" + branch, x.detach().clone()))
            dist.all_reduce(x)  # Real CPU/Gloo, not a communication no-op.
        return x


def _reference(config, x, dynamic):
    """Unsharded oracle, independent of the factory/linear spies."""
    prefix = "model.layers.0.mlp"
    logits = F.linear(x, _weight(prefix + ".gate", (config.num_experts, 7)))
    scores = logits.sigmoid()
    bias = torch.tensor([0.1, -0.15, 0.05, 0.0])
    ids = (scores + bias).topk(config.num_experts_per_tok, -1).indices
    weights = scores.gather(1, ids)
    if config.route_norm:
        weights /= weights.sum(-1, keepdim=True)
    weights *= config.router_scaling_factor
    latent = config.routed_expert_hidden_size
    z = x
    if latent is not None:
        z = F.linear(
            _dynamic_fp8(x) if dynamic else x,
            _weight(prefix + ".routed_expert_down_proj", (latent, 7)),
        )
    width, size = config.expert_hidden_dim, z.shape[-1]
    w13 = _weight(prefix + ".experts.w13", (config.num_experts, 2, width, size))
    w2 = _weight(prefix + ".experts.w2", (config.num_experts, size, width))
    routed = torch.zeros_like(z)
    for expert in range(config.num_experts):
        gate, up = F.linear(z, w13[expert].reshape(-1, size)).chunk(2, -1)
        if config.swiglu_limit > 0:
            gate = gate.clamp(max=config.swiglu_limit)
            up = up.clamp(-config.swiglu_limit, config.swiglu_limit)
        out = F.linear(F.silu(gate) * up, w2[expert])
        routed += out * ((ids == expert) * weights).sum(-1, keepdim=True)
    reduced_latent = routed.clone()
    if latent is not None:
        routed = F.linear(
            _dynamic_fp8(routed) if dynamic else routed,
            _weight(prefix + ".routed_expert_up_proj", (7, latent)),
        )
    if config.num_shared_experts:
        width *= config.num_shared_experts
        gu = F.linear(
            x, _weight(prefix + ".shared_experts.gate_up_proj", (2 * width, 7))
        )
        gate, up = gu.chunk(2, -1)
        routed += F.linear(
            F.silu(gate) * up, _weight(prefix + ".shared_experts.down_proj", (7, width))
        )
    if config.sandwich_norm:
        routed *= torch.rsqrt(
            routed.square().mean(-1, keepdim=True) + config.rms_norm_eps
        )
    return routed, reduced_latent


def _numeric_worker(rank, tp, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=tp,
        timeout=timedelta(seconds=60),
    )
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            harness = _Harness(monkeypatch, tp, rank)
            x = _weight("input", (6, 7))
            quant = harness.fp8_config()
            cases, max_error, wrong_gap = 0, 0.0, 0.0
            for latent in (None, 5):
                for shared in (0, 1, 2):
                    for dynamic in (False, True):
                        for norm in (False, True):
                            harness.events.clear()
                            config = _config(
                                routed_expert_hidden_size=latent,
                                num_shared_experts=shared,
                                sandwich_norm=norm,
                            )
                            model = harness.active.HYV4FlashMoE(
                                config, quant if dynamic else None, "model.layers.0.mlp"
                            )
                            model.expert_bias.data.copy_(
                                torch.tensor([0.1, -0.15, 0.05, 0.0])
                            )
                            actual = model(x.reshape(2, 3, 7))
                            # With no latent, no replicated quantized projections exist.
                            expected, reduced = _reference(config, x, dynamic)
                            torch.testing.assert_close(
                                actual.reshape(6, 7), expected, rtol=3e-5, atol=3e-5
                            )
                            cases += 1
                            max_error = max(
                                max_error,
                                (actual.reshape(6, 7) - expected).abs().max().item(),
                            )
                            reductions = [
                                n
                                for n, _ in harness.events
                                if n.startswith("all_reduce:")
                            ]
                            expected_count = (
                                1 + int(shared > 0 and latent is not None)
                                if tp > 1
                                else 0
                            )
                            assert len(reductions) == expected_count, reductions
                            calls = [n for n, _ in harness.events]
                            assert calls.count("model.layers.0.mlp.gate") == 1
                            assert calls.count("latent_partial") == 1
                            assert calls.count(
                                "model.layers.0.mlp.shared_experts.down_proj"
                            ) == int(shared > 0)
                            if latent is not None:
                                assert (
                                    calls.count(
                                        "model.layers.0.mlp.routed_expert_down_proj"
                                    )
                                    == 1
                                )
                                up = "model.layers.0.mlp.routed_expert_up_proj"
                                assert calls.count(up) == 1
                                up_input = next(t for n, t in harness.events if n == up)
                                torch.testing.assert_close(
                                    up_input, reduced, rtol=3e-5, atol=3e-5
                                )
                                if tp > 1:
                                    assert calls.index(
                                        "all_reduce:routed"
                                    ) < calls.index(up)
                                    assert reductions.count("all_reduce:shared") == int(
                                        shared > 0
                                    )
                                    if dynamic:
                                        partial = next(
                                            t
                                            for n, t in harness.events
                                            if n == "latent_partial"
                                        )
                                        wrong = _dynamic_fp8(partial)
                                        dist.all_reduce(
                                            wrong
                                        )  # Independent negative oracle.
                                        wrong_gap = max(
                                            wrong_gap,
                                            (wrong - _dynamic_fp8(reduced))
                                            .abs()
                                            .max()
                                            .item(),
                                        )
                                        assert not torch.allclose(
                                            wrong,
                                            _dynamic_fp8(reduced),
                                            rtol=1e-4,
                                            atol=1e-5,
                                        )
                            assert isinstance(model, harness.core.HYV4MoEFused)
            print(
                "CPU_GLOO_SUMMARY="
                + json.dumps(
                    dict(
                        tp=tp,
                        rank=rank,
                        cases=cases,
                        max_abs_error=max_error,
                        nonlinear_wrong_order_max_gap=wrong_gap,
                    )
                ),
                flush=True,
            )
            # Non-contiguous and empty token inputs retain the external shape.
            model = harness.active.HYV4FlashMoE(_config(), None, "model.layers.0.mlp")
            model.expert_bias.data.zero_()
            assert model(torch.empty(0, 7)).shape == (0, 7)
            assert model(x.reshape(2, 3, 7).transpose(0, 1)).shape == (3, 2, 7)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp", [1, 2])
def test_real_cpu_gloo_composition_and_collective_counts(tp, tmp_path):
    """24 cases/rank: latent/plain, shared 0/1/2, dynamic FP8, sandwich norm."""
    mp.spawn(
        _numeric_worker, args=(tp, (tmp_path / "gloo").as_uri()), nprocs=tp, join=True
    )


def test_core_identity_loader_metadata_factory_and_gate_contract(monkeypatch):
    h = _Harness(monkeypatch, tp=2, rank=1)
    model = h.active.HYV4FlashMoE(_config(), None, "layers.3.mlp")
    assert isinstance(model, h.core.HYV4MoEFused)
    baseline = h.core.HYV4MoEFused(_config(), prefix="layers.3.mlp")
    for name in (
        "tp_size",
        "ep_rank",
        "ep_size",
        "n_routed_experts",
        "n_logical_experts",
        "n_redundant_experts",
        "n_physical_experts",
        "n_local_physical_experts",
        "physical_expert_start",
        "physical_expert_end",
    ):
        assert getattr(model, name) == getattr(baseline, name), name
    names = dict(model.named_parameters())
    assert "gate.weight" in names and "expert_bias" in names
    assert model.gate.weight.dtype == model.expert_bias.dtype == torch.float32
    assert "routed_expert_down_proj.weight" in names
    assert "routed_expert_up_proj.weight" in names
    args = h.factory_calls[0]
    assert args["hidden_size"] == 5 and args["shared_experts"] is None
    assert args["e_score_correction_bias"] is model.expert_bias
    assert args["reduce_results"] is True
    assert model.shared_experts.down_proj.reduce_results is True
    assert not any("transform" in key for key in args)
    assert not any("hcu" in key for key in vars(model.routed_expert_up_proj))


@pytest.mark.parametrize("latent", [None, 5])
@pytest.mark.parametrize("name", ["compressed-tensors", "gptq", "hcu_dsv41_ct_w4a8"])
def test_unvalidated_quantization_fails_before_factory(monkeypatch, latent, name):
    h = _Harness(monkeypatch)
    with pytest.raises(NotImplementedError, match="real Core FP8 or compressed-tensors"):
        h.active.HYV4FlashMoE(
            _config(routed_expert_hidden_size=latent),
            SimpleNamespace(get_name=lambda: name),
        )
    assert not h.factory_calls


def test_flash_w4a8_fails_before_factory(monkeypatch):
    h = _Harness(monkeypatch)
    with pytest.raises(ValueError, match="serialized channel W4A8 CT"):
        h.active.HYV4FlashMoE(_config(flash_w4a8_backend="adapter_reference"))
    assert not h.factory_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_expert_parallel", True),
        ("enable_elastic_ep", True),
        ("data_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("use_sequence_parallel_moe", True),
        ("enable_eplb", True),
    ],
)
def test_latent_rejects_unverified_parallel_domains(monkeypatch, field, value):
    h = _Harness(monkeypatch)
    setattr(h.parallel, field, value)
    with pytest.raises(NotImplementedError, match="ordinary TP"):
        h.active.HYV4FlashMoE(_config())
    assert not h.factory_calls


@pytest.mark.parametrize("latent", [0, -2, True, 2.5])
def test_invalid_latent_dimension_fails_before_factory(monkeypatch, latent):
    h = _Harness(monkeypatch)
    with pytest.raises(ValueError, match="positive integer"):
        h.active.HYV4FlashMoE(_config(routed_expert_hidden_size=latent))
    assert not h.factory_calls


@pytest.mark.parametrize("latent", [None, 5])
def test_factory_rejection_is_not_swallowed_or_fallback(monkeypatch, latent):
    h = _Harness(monkeypatch, factory_error=True)
    with pytest.raises(NotImplementedError, match="native factory rejection"):
        h.active.HYV4FlashMoE(_config(routed_expert_hidden_size=latent))
    assert len(h.factory_calls) == 1


def test_dense_flash_ffn_norm_and_reduction(monkeypatch):
    h = _Harness(monkeypatch)
    config = _config(sandwich_norm=True)
    block = h.active.HYV4FlashFeedForward(config, prefix="dense")
    x = _weight("input", (6, 7))
    gate, up = F.linear(x, _weight("dense.gate_up_proj", (16, 7))).chunk(2, -1)
    out = F.linear(F.silu(gate) * up, _weight("dense.down_proj", (7, 8)))
    expected = out * torch.rsqrt(
        out.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    torch.testing.assert_close(block(x), expected)
    assert block.down_proj.reduce_results is True


def test_production_imports_only_core_and_no_registration():
    tree = ast.parse(ACTIVE.read_text(encoding="utf-8"))
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert all(
        module.startswith(("vllm.", "torch", "transformers",
                           "hcu_vllm_das.models.hy_v4_flash.quantization",
                           "hcu_vllm_das.models.hy_v4_flash.router")) for module in imports
    )
    text = ACTIVE.read_text(encoding="utf-8")
    for forbidden in (
        "vllm_hcu",
        "routed_input_transform",
        "routed_output_transform",
        "register_model",
        "_hcu_fuse_shared",
        "__globals__",
    ):
        assert forbidden not in text


@pytest.mark.parametrize("kind", ["argument", "redundancy"])
def test_latent_eplb_argument_and_redundancy_are_closed(monkeypatch, kind):
    h = _Harness(monkeypatch)
    if kind == "redundancy":
        h.parallel.eplb_config.num_redundant_experts = 1
    with pytest.raises(NotImplementedError, match="ordinary TP"):
        h.active.HYV4FlashMoE(_config(), enable_eplb=kind == "argument")
    assert not h.factory_calls


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"hidden_act": "gelu"}, "silu"),
        ({"num_experts": 1}, "Tensor parallel"),
    ],
)
def test_latent_rejects_invalid_core_factory_shapes(monkeypatch, changes, match):
    h = _Harness(monkeypatch, tp=2)
    with pytest.raises(ValueError, match=match):
        h.active.HYV4FlashMoE(_config(**changes))
    assert not h.factory_calls


def test_quant_config_is_prepared_for_dense_and_moe(monkeypatch):
    h = _Harness(monkeypatch)
    calls = []
    sentinel = object()
    def prepare(config, quant_config, *, for_moe=False):
        calls.append((config, quant_config, for_moe))
        return sentinel
    monkeypatch.setattr(h.quantization, "prepare_flash_quant_config", prepare)
    config = _config()
    assert h.active._check_quantization(config, None) is sentinel
    assert h.active._check_quantization(config, None, for_moe=True) is sentinel
    assert calls == [(config, None, False), (config, None, True)]
