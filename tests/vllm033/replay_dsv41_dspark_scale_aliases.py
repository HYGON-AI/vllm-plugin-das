#!/usr/bin/env python3
"""CPU source integration for DSpark alias loading, written before implementation.

Real native draft loop, remapper, expert mapping, CT Linear parameters and merged
shard loaders. Hardware admission/TP/kernel construction and expert destination
callbacks are doubles. Never runs a model forward, GPU, install or download.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext
from hcu_vllm_das import dsv41_activation as activation
from hcu_vllm_das.quantization import dsv41_channel_fp8 as candidate
from replay_dsv41_channel_fp8_loading import CpuKernel
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.quantization import compressed_tensors
from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors as ct
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_fp8 as scheme
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import CompressedTensorsW8A8Fp8MoEMethod
from vllm.model_executor.models import ModelRegistry
from vllm.models.deepseek_v41.amd.dspark import DSparkDeepseekV4ForCausalLM as Native
import vllm.model_executor.layers.quantization as quant

ARCH = "DSparkV41DraftModel"
MODULE = "hcu_vllm_das.models.dsv41_dspark"
SCHEMA = {"quant_method": "fp8", "is_per_channel": True, "activation_scheme": "dynamic"}
CONTEXT = ExtensionContext("0.1.0", "0.33.0")
ROOT = Path(__file__).resolve().parents[2]
RESULTS = []
K, WIDTH = 16, 3


def case(name, callback):
    try:
        callback()
        RESULTS.append({"case": name, "status": "passed"})
    except Exception as error:
        RESULTS.append({"case": name, "status": "failed", "error": f"{type(error).__name__}: {error}"})


def rejects(callback, kind=Exception, message=None):
    try:
        callback()
    except kind as error:
        if message is not None:
            assert message in str(error), str(error)
        return
    raise AssertionError("expected rejection did not occur")


def setup_doubles(stack):
    stack.enter_context(patch.object(candidate, "_is_hcu", return_value=True))
    stack.enter_context(patch.object(ct.CompressedTensorsConfig, "_check_scheme_supported", return_value=True))
    stack.enter_context(patch.object(scheme, "get_current_vllm_config", return_value=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))))
    stack.enter_context(patch.object(scheme, "init_fp8_linear_kernel", side_effect=lambda **kwargs: CpuKernel(**kwargs)))
    stack.enter_context(patch.object(parameter, "get_tensor_model_parallel_rank", return_value=0))
    stack.enter_context(patch.object(parameter, "get_tensor_model_parallel_world_size", return_value=1))


def configuration():
    value = candidate.HcuDSV41ChannelFP8Config.from_config(SCHEMA)
    value.packed_modules_mapping = {"fused_wqa_wkv": ["wq_a", "wkv"]}
    return value


def fixture(base, config=None, expert_dtype="fp8", confidence=True):
    obj = base.__new__(base)
    torch.nn.Module.__init__(obj)
    obj.quant_config = config if config is not None else configuration()
    obj.config = SimpleNamespace(expert_dtype=expert_dtype, dspark_n_routed_experts=2, n_routed_experts=7)
    obj.linear_scale_name = "weight_scale_inv"
    obj.pad_shared_expert = False
    obj.model = torch.nn.Module()
    def column(prefix):
        return linear.ColumnParallelLinear(K, WIDTH, bias=False, disable_tp=True,
                    quant_config=configuration(), params_dtype=torch.bfloat16, prefix=prefix)
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.fused_wqa_wkv = linear.MergedColumnParallelLinear(
        K, [2, 3], bias=False, disable_tp=True, quant_config=configuration(),
        params_dtype=torch.bfloat16, prefix="model.layers.0.attn.fused_wqa_wkv")
    block.ffn = torch.nn.Module()
    block.ffn.experts = torch.nn.Module()
    experts = torch.nn.Module()
    experts.quant_method = object.__new__(CompressedTensorsW8A8Fp8MoEMethod)
    block.ffn.experts.routed_experts = experts
    obj.expert_calls = []
    for name, shape in (("w13_weight", (2, WIDTH * 2, K)),
                        ("w2_weight", (2, K, WIDTH)),
                        ("w13_weight_scale", (2, WIDTH * 2, 1)),
                        ("w2_weight_scale", (2, K, 1))):
        dtype = torch.float32 if "scale" in name else torch.float8_e4m3fn
        param = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        def expert_loader(param, loaded, weight_name, *, shard_id, expert_id, return_success, name=name):
            assert return_success is True and expert_id in (0, 1)
            assert shard_id in ("w1", "w2", "w3")
            obj.expert_calls.append((name, shard_id, expert_id, weight_name))
            destination = param.data[expert_id]
            if name.startswith("w13_"):
                start = WIDTH if shard_id == "w3" else 0
                destination = destination.narrow(0, start, WIDTH)
            destination.copy_(loaded)
            return True
        param.weight_loader = expert_loader
        param.quant_method = "channel"
        experts.register_parameter(name, param)
    block.unrelated = torch.nn.Module()
    block.unrelated.register_parameter("weight_scale", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
    obj.model.layers = torch.nn.ModuleList([block])
    obj.model.main_proj = column("model.main_proj")
    obj.model.markov_head = torch.nn.Module()
    obj.model.markov_head.markov_w1 = column("model.markov_head.markov_w1")
    obj.model.markov_head.markov_w2 = column("model.markov_head.markov_w2")
    obj.model.confidence_head = torch.nn.Module() if confidence else None
    if confidence:
        obj.model.confidence_head.register_parameter("weight", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
    obj.model.embed_tokens = torch.nn.Embedding(1, 1)
    obj.lm_head = torch.nn.Linear(1, 1, bias=False)
    obj.model.embed_tokens.weight.data.zero_()
    obj.lm_head.weight.data.zero_()
    with torch.no_grad():
        for param in obj.parameters():
            param.zero_()
    return obj


def weights(seed, suffix="weight_scale", confidence=True):
    generator = torch.Generator().manual_seed(seed)
    def values(shape, scale=False):
        if scale:
            return torch.randint(1, 17, shape, generator=generator).float() / 4
        return torch.randint(-4, 5, shape, generator=generator).to(torch.float8_e4m3fn)
    result = []
    for name, rows in (("attn.wq_a", 2), ("attn.wkv", 3)):
        result.extend([(f"mtp.0.{name}.weight", values((rows, K))),
                       (f"mtp.0.{name}.{suffix}", values((rows, 1), True))])
    for prefix in ("main_proj", "markov_head.embed", "markov_head.head"):
        result.extend([(f"mtp.0.{prefix}.weight", values((WIDTH, K))),
                       (f"mtp.0.{prefix}.{suffix}", values((WIDTH, 1), True))])
    for expert in range(2):
        for shard, rows, columns in (("w1", WIDTH, K), ("w3", WIDTH, K), ("w2", K, WIDTH)):
            prefix = f"mtp.0.ffn.experts.{expert}.{shard}"
            result.extend([(f"{prefix}.weight", values((rows, columns))),
                           (f"{prefix}.{suffix}", values((rows, 1), True))])
    if confidence:
        result.append(("mtp.0.confidence_head.weight", torch.tensor([0.5])))
    result.append(("embed.weight", torch.full((1, 1), 999.0)))
    return result


def same_state(reference, current):
    left, right = reference.state_dict(), current.state_dict()
    assert left.keys() == right.keys()
    for name, tensor in left.items():
        other = right[name]
        assert tensor.shape == other.shape and tensor.dtype == other.dtype and tensor.device == other.device, name
        assert torch.equal(tensor.contiguous().view(torch.uint8), other.contiguous().view(torch.uint8)), name


def isolated(stock):
    stack = ExitStack()
    stack.enter_context(patch.object(ModelRegistry, "models", dict(ModelRegistry.models)))
    ModelRegistry.models[ARCH] = stock  # Fixture isolation only; production uses public registration.
    stack.enter_context(patch.object(activation, "_publication", None))
    stack.enter_context(patch.object(activation, "_failure", None))
    stack.enter_context(patch.object(quant, "QUANTIZATION_METHODS", [n for n in quant.QUANTIZATION_METHODS if n != activation.QUANTIZATION_METHOD]))
    stack.enter_context(patch.object(quant, "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG", {n:v for n,v in quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.items() if n != activation.QUANTIZATION_METHOD}))
    return stack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = (Native.__init__, Native.forward, Native.load_weights, torch.nn.Module.named_parameters)
    stock = ModelRegistry.models[ARCH]
    previous_entries = dict(ModelRegistry.models)
    with ExitStack() as stack:
        setup_doubles(stack)
        activation.activate_extension(CONTEXT)
        import sys
        was_lazy = MODULE not in sys.modules
        selected, resolved_arch = ModelRegistry.resolve_model_cls(ARCH, SimpleNamespace(model_impl="vllm"))
        def lazy_registration():
            assert was_lazy, "activation eagerly imported the model implementation"
        case("model_registration_stays_lazy", lazy_registration)
        def selected_model():
            assert resolved_arch == ARCH and selected.__module__ == MODULE
            assert issubclass(selected, Native)
            assert tuple(inspect.signature(selected.__init__).parameters) == tuple(inspect.signature(Native.__init__).parameters) and selected.forward is Native.forward
            for arch, entry in previous_entries.items():
                if arch != ARCH:
                    assert ModelRegistry.models[arch] is entry, arch
        case("public_model_selection_and_inherited_execution", selected_model)
        # The baseline can still execute its real AMD loader and show its failures.
        cls = selected if selected.__module__ == MODULE else Native

        def differential():
            for seed in range(200):
                dtype = "fp4" if seed % 2 else "fp8"
                confidence = seed % 3 != 0
                reference = fixture(Native, expert_dtype=dtype, confidence=confidence)
                expected = reference.load_weights(iter(weights(seed, confidence=confidence)))
                for suffix in ("scale", "weight_scale", "weight_scale_inv"):
                    current = fixture(cls, expert_dtype=dtype, confidence=confidence)
                    before = {n:id(p) for n,p in current.named_parameters()}
                    actual = current.load_weights(iter(weights(seed, suffix, confidence)))
                    assert actual == expected, (seed, suffix, actual, expected)
                    same_state(reference, current)
                    after = {n:id(p) for n,p in current.named_parameters()}
                    assert before == after, (seed, suffix, "parameter identity changed")
                    assert actual <= set(after)
                    assert "model.embed_tokens.weight" not in actual and "lm_head.weight" not in actual
                    assert len(current.expert_calls) == 12
                    assert {x[2] for x in current.expert_calls} == {0, 1}
        case("200_seed_native_canonical_vs_three_alias_spellings", differential)

        def cleanup():
            obj = fixture(cls)
            names = {n:id(p) for n,p in obj.named_parameters()}
            error = RuntimeError("iterator interrupted")
            def broken():
                yield weights(3, "scale")[0]
                raise error
            try:
                obj.load_weights(broken())
            except Exception as caught:
                assert caught is error
            else:
                raise AssertionError("iterator failure swallowed")
            assert names == {n:id(p) for n,p in obj.named_parameters()}
            assert obj.load_weights(iter(weights(4, "scale")))
        case("exception_cleanup_and_successful_retry", cleanup)

        def view_and_thread():
            obj = fixture(cls)
            canonical = dict(obj.named_parameters())
            def stream():
                view = dict(obj.named_parameters())
                name = "model.main_proj.weight_scale"
                assert view[name + "_inv"] is canonical[name]
                seen = []
                thread = threading.Thread(target=lambda: seen.append(dict(obj.named_parameters())))
                thread.start(); thread.join(3)
                assert not thread.is_alive() and seen[0].keys() == canonical.keys()
                rejects(lambda: obj.load_weights(iter(())), RuntimeError)
                yield from weights(5, "scale")
            obj.load_weights(stream())
            assert dict(obj.named_parameters()).keys() == canonical.keys()
        case("load_only_view_thread_isolation_and_reentrancy", view_and_thread)

        def native_pair():
            obj = fixture(cls)
            layer = obj.model.main_proj
            extra = torch.nn.Parameter(torch.zeros_like(layer.weight_scale), requires_grad=False)
            layer.register_parameter("weight_scale_inv", extra)
            real = layer.weight_scale.clone()
            loaded = obj.load_weights(iter([("mtp.0.main_proj.weight_scale_inv", torch.full_like(extra, 7))]))
            assert loaded == {"model.main_proj.weight_scale_inv"}
            assert torch.equal(extra, torch.full_like(extra, 7)) and torch.equal(layer.weight_scale, real)
        case("dual_native_names_win_without_normalization", native_pair)
        def nonct():
            obj = fixture(cls)
            rejects(lambda: obj.load_weights(iter([("mtp.0.unrelated.weight_scale_inv", torch.ones(1))])), KeyError)
            assert not any(n.endswith("unrelated.weight_scale_inv") for n,_ in obj.named_parameters())
        case("non_ct_and_unknown_aliases_remain_native_errors", nonct)
        def native_delegate():
            obj = fixture(cls, config=object())
            marker = object()
            with patch.object(Native, "load_weights", return_value=marker) as call:
                assert obj.load_weights(iter(())) is marker
                call.assert_called_once()
        case("other_quantization_delegates_unchanged", native_delegate)
        def signature_drift():
            obj = fixture(cls)
            with patch.object(Native, "load_weights", lambda self, weights, extra: set()):
                rejects(lambda: obj.load_weights(iter(())), RuntimeError, "signature")
        case("native_loader_signature_drift_rejected", signature_drift)
        def foreign_enumeration():
            obj = fixture(cls)
            bound = obj.named_parameters
            obj.named_parameters = lambda *args, **kwargs: bound(*args, **kwargs)
            rejects(lambda: obj.load_weights(iter(())), RuntimeError, "named_parameters")
            assert "named_parameters" in vars(obj)
        case("foreign_instance_enumeration_rejected_without_replacement", foreign_enumeration)
        def confidence_removal():
            obj = fixture(cls, confidence=True)
            obj.load_weights(iter(weights(6, "scale", False)))
            assert obj.model.confidence_head is None
        case("native_absent_confidence_head_behavior", confidence_removal)
        def idempotent():
            entry = ModelRegistry.models[ARCH]
            activation.activate_extension(CONTEXT)
            activation.check_health()
            assert ModelRegistry.models[ARCH] is entry
        case("activation_and_model_publication_idempotence", idempotent)
        def foreign():
            with isolated(stock):
                ModelRegistry.register_model(ARCH, "foreign_module:ForeignDraft")
                entry = ModelRegistry.models[ARCH]
                rejects(lambda: activation.activate_extension(CONTEXT), RuntimeError)
                assert ModelRegistry.models[ARCH] is entry
                assert activation.QUANTIZATION_METHOD not in quant.QUANTIZATION_METHODS
        case("foreign_model_owner_refused_before_quant_publication", foreign)
        def nonhcu():
            with isolated(stock), patch.object(candidate, "_is_hcu", return_value=False):
                activation.activate_extension(CONTEXT)
                assert ModelRegistry.models[ARCH] is stock
        case("non_hcu_activation_preserves_native_model_entry", nonhcu)
        def lost_owner():
            with isolated(stock):
                activation.activate_extension(CONTEXT)
                ModelRegistry.register_model(ARCH, "foreign_module:ForeignDraft")
                rejects(activation.check_health, ExtensionCommitError)
                rejects(lambda: activation.activate_extension(CONTEXT), ExtensionCommitError)
        case("lost_model_owner_is_fatal_and_sticky", lost_owner)
        def partial():
            with isolated(stock):
                original_register = ModelRegistry.register_model
                def broken(model_arch, model_cls):
                    original_register(model_arch, model_cls)
                    raise RuntimeError("model published then failed")
                with patch.object(ModelRegistry, "register_model", broken):
                    rejects(lambda: activation.activate_extension(CONTEXT), ExtensionCommitError)
                    rejects(lambda: activation.activate_extension(CONTEXT), ExtensionCommitError)
        case("partial_model_publication_is_fatal_and_sticky", partial)
        def unchanged():
            assert original == (Native.__init__, Native.forward, Native.load_weights, torch.nn.Module.named_parameters)
        case("native_methods_are_not_rebound", unchanged)
    assert not torch.cuda.is_initialized()
    def sha(root):
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    sources = {}
    for relative in ("adapter_extension/src/hcu_vllm_das/dsv41_activation.py",
                     "adapter_extension/src/hcu_vllm_das/dsv41_dspark_registration.py",
                     "adapter_extension/src/hcu_vllm_das/models/dsv41_dspark.py"):
        path = ROOT / relative
        if path.is_file(): sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"level": "CPU native draft-loading source integration, not model/GPU acceptance",
              "native_loader_source": inspect.getsourcefile(Native), "candidate_module": selected.__module__,
              "plugin": sha(ROOT), "core": sha(ROOT.parent / "vllm"), "adapter": sha(ROOT.parent / "adapter-das"),
              "installed_vllm_metadata": metadata.version("vllm"), "cuda_initialized": torch.cuda.is_initialized(),
              "source_sha256": sources, "cases": RESULTS,
              "passed": sum(r["status"] == "passed" for r in RESULTS), "total": len(RESULTS)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
