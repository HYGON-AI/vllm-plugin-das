# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts and real-Core-source replay for PCP inverse RoPE."""

from hcu_vllm_das.adapter_api import adapter_module as _adapter_module
import ast
import __future__
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import inverse_rope_hcu as inverse
from hcu_vllm_das import inverse_rope_pcp
hcu_config = _adapter_module('hcu_config')
hcu_env = _adapter_module('hcu_env')
from hcu_adapter.kernels import pcp_fork, pcp_rope_policy
from hcu_vllm_das.inverse_rope_policy import selected_mode


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    config = {"ops": {}, "overrides": {}}
    monkeypatch.setattr(hcu_config, "load", lambda: config)
    monkeypatch.setenv("HCU_DISABLE_OPS", "")
    monkeypatch.setenv("HCU_VLLM_PCP_IMPL", "native")
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_ATTENTION", "1")
    monkeypatch.delenv(pcp_rope_policy.MODE_VARIABLE, raising=False)
    monkeypatch.setattr(pcp_rope_policy, "_process_arch", lambda _pid: "gfx938")
    monkeypatch.setattr(inverse_rope_pcp, "_state", None)
    monkeypatch.setattr(inverse_rope_pcp, "_failure", None)
    return config["ops"]


def test_catalog_validates_inverse_rope():
    chosen, overrides = hcu_config._validate(
        {"version": 1, "ops": {"vllm_inverse_rope": "hcu"}}, "test-config")
    assert chosen == {"vllm_inverse_rope": "hcu"}
    assert overrides == {}


@pytest.mark.parametrize("op", ["vllm_pcp", "vllm_inverse_rope"])
def test_kill_switch_overrides_inverse_rope(monkeypatch, isolated_policy, op):
    assert selected_mode() == "hcu"
    isolated_policy["vllm_inverse_rope"] = "hcu"
    assert selected_mode() == "hcu"
    monkeypatch.setenv("HCU_DISABLE_OPS", op)
    assert selected_mode() == "stock"


def test_invalid_policy_is_rejected(isolated_policy):
    isolated_policy["vllm_inverse_rope"] = "typo"
    with pytest.raises(ValueError, match="vllm_inverse_rope"):
        selected_mode()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_ineligible_inverse_uses_original(missing):
    calls = []
    route = inverse.route(lambda *args: calls.append("stock"),
                          None if missing else lambda *args: calls.append("native"))
    route(torch.zeros(2, 1, 512, dtype=torch.bfloat16), torch.arange(2),
          torch.ones(3, 64), 64)
    assert calls == ["stock"]


def test_empty_inverse_needs_no_native_operator():
    route = inverse.route(lambda *args: pytest.fail("stock called"), None)
    assert route(torch.empty(0, 1, 512), None, None, 64) is None


def test_inverse_loader_stages_sources_outside_the_installed_package(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    original_sources = Path(inverse.__file__).with_name("csrc")
    before = sorted(path.name for path in original_sources.iterdir())
    monkeypatch.setattr(torch.version, "hip", "cpu-contract")
    monkeypatch.setattr(cpp_extension, "_get_build_directory", lambda *args: str(tmp_path))
    calls = []
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(torch.library, "register_fake", lambda name: lambda function: function)
    operator = lambda *args: None
    monkeypatch.setattr(torch.ops.hcu_vllm_das, "deepseek_v4_inv_rope", operator, raising=False)
    inverse.load_operator.cache_clear()
    try:
        assert inverse.load_operator() is operator
        assert len(calls) == 1
        assert calls[0]["extra_cuda_cflags"] == ["-O3", "-ffp-contract=off"]
        for source in calls[0]["sources"]:
            source = Path(source)
            assert source.is_relative_to(tmp_path)
            assert source.read_bytes() == (original_sources / source.name).read_bytes()
        assert sorted(path.name for path in original_sources.iterdir()) == before
        Path(calls[0]["sources"][0]).write_bytes(b"corrupt")
        inverse.load_operator.cache_clear()
        with pytest.raises(RuntimeError, match="corrupt"):
            inverse.load_operator()
    finally:
        inverse.load_operator.cache_clear()


@pytest.mark.parametrize("mode,admitted", [("stock", True), ("native", False)])
def test_inverse_rope_leaves_unowned_attention_on_core(
        monkeypatch, isolated_policy, mode, admitted):
    from hcu_vllm_das.models import v41_pcp as model

    isolated_policy["vllm_inverse_rope"] = "hcu"
    monkeypatch.setattr(pcp_fork, "selected_mode", lambda: mode)
    monkeypatch.setattr(model, "validate_config", lambda cfg: admitted)
    stock = object()
    monkeypatch.setattr(model.native, "_select_dsv4_attn_cls", lambda cfg: stock)
    assert model.select_attention(object()) is stock


def test_inverse_rope_without_pcp_owner_fails_activation(monkeypatch):
    from hcu_adapter.extensions import ExtensionCommitError
    from hcu_vllm_das import v41_pcp_activation as activation

    monkeypatch.setattr(activation, "_state", None)
    monkeypatch.setattr(activation, "_failure", None)
    monkeypatch.setattr(activation, "_publishing", False)
    monkeypatch.setattr(activation, "_is_hcu", lambda: True)
    monkeypatch.setattr(activation, "fp4_selected", lambda: False)
    monkeypatch.setattr(activation, "inverse_rope_mode", lambda: "hcu")
    monkeypatch.setenv(activation._FLAG, "0")
    monkeypatch.setenv(activation._HOST_FLAG, "0")
    with pytest.raises(ExtensionCommitError, match="requires the PCP model owner"):
        activation.activate_extension(NS(vllm_version="0.33.0"))
    assert activation._state is None


def test_method_provider_cannot_change_after_binding(monkeypatch):
    monkeypatch.setattr(pcp_fork, "_forward_mqa_provider", None)
    monkeypatch.setattr(pcp_fork, "_classes", None)
    provider = lambda base: base.forward_mqa
    pcp_fork.set_forward_mqa_provider(provider)
    pcp_fork.set_forward_mqa_provider(provider)
    assert pcp_fork.bound_forward_mqa_provider() is provider
    with pytest.raises(pcp_fork.PcpAttentionError, match="already bound"):
        pcp_fork.set_forward_mqa_provider(lambda base: base.forward_mqa)


def test_method_provider_cannot_be_added_after_class_construction(monkeypatch):
    monkeypatch.setattr(pcp_fork, "_forward_mqa_provider", None)
    monkeypatch.setattr(pcp_fork, "_classes", {})
    with pytest.raises(pcp_fork.PcpAttentionError, match="already bound"):
        pcp_fork.set_forward_mqa_provider(lambda base: base.forward_mqa)


def _core_method(name, namespace):
    root = os.environ.get("VLLM_TARGET_ROOT") or os.environ.get("VLLM_SOURCE_ROOT")
    assert root, "VLLM_TARGET_ROOT must identify the vLLM source tree for replay"
    path = Path(root) / "vllm/models/deepseek_v41/amd/rocm.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "DeepseekV41ROCMAiterMLAAttention")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def test_policy_drift_remains_failed_after_restore(monkeypatch, isolated_policy):
    base = NS(forward_mqa=lambda: None)
    monkeypatch.setattr(inverse_rope_pcp, "_state",
                        (os.getpid(), base, base.forward_mqa, base.forward_mqa))
    isolated_policy["vllm_inverse_rope"] = "stock"
    with pytest.raises(RuntimeError, match="changed"):
        inverse_rope_pcp.check_health()
    isolated_policy["vllm_inverse_rope"] = "hcu"
    with pytest.raises(RuntimeError, match="changed"):
        inverse_rope_pcp.check_health()


def test_real_core_forward_applies_inverse_only_to_unrotated_rows(
        monkeypatch, isolated_policy):
    isolated_policy["vllm_inverse_rope"] = "hcu"
    calls = []

    def stock_inverse(*args):
        pytest.fail("eligible inverse did not reach the prepared provider")

    def native(tail, positions, cache):
        assert tail.shape == (2, 1, 64)
        assert torch.equal(positions, torch.tensor([1, 2]))
        tail.add_(1)
        calls.append("inverse")

    metadata = NS(num_decodes=1, num_prefills=1, num_decode_tokens=1)
    namespace = dict(torch=torch, __name__=__name__,
        QuantizedActivation=type("QuantizedActivation", (), {}),
        get_forward_context=lambda: NS(attn_metadata={"swa": metadata, "compressed": NS()}),
        cast=lambda kind, value: value, DeepseekV4FlashMLAMetadata=NS,
        DeepseekV41ROCMAiterSparseSWAMetadata=NS,
        rocm_inverse_rope_rows_=stock_inverse)
    base = type("Attention", (), {
        name: _core_method(name, namespace) for name in ("forward_mqa", "_forward_prefill")})
    monkeypatch.setitem(sys.modules, "vllm.platforms", NS(
        current_platform=NS(is_rocm=lambda: True)))
    monkeypatch.setattr(hcu_env, "arch_name", lambda: "gfx938")
    def prepare(original):
        assert original is stock_inverse
        return native

    monkeypatch.setattr(inverse, "prepare_operator", prepare)
    monkeypatch.setattr(inverse, "eligible", lambda *args: True)
    monkeypatch.setattr(inverse_rope_pcp.registry, "_RECORDS", {})
    monkeypatch.setattr(inverse_rope_pcp.registry, "_BOUND", {})
    stock_prefill = base._forward_prefill
    forward_mqa = inverse_rope_pcp.prepare_forward_mqa(base)
    prefix = base.__module__ + "." + base.__name__
    assert inverse_rope_pcp.registry.get("vllm_inverse_rope")["target"] == prefix + ".forward_mqa"
    assert inverse_rope_pcp.prepare_forward_mqa(base) is forward_mqa
    assert base.forward_mqa is namespace["forward_mqa"]
    assert base.forward_mqa.__globals__["rocm_inverse_rope_rows_"] is stock_inverse
    assert base._forward_prefill is stock_prefill

    def decode(**arguments):
        arguments["output"].fill_(5)
        arguments["output"][..., -64:].add_(1)
        return 1

    obj = NS(compress_ratio=2, compressed_cache_prefix="compressed",
        swa_cache_layer=NS(prefix="swa", kv_cache=object()),
        _compressed_kv_cache=lambda: object(), n_local_heads=1, rope_head_dim=64,
        rotary_emb=NS(cos_sin_cache=torch.empty(4, 64)),
        _forward_decode=decode,
        _forward_prefill=lambda **kwargs: kwargs["output"].fill_(2))
    output = torch.empty(3, 2, 512, dtype=torch.bfloat16)
    forward_mqa(obj, torch.zeros_like(output), None, torch.arange(3), output)
    assert calls == ["inverse"]
    assert torch.equal(output[0, :, -64:], torch.full((2, 64), 6, dtype=output.dtype))
    assert torch.equal(output[1:, 0, -64:], torch.full((2, 64), 3, dtype=output.dtype))
    assert torch.equal(output[1:, 1, -64:], torch.full((2, 64), 2, dtype=output.dtype))


def _cpu_rounding(output, positions, cache, *, fuse_cosine):
    """Independent FP64 reference; tests probe policy, not device execution."""
    a, b = output[..., -64::2].float(), output[..., -63::2].float()
    c, s = cache[positions.long(), :32, None].transpose(1, 2), cache[positions.long(), 32:, None].transpose(1, 2)
    sine_even = (b.double() * s.double()).float()
    sine_odd = (a.double() * s.double()).float()
    cosine_even, cosine_odd = a.double() * c.double(), b.double() * c.double()
    if not fuse_cosine:
        cosine_even, cosine_odd = cosine_even.float().double(), cosine_odd.float().double()
    output[..., -64::2] = (cosine_even + sine_even.double()).float().to(torch.bfloat16)
    output[..., -63::2] = (cosine_odd - sine_odd.double()).float().to(torch.bfloat16)


@pytest.mark.parametrize("fused", [False, True])
def test_preparation_selects_the_observed_core_arithmetic(fused):
    def original(out, pos, cache, rope_dim):
        assert rope_dim == 64
        _cpu_rounding(out, pos, cache, fuse_cosine=fused)

    def native(tail, pos, cache, *, fuse_cosine):
        _cpu_rounding(tail, pos, cache, fuse_cosine=fuse_cosine)

    assert inverse._probe_mode(original, native, torch.device("cpu")) is fused
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kind", ["unsupported", "ambiguous", "dtype-drift"])
def test_preparation_rejects_unmatched_or_inconsistent_arithmetic(kind):
    def original(out, pos, cache, rope_dim):
        if kind == "unsupported":
            out.fill_(42)
        else:
            _cpu_rounding(out, pos, cache, fuse_cosine=(pos.dtype == torch.int32))

    def native(tail, pos, cache, *, fuse_cosine):
        selected = pos.dtype == torch.int32 if kind == "ambiguous" else fuse_cosine
        _cpu_rounding(tail, pos, cache, fuse_cosine=selected)

    with pytest.raises(RuntimeError, match="unsupported or ambiguous"):
        inverse._probe_mode(original, native, torch.device("cpu"))


def test_inverse_preparation_is_rejected_inside_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(inverse, "load_operator", lambda: pytest.fail("build during capture"))
    with pytest.raises(RuntimeError, match="before Graph capture"):
        inverse.prepare_operator(lambda *args: None)
