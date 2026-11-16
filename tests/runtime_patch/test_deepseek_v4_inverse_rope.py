# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def patch(monkeypatch):
    # Load just this patch: core_fix.__init__ imports unrelated GPU adapters.
    directory = Path(__file__).parents[2] / "vllm_hcu/patch/worker/core_fix"
    package = ModuleType("_inverse_rope_contract")
    package.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: SimpleNamespace(
        info_once=lambda *args: None
    )
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    for stem in ("_common", "patch_deepseek_v4_inverse_rope"):
        name = package.__name__ + "." + stem
        spec = importlib.util.spec_from_file_location(name, directory / (stem + ".py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    if not torch.cuda.is_available():
        monkeypatch.setattr(module, "_register_inverse_rope_fake", lambda: False)
    return module


@pytest.fixture
def ops(monkeypatch):
    # Only the cached projection dependency is needed by the wrapper contract.
    names = ["vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.ops"]
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    module = ModuleType("vllm.v1.attention.ops.rocm_aiter_mla_sparse")
    module._get_cached_wo_a_bf16 = lambda *args: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sys.modules["vllm.v1.attention.ops"].rocm_aiter_mla_sparse = module
    return module


def original(rotary_emb, o, positions, rope_head_dim, n_local_groups,
             o_lora_rank, wo_a):
    return ("original", o, positions)


def target(patch, function=original):
    module = ModuleType(patch.TARGET_MODULE)
    module.rocm_inv_rope_einsum = function
    return module


def test_registration_and_fallback(monkeypatch, patch):
    module = target(patch)
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    monkeypatch.setattr(patch, "_eligible", lambda *args: False)
    o, positions = torch.ones(3, 2, 512), torch.arange(3)
    result = module.rocm_inv_rope_einsum(None, o, positions, 64, 2, 7, None)
    assert result[0] == "original"
    assert result[1] is o
    assert result[2] is positions


def test_stale_marker_and_signature(patch):
    module = target(patch)
    patch.apply_to_module(module)
    module.rocm_inv_rope_einsum = lambda *args: None
    error = sys.modules[patch.__package__ + "._common"].PatchCompatibilityError
    with pytest.raises(error):
        patch.apply_to_module(module)

    module = target(patch, lambda rotary_emb, o: None)
    with pytest.raises(error):
        patch.apply_to_module(module)


@pytest.mark.parametrize(
    "flag", ["enabled", "master", "child", "grad", "cpu", "short", "fp32"]
)
def test_guards_keep_reference(monkeypatch, flag, patch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0" if flag == "master" else "1")
    monkeypatch.setenv(
        "VLLM_HCU_INVERSE_ROPE_SUFFIX", "0" if flag == "child" else "1"
    )
    monkeypatch.setattr(torch.version, "hip", "test")
    rows = 17 if flag == "short" else 512
    rotary = SimpleNamespace(cos_sin_cache=torch.empty(1024, 64))
    o = SimpleNamespace(
        is_cuda=flag != "cpu",
        dtype=torch.float32 if flag == "fp32" else torch.bfloat16,
        ndim=3,
        shape=(rows, 2, 512),
        device=torch.device("cpu"),
        is_contiguous=lambda: True,
    )
    scope = torch.enable_grad() if flag == "grad" else torch.no_grad()
    with scope:
        assert patch._eligible(rotary, o, torch.arange(rows), 64, 2) == (
            flag == "enabled"
        )


def test_optimized_helper_uses_native_op_and_cached_weight(monkeypatch, patch, ops):
    module = target(patch)
    patch.apply_to_module(module)
    monkeypatch.setattr(patch, "_eligible", lambda *args: True)
    weight = torch.randn(2, 7, 512, dtype=torch.bfloat16)
    monkeypatch.setattr(ops, "_get_cached_wo_a_bf16", lambda *args: weight)
    calls = []

    def inverse_rope(rope, positions, cache):
        calls.append((positions, cache))
        rope.mul_(-1)

    monkeypatch.setattr(
        torch.ops.hcu_ops, "deepseek_v4_inv_rope", inverse_rope, raising=False
    )
    rotary = SimpleNamespace(cos_sin_cache=torch.empty(10, 64))
    o = torch.randn(3, 2, 512, dtype=torch.bfloat16)
    expected_o = o.clone()
    expected_o[..., -64:] *= -1
    expected = torch.einsum("tgd,grd->tgr", expected_o, weight)

    result = module.rocm_inv_rope_einsum(
        rotary, o, torch.arange(3), 64, 2, 7, object()
    )

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(o, expected_o)
    assert len(calls) == 1


@pytest.mark.hcu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="inverse RoPE tests need a working GPU"
)
def test_inverse_rope_fake_supports_fullgraph(patch):
    if not patch._register_inverse_rope_fake():
        pytest.skip("HCU inverse RoPE operator is unavailable")
    assert patch._register_inverse_rope_fake()

    def invoke(rope, positions, cache):
        torch.ops.hcu_ops.deepseek_v4_inv_rope(rope, positions, cache)
        return rope

    compiled = torch.compile(invoke, fullgraph=True, backend="eager")
    rope = torch.randn(512, 2, 64, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(512, device="cuda", dtype=torch.int64)
    cache = torch.randn(512, 64, dtype=torch.float32, device="cuda")
    expected = rope.clone()
    torch.ops.hcu_ops.deepseek_v4_inv_rope(expected, positions, cache)

    result = compiled(rope, positions, cache)
    torch.testing.assert_close(result, expected)


def test_complete_wrapper_fullgraph_has_no_forward_logging(monkeypatch, patch, ops):
    messages = []
    monkeypatch.setattr(patch.logger, "info_once", lambda *args: messages.append(args))
    module = target(patch)
    patch.apply_to_module(module)
    assert len(messages) == 1

    def forbidden_log(*args):
        raise AssertionError("logging inside compiled forward")

    monkeypatch.setattr(patch.logger, "info_once", forbidden_log)
    # CPU contract exercises the entire installed wrapper; hardware guards and
    # the custom kernel are covered separately by the HCU test.
    monkeypatch.setattr(patch, "_eligible", lambda *args: True)
    monkeypatch.setattr(
        torch.ops.hcu_ops, "deepseek_v4_inv_rope",
        lambda rope, positions, cache: rope.neg_(), raising=False,
    )
    weight = torch.randn(2, 7, 512, dtype=torch.bfloat16)
    monkeypatch.setattr(ops, "_get_cached_wo_a_bf16", lambda *args: weight)
    rotary = SimpleNamespace(cos_sin_cache=torch.empty(512, 64))
    positions = torch.arange(512)
    o = torch.randn(512, 2, 512, dtype=torch.bfloat16)
    expected_o = o.clone()
    expected_o[..., -64:].neg_()
    expected = torch.einsum("tgd,grd->tgr", expected_o, weight)
    compiled = torch.compile(
        module.rocm_inv_rope_einsum, fullgraph=True, backend="eager"
    )
    result = compiled(rotary, o, positions, 64, 2, 7, None)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(o, expected_o)
