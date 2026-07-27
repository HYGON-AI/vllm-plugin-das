# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""vLLM v0.25.1 ownership and HCU-delta contracts for Qwen GDN."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.op_opt._common import PatchCompatibilityError


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_VLLM_ROOT = Path(
    os.environ.get("VLLM_V0251_SOURCE_ROOT", REPO_ROOT.parent / "vllm_0251")
).resolve()
# Do not resolve this symlink: resolving it would lose the venv package view.
TARGET_PYTHON = Path(
    os.environ.get(
        "VLLM_V0251_PYTHON",
        TARGET_VLLM_ROOT / ".venv/bin/python",
    )
)
QWEN_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"


def _adapter(name: str):
    return importlib.import_module(f"vllm_hcu.patch.worker.op_opt.{name}")


def _run_fresh_v0251(code: str) -> subprocess.CompletedProcess[str]:
    assert (TARGET_VLLM_ROOT / "vllm/__init__.py").is_file()
    assert TARGET_PYTHON.is_file()
    env = dict(os.environ)
    env.update(
        {
            "VLLM_PLUGINS": "__disabled__",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "VLLM_V0251_SOURCE_ROOT": str(TARGET_VLLM_ROOT),
            "VLLM_HCU_SOURCE_ROOT": str(REPO_ROOT),
            "VLLM_USE_NN": "1",
            "VLLM_HCU_USE_CUSTOM_OPS": "0",
            "VLLM_HCU_USE_CUSTOM_AITER_FLA": "0",
            "VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D": "0",
            "VLLM_HCU_MAMBA_SSM_CACHE_DTYPE": "0",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), str(TARGET_VLLM_ROOT))
    )
    return subprocess.run(
        [str(TARGET_PYTHON), "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


def _causal_contract(
    x,
    weight,
    bias=None,
    conv_states=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=-1,
    null_block_id=0,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    num_computed_tokens=None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    del (
        x,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
        null_block_id,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        block_size_to_align,
        metadata,
        validate_data,
    )
    return weight


def _causal_update_contract(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1,
    null_block_id=0,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    validate_data=False,
):
    del (
        x,
        conv_state,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        null_block_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
        validate_data,
    )
    return weight


def _aiter_update_contract(
    x,
    num_actual_tokens,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    ba,
    z_out,
    core_attn_out,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1,
    pad_slot_id=-1,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    validate_data=False,
    qkvz_layout="interleaved",
):
    del (
        x,
        num_actual_tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        ba,
        z_out,
        core_attn_out,
        conv_state,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        pad_slot_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
        validate_data,
        qkvz_layout,
    )
    return weight


def _recording_callable(contract, name: str, calls: dict[str, object]):
    signature = inspect.signature(contract)

    def target(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        calls[name] = {
            "raw_args": args,
            "raw_kwargs": dict(kwargs),
            "arguments": dict(bound.arguments),
        }
        return bound.arguments["weight"]

    target.__name__ = name
    target.__signature__ = signature  # type: ignore[attr-defined]
    return target


def _fake_qwen(*, aiter_available: bool = True):
    calls: dict[str, object] = {}
    causal = _recording_callable(_causal_contract, "causal", calls)
    update = _recording_callable(_causal_update_contract, "update", calls)
    aiter_update = _recording_callable(_aiter_update_contract, "aiter", calls)

    class GatedDeltaNetAttention:
        model_config = SimpleNamespace(dtype=torch.float16)
        cache_config = SimpleNamespace(
            mamba_cache_dtype="float16",
            mamba_ssm_cache_dtype="float32",
        )

        def get_state_dtype(self):
            calls["target_state_dtype"] = self
            return ("target", self.cache_config.mamba_ssm_cache_dtype)

    class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
        pass

    def recurrent(*args, **kwargs):
        del args, kwargs
        return "target-recurrent"

    def sigmoid(*args, **kwargs):
        del args, kwargs
        return "target-sigmoid"

    values = {
        "GDN_AITER_TRITON_AVAILABLE": aiter_available,
        "causal_conv1d_fn": causal,
        "causal_conv1d_update": update,
        "fused_recurrent_gated_delta_rule_packed_decode": recurrent,
        "fused_sigmoid_gating_delta_rule_update": sigmoid,
        "QwenGatedDeltaNetAttention": QwenGatedDeltaNetAttention,
    }
    if aiter_available:
        values[
            "gdn_aiter_fused_reshape_causal_conv1d_update_single_token"
        ] = aiter_update
    module = ModuleType(QWEN_MODULE)
    module.__dict__.update(values)
    return module, calls, GatedDeltaNetAttention, recurrent, sigmoid


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **values):
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        module_name = ".".join(parts[:index])
        module = sys.modules.get(module_name)
        if module is None or index == len(parts):
            module = ModuleType(module_name)
            module.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, module_name, module)
        if index > 1:
            parent = sys.modules[".".join(parts[: index - 1])]
            monkeypatch.setattr(parent, parts[index - 1], module, raising=False)
    module.__dict__.update(values)
    return module


def _call_aiter(module, conv_state, weight):
    return module.gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
        x=torch.empty(1),
        num_actual_tokens=1,
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=1,
        head_v_dim=1,
        ba=torch.empty(1),
        z_out=torch.empty(1),
        core_attn_out=torch.empty(1),
        conv_state=conv_state,
        weight=weight,
        bias=None,
        activation="silu",
        conv_state_indices=None,
        validate_data=True,
    )


def test_dispatcher_scopes_all_gdn_callbacks_to_qwen():
    import vllm_hcu.patch.worker as worker

    expected = {
        "worker.op_opt.mamba.gdn.causal_conv1d": QWEN_MODULE,
        "worker.op_opt.mamba.gdn.base_state_dtype": QWEN_MODULE,
        "worker.op_opt.mamba.gdn.qwen_kernel_bindings": QWEN_MODULE,
    }
    callbacks = dict(worker.worker_callback_names())
    assert {patch_id: callbacks[patch_id] for patch_id in expected} == expected
    assert not any(
        target
        in {
            "vllm.model_executor.layers.mamba.ops.causal_conv1d",
            "vllm.model_executor.layers.mamba.gdn.base",
        }
        for patch_id, target in worker.worker_callback_names()
        if ".gdn." in patch_id
    )


@pytest.mark.parametrize("use_nn", [False, True])
def test_qwen_local_weight_deltas_and_target_fla_ownership(
    monkeypatch: pytest.MonkeyPatch,
    use_nn: bool,
):
    causal_adapter = _adapter("patch_gdn_causal_conv1d")
    aiter_adapter = _adapter("patch_gdn_linear_attention")
    module, calls, _, recurrent, sigmoid = _fake_qwen(aiter_available=True)
    canonical = ModuleType("vllm.model_executor.layers.mamba.ops.causal_conv1d")
    canonical.causal_conv1d_fn = module.causal_conv1d_fn
    canonical.causal_conv1d_update = module.causal_conv1d_update
    consumer = ModuleType("consumer")
    consumer.causal_conv1d_fn = canonical.causal_conv1d_fn
    consumer.causal_conv1d_update = canonical.causal_conv1d_update

    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_USE_NN", use_nn)
    assert causal_adapter.apply_to_module(module) is True
    assert aiter_adapter.apply_to_module(module) is True
    assert causal_adapter.apply_to_module(module) is False
    assert aiter_adapter.apply_to_module(module) is False

    assert canonical.causal_conv1d_fn is consumer.causal_conv1d_fn
    assert canonical.causal_conv1d_update is consumer.causal_conv1d_update
    assert module.causal_conv1d_fn is not canonical.causal_conv1d_fn
    assert module.causal_conv1d_update is not canonical.causal_conv1d_update
    assert module.fused_recurrent_gated_delta_rule_packed_decode is recurrent
    assert module.fused_sigmoid_gating_delta_rule_update is sigmoid
    assert not hasattr(module, "_vllm_hcu_original_fused_recurrent")
    assert not hasattr(module, "_vllm_hcu_original_fused_sigmoid")

    conv_state = torch.empty(1, 8, 3)
    x_fn = torch.empty(8, 2)
    x_update = torch.empty(2, 8)
    weight = torch.arange(32, dtype=torch.float32).reshape(
        (4, 8) if use_nn else (8, 4)
    )
    expected = weight.T.contiguous() if use_nn else weight

    fn_result = module.causal_conv1d_fn(
        x=x_fn,
        weight=weight,
        bias=None,
        conv_states=conv_state,
        query_start_loc=torch.tensor([0, 2]),
    )
    update_result = module.causal_conv1d_update(
        x_update,
        conv_state,
        weight,
    )
    aiter_result = _call_aiter(module, conv_state, weight)
    for result in (fn_result, update_result, aiter_result):
        torch.testing.assert_close(result, expected)

    for name in ("causal", "update", "aiter"):
        record = calls[name]
        torch.testing.assert_close(record["arguments"]["weight"], expected)
    if not use_nn:
        # Feature-off preserves the original keyword invocation exactly.
        assert calls["causal"]["raw_args"] == ()
        assert calls["causal"]["raw_kwargs"]["weight"] is weight


@pytest.mark.parametrize("enabled", [False, True])
def test_state_dtype_override_is_qwen_local_and_feature_gated(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
):
    adapter = _adapter("patch_gdn_base")
    module, _, base_class, _, _ = _fake_qwen(aiter_available=False)
    qwen_class = module.QwenGatedDeltaNetAttention
    base_method = base_class.get_state_dtype
    calculator_calls = []

    class Calculator:
        @staticmethod
        def gated_delta_net_state_dtype(model_dtype, cache_dtype, ssm_dtype="auto"):
            calculator_calls.append((model_dtype, cache_dtype, ssm_dtype))
            return ("hcu", ssm_dtype)

    _install_module(
        monkeypatch,
        "vllm.model_executor.layers.mamba.mamba_utils",
        MambaStateDtypeCalculator=Calculator,
    )
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_MAMBA_SSM_CACHE_DTYPE", enabled)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", enabled)
    assert adapter.apply_to_module(module) is True
    assert adapter.apply_to_module(module) is False

    class OtherGDN(base_class):
        pass

    qwen_result = qwen_class().get_state_dtype()
    assert base_class.get_state_dtype is base_method
    assert OtherGDN.get_state_dtype is base_method
    assert base_class().get_state_dtype() == ("target", "float32")
    assert OtherGDN().get_state_dtype() == ("target", "float32")
    if enabled:
        assert qwen_result == ("hcu", "auto")
        assert calculator_calls == [(torch.float16, "float16", "auto")]
    else:
        assert qwen_result == ("target", "float32")
        assert calculator_calls == []


def test_native_aiter_unavailable_is_idempotent_and_does_not_require_symbol():
    adapter = _adapter("patch_gdn_linear_attention")
    module, _, _, recurrent, sigmoid = _fake_qwen(aiter_available=False)
    assert adapter.apply_to_module(module) is True
    assert adapter.apply_to_module(module) is False
    assert module.fused_recurrent_gated_delta_rule_packed_decode is recurrent
    assert module.fused_sigmoid_gating_delta_rule_update is sigmoid
    assert not hasattr(
        module,
        "gdn_aiter_fused_reshape_causal_conv1d_update_single_token",
    )


def test_native_aiter_signature_and_keyword_calls_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = _adapter("patch_gdn_linear_attention")
    module, _, _, _, _ = _fake_qwen(aiter_available=True)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    assert adapter.apply_to_module(module) is True
    with pytest.raises(PatchCompatibilityError, match="audited vLLM v0.25.1"):
        module.gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
            x=torch.empty(1),
            unexpected=torch.empty(1),
        )

    bad_module, _, _, _, _ = _fake_qwen(aiter_available=True)

    def incompatible(x, weight):
        return x, weight

    bad_module.gdn_aiter_fused_reshape_causal_conv1d_update_single_token = (
        incompatible
    )
    with pytest.raises(PatchCompatibilityError, match="incompatible parameters"):
        adapter.apply_to_module(bad_module)


def test_real_v0251_cold_import_scopes_gdn_deltas_to_qwen():
    result = _run_fresh_v0251(
        r'''\
import importlib
import json
import os
import sys
from pathlib import Path

import vllm
import vllm_hcu
import vllm.platforms as platforms
from vllm.platforms.interface import UnspecifiedPlatform

source_root = Path(os.environ["VLLM_V0251_SOURCE_ROOT"]).resolve()
hcu_root = Path(os.environ["VLLM_HCU_SOURCE_ROOT"]).resolve()
assert Path(vllm.__file__).resolve() == source_root / "vllm/__init__.py"
assert Path(vllm_hcu.__file__).resolve() == hcu_root / "vllm_hcu/__init__.py"
platforms._current_platform = UnspecifiedPlatform()

from vllm_hcu.patch.runtime_state import patch_report
from vllm_hcu.patch.worker import prepare_worker_patches, worker_callback_names

qwen_name = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
canonical_name = "vllm.model_executor.layers.mamba.ops.causal_conv1d"
base_name = "vllm.model_executor.layers.mamba.gdn.base"
assert qwen_name not in sys.modules
assert canonical_name not in sys.modules
assert base_name not in sys.modules
assert "vllm._aiter_ops" not in sys.modules

ids = (
    "worker.op_opt.mamba.gdn.causal_conv1d",
    "worker.op_opt.mamba.gdn.base_state_dtype",
    "worker.op_opt.mamba.gdn.qwen_kernel_bindings",
)
callback_targets = dict(worker_callback_names())
assert {patch_id: callback_targets[patch_id] for patch_id in ids} == {
    patch_id: qwen_name for patch_id in ids
}

prepare_worker_patches()
qwen = importlib.import_module(qwen_name)
canonical = importlib.import_module(canonical_name)
base = importlib.import_module(base_name)
fla = importlib.import_module("vllm.model_executor.layers.fla.ops")

assert Path(qwen.__file__).resolve() == (
    source_root
    / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
)
assert qwen.causal_conv1d_fn is not canonical.causal_conv1d_fn
assert qwen.causal_conv1d_update is not canonical.causal_conv1d_update
assert qwen._vllm_hcu_original_causal_conv1d_fn is canonical.causal_conv1d_fn
assert qwen._vllm_hcu_original_causal_conv1d_update is canonical.causal_conv1d_update
assert not hasattr(canonical, "_vllm_hcu_gdn_causal_conv1d_applied")
assert not hasattr(canonical, "_vllm_hcu_original_causal_conv1d_fn")

qwen_cls = qwen.QwenGatedDeltaNetAttention
base_method = base.GatedDeltaNetAttention.get_state_dtype
assert qwen_cls.get_state_dtype is not base_method
assert qwen_cls._vllm_hcu_original_get_state_dtype is base_method
assert not hasattr(base, "_vllm_hcu_gdn_base_applied")
assert not getattr(base_method, "_vllm_hcu_gdn_base_wrapper", False)

assert (
    qwen.fused_recurrent_gated_delta_rule_packed_decode
    is fla.fused_recurrent_gated_delta_rule_packed_decode
)
assert (
    qwen.fused_sigmoid_gating_delta_rule_update
    is fla.fused_sigmoid_gating_delta_rule_update
)
assert not hasattr(qwen, "_vllm_hcu_original_fused_recurrent")
assert not hasattr(qwen, "_vllm_hcu_original_fused_sigmoid")
assert not bool(qwen.GDN_AITER_TRITON_AVAILABLE)
assert getattr(qwen, "_vllm_hcu_qwen_gdn_aiter_layout_applied", False)

consumer_specs = (
    "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn",
    "vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn",
    "vllm.model_executor.layers.mamba.mamba_mixer",
    "vllm.model_executor.layers.mamba.mamba_mixer2",
    "vllm.model_executor.layers.mamba.short_conv",
)
consumers = {name: importlib.import_module(name) for name in consumer_specs}
for name, consumer in consumers.items():
    assert consumer.causal_conv1d_fn is canonical.causal_conv1d_fn, name
    assert consumer.causal_conv1d_update is canonical.causal_conv1d_update, name

kimi = consumers["vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn"]
olmo = consumers["vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn"]
assert "get_state_dtype" in kimi.KimiGatedDeltaNetAttention.__dict__
assert kimi.KimiGatedDeltaNetAttention.get_state_dtype is not base_method
assert "get_state_dtype" not in olmo.OlmoHybridGatedDeltaNetAttention.__dict__
assert olmo.OlmoHybridGatedDeltaNetAttention.get_state_dtype is base_method

report = patch_report()["patches"]
statuses = {patch_id: report[patch_id]["status"] for patch_id in ids}
assert set(statuses.values()) == {"applied"}
print(json.dumps({
    "sentinel": "GDN_V0251_OWNERSHIP_OK",
    "statuses": statuses,
}, sort_keys=True))
'''
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["sentinel"] == "GDN_V0251_OWNERSHIP_OK"
    assert set(payload["statuses"].values()) == {"applied"}
