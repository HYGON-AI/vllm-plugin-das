# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import (
    patch_deepseek_v32_config,
    patch_gpt_oss_mlp_block,
    patch_qwen4_exp,
    patch_qwen3_5_mamba_state_dtype,
    patch_qwen3_vl,
    patch_qwen3_vl_moe,
)
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError
from vllm_hcu.patch.runtime_state import PATCH_REGISTRY


_MISSING = object()


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _vllm_config(hf_config: object, **attributes: object) -> SimpleNamespace:
    values = {"model_config": SimpleNamespace(hf_config=hf_config), **attributes}
    return SimpleNamespace(**values)


def test_deepseek_retirement_preserves_official_verifier_and_is_idempotent():
    calls: list[object] = []

    class DeepseekV32ForCausalLM:
        @classmethod
        def verify_and_update_config(cls, vllm_config):
            calls.append(vllm_config)
            assert hasattr(vllm_config.model_config.hf_config, "index_topk")
            return "official"

    module = _module(
        patch_deepseek_v32_config.TARGET_MODULE,
        DeepseekV32ForCausalLM=DeepseekV32ForCausalLM,
    )
    original = vars(DeepseekV32ForCausalLM)["verify_and_update_config"]

    assert patch_deepseek_v32_config.apply_to_module(module) is True
    assert patch_deepseek_v32_config.apply_to_module(module) is False
    assert vars(DeepseekV32ForCausalLM)["verify_and_update_config"] is original

    config = _vllm_config(SimpleNamespace(index_topk=8))
    assert DeepseekV32ForCausalLM.verify_and_update_config(config) == "official"
    assert calls == [config]


@pytest.mark.parametrize(
    ("has_index_topk", "disable_dsa", "expected"),
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
    ],
)
def test_deepseek_model_side_dsa_guard(
    has_index_topk: bool, disable_dsa: bool, expected: bool
):
    hf_config = SimpleNamespace()
    if has_index_topk:
        hf_config.index_topk = 8
    assert (
        patch_deepseek_v32_config.is_hcu_dsa_enabled(
            hf_config, disable_dsa=disable_dsa
        )
        is expected
    )


def test_deepseek_retirement_rejects_non_classmethod():
    class DeepseekV32ForCausalLM:
        def verify_and_update_config(self, vllm_config):
            return None

    module = _module(
        patch_deepseek_v32_config.TARGET_MODULE,
        DeepseekV32ForCausalLM=DeepseekV32ForCausalLM,
    )
    with pytest.raises(PatchCompatibilityError, match="must be a classmethod"):
        patch_deepseek_v32_config.apply_to_module(module)


class _Weight:
    def __init__(self, name: str, calls: list[tuple[object, ...]]):
        self.name = name
        self.calls = calls

    def t(self):
        self.calls.append(("transpose", self.name))
        return _Weight(f"{self.name}.T", self.calls)


@pytest.mark.parametrize("use_nn", [False, True])
def test_gpt_oss_adapter_transposes_only_for_nn_layout(
    monkeypatch: pytest.MonkeyPatch, use_nn: bool
):
    calls: list[tuple[object, ...]] = []

    def rocm_unquantized_gemm(layer, x, weight, bias=None):
        calls.append(("gemm", layer, x, weight.name, bias))
        return weight.name

    def linear(x, weight, bias=None):
        calls.append(("linear", x, weight.name, bias))
        return weight.name

    module = _module(
        patch_gpt_oss_mlp_block.TARGET_MODULE,
        rocm_unquantized_gemm=rocm_unquantized_gemm,
        torch=SimpleNamespace(
            nn=SimpleNamespace(functional=SimpleNamespace(linear=linear))
        ),
    )
    monkeypatch.setattr(
        patch_gpt_oss_mlp_block, "_use_nn_layout", lambda: use_nn
    )

    assert patch_gpt_oss_mlp_block.apply_to_module(module) is True
    wrapper = module.rocm_unquantized_gemm
    assert patch_gpt_oss_mlp_block.apply_to_module(module) is False
    assert module.rocm_unquantized_gemm is wrapper

    result = wrapper("layer", "x", _Weight("W", calls), "bias")
    assert result == ("W.T" if use_nn else "W")
    if use_nn:
        assert ("transpose", "W") in calls
        assert any(call[0] == "linear" for call in calls)
        assert not any(call[0] == "gemm" for call in calls)
    else:
        assert ("transpose", "W") not in calls
        assert any(call[0] == "gemm" for call in calls)
        assert not any(call[0] == "linear" for call in calls)


def test_gpt_oss_nn_layout_adapter_is_numerically_equivalent(
    monkeypatch: pytest.MonkeyPatch,
):
    original_calls: list[tuple[object, ...]] = []

    def rocm_unquantized_gemm(layer, x, weight, bias=None):
        original_calls.append((layer, x, weight, bias))
        raise AssertionError("NN-layout path must not call the upstream custom op")

    module = _module(
        patch_gpt_oss_mlp_block.TARGET_MODULE,
        rocm_unquantized_gemm=rocm_unquantized_gemm,
        torch=torch,
    )
    monkeypatch.setattr(patch_gpt_oss_mlp_block, "_use_nn_layout", lambda: True)
    patch_gpt_oss_mlp_block.apply_to_module(module)

    x = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 4.0]])
    # HCU NN layout is [in_features, out_features].
    weight = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    bias = torch.tensor([0.25, -0.5])
    actual = module.rocm_unquantized_gemm(object(), x, weight, bias)
    expected = x @ weight + bias
    torch.testing.assert_close(actual, expected)
    assert original_calls == []


def test_gpt_oss_adapter_fails_clearly_for_bad_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module(
        patch_gpt_oss_mlp_block.TARGET_MODULE,
        rocm_unquantized_gemm=lambda x, weight: None,
    )
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_gpt_oss_mlp_block.apply_to_module(module)

    def rocm_unquantized_gemm(layer, x, weight, bias=None):
        return None

    def linear(x, weight, bias=None):
        return None

    module = _module(
        patch_gpt_oss_mlp_block.TARGET_MODULE,
        rocm_unquantized_gemm=rocm_unquantized_gemm,
        torch=SimpleNamespace(
            nn=SimpleNamespace(functional=SimpleNamespace(linear=linear))
        ),
    )
    monkeypatch.setattr(patch_gpt_oss_mlp_block, "_use_nn_layout", lambda: True)
    patch_gpt_oss_mlp_block.apply_to_module(module)
    with pytest.raises(PatchCompatibilityError, match=r"without t\(\)"):
        module.rocm_unquantized_gemm("layer", "x", object())


def _qwen3_5_module(calls: list[tuple[object, ...]]) -> ModuleType:
    class MambaStateDtypeCalculator:
        @staticmethod
        def gated_delta_net_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype="auto"
        ):
            calls.append(
                (
                    "calculate",
                    model_dtype,
                    mamba_cache_dtype,
                    mamba_ssm_cache_dtype,
                )
            )
            return (mamba_cache_dtype, mamba_ssm_cache_dtype)

    class Qwen3_5ForConditionalGeneration:
        @classmethod
        def get_mamba_state_dtype_from_config(cls, vllm_config):
            calls.append(("official", cls))
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            )

    return _module(
        patch_qwen3_5_mamba_state_dtype.TARGET_MODULE,
        MambaStateDtypeCalculator=MambaStateDtypeCalculator,
        Qwen3_5ForConditionalGeneration=Qwen3_5ForConditionalGeneration,
    )


@pytest.mark.parametrize("feature_enabled", [False, True])
def test_qwen3_5_classmethod_wrapper_preserves_off_behavior(
    monkeypatch: pytest.MonkeyPatch, feature_enabled: bool
):
    calls: list[tuple[object, ...]] = []
    module = _qwen3_5_module(calls)
    model_class = module.Qwen3_5ForConditionalGeneration
    monkeypatch.setattr(
        patch_qwen3_5_mamba_state_dtype,
        "_auto_dtype_enabled",
        lambda: feature_enabled,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype="bf16"),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="float16",
            mamba_ssm_cache_dtype="float32",
        ),
    )

    assert patch_qwen3_5_mamba_state_dtype.apply_to_module(module) is True
    assert patch_qwen3_5_mamba_state_dtype.apply_to_module(module) is False
    assert model_class.get_mamba_state_dtype_from_config(config) == (
        "float16",
        "auto" if feature_enabled else "float32",
    )
    assert any(call[0] == "official" for call in calls) is (not feature_enabled)


def test_qwen3_5_feature_off_preserves_subclass_cls(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    module = _qwen3_5_module(calls)
    model_class = module.Qwen3_5ForConditionalGeneration

    class Derived(model_class):
        pass

    monkeypatch.setattr(
        patch_qwen3_5_mamba_state_dtype,
        "_auto_dtype_enabled",
        lambda: False,
    )
    patch_qwen3_5_mamba_state_dtype.apply_to_module(module)
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype="bf16"),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="float16",
            mamba_ssm_cache_dtype="float32",
        ),
    )
    Derived.get_mamba_state_dtype_from_config(config)
    assert ("official", Derived) in calls


def test_qwen3_5_rejects_non_classmethod_contract():
    class MambaStateDtypeCalculator:
        @staticmethod
        def gated_delta_net_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype="auto"
        ):
            return None

    class Qwen3_5ForConditionalGeneration:
        def get_mamba_state_dtype_from_config(self, vllm_config):
            return None

    module = _module(
        patch_qwen3_5_mamba_state_dtype.TARGET_MODULE,
        MambaStateDtypeCalculator=MambaStateDtypeCalculator,
        Qwen3_5ForConditionalGeneration=Qwen3_5ForConditionalGeneration,
    )
    with pytest.raises(PatchCompatibilityError, match="must be a classmethod"):
        patch_qwen3_5_mamba_state_dtype.apply_to_module(module)


@pytest.mark.parametrize("value", [_MISSING, False, True])
def test_qwen3_vl_dense_normalizes_missing_only(value: object):
    calls: list[object] = []

    class Qwen3LLMForCausalLM:
        def __init__(self, *, vllm_config, prefix=""):
            config = vllm_config.model_config.hf_config
            calls.append(getattr(config, "tie_word_embeddings", _MISSING))

    module = _module(
        patch_qwen3_vl.TARGET_MODULE,
        Qwen3LLMForCausalLM=Qwen3LLMForCausalLM,
    )
    hf_config = SimpleNamespace()
    if value is not _MISSING:
        hf_config.tie_word_embeddings = value

    assert patch_qwen3_vl.apply(module) is True
    assert patch_qwen3_vl.apply_to_module(module) is False
    Qwen3LLMForCausalLM(vllm_config=_vllm_config(hf_config))
    expected = False if value is _MISSING else value
    assert hf_config.tie_word_embeddings is expected
    assert calls == [expected]


def test_qwen3_vl_dense_rejects_signature_drift():
    class Qwen3LLMForCausalLM:
        def __init__(self, vllm_config, prefix=""):
            pass

    module = _module(
        patch_qwen3_vl.TARGET_MODULE,
        Qwen3LLMForCausalLM=Qwen3LLMForCausalLM,
    )
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_qwen3_vl.apply_to_module(module)


def _qwen3_vl_moe_module(calls: list[tuple[object, ...]]) -> ModuleType:
    class Qwen3MoeLLMForCausalLM:
        def __init__(self, *, vllm_config, prefix=""):
            config = vllm_config.model_config.hf_config
            calls.append(
                ("llm", getattr(config, "tie_word_embeddings", _MISSING))
            )

    class Qwen3VLMoeForConditionalGeneration:
        def __init__(self, *, vllm_config, prefix=""):
            config = vllm_config.model_config.hf_config
            calls.append(
                (
                    "vl",
                    getattr(config, "tie_word_embeddings", _MISSING),
                    getattr(config.text_config, "tie_word_embeddings", _MISSING),
                )
            )

    return _module(
        patch_qwen3_vl_moe.TARGET_MODULE,
        Qwen3MoeLLMForCausalLM=Qwen3MoeLLMForCausalLM,
        Qwen3VLMoeForConditionalGeneration=Qwen3VLMoeForConditionalGeneration,
    )


def _qwen4_exp_module(calls: list[tuple[object, ...]]) -> ModuleType:
    class Qwen4ExpForCausalLM:
        def __init__(self, *, vllm_config, prefix=""):
            config = vllm_config.model_config.hf_config
            calls.append(
                ("llm", getattr(config, "tie_word_embeddings", _MISSING))
            )

    class Qwen4ExpForConditionalGeneration:
        def __init__(self, *, vllm_config, prefix="model"):
            config = vllm_config.model_config.hf_config
            calls.append(
                (
                    "vl",
                    getattr(config, "tie_word_embeddings", _MISSING),
                    getattr(config.text_config, "tie_word_embeddings", _MISSING),
                )
            )

    return _module(
        patch_qwen4_exp.TARGET_MODULE,
        Qwen4ExpForCausalLM=Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration=Qwen4ExpForConditionalGeneration,
    )


@pytest.mark.parametrize(
    ("top_value", "text_value", "expected_text"),
    [
        (_MISSING, _MISSING, False),
        (_MISSING, True, True),
        (None, True, True),
        (False, True, False),
        (True, False, True),
    ],
)
def test_qwen4_exp_two_wrappers_are_atomic_and_preserve_values(
    top_value: object, text_value: object, expected_text: bool
):
    calls: list[tuple[object, ...]] = []
    module = _qwen4_exp_module(calls)
    top_config = SimpleNamespace(text_config=SimpleNamespace())
    if top_value is not _MISSING:
        top_config.tie_word_embeddings = top_value
    if text_value is not _MISSING:
        top_config.text_config.tie_word_embeddings = text_value

    assert patch_qwen4_exp.apply_to_module(module) is True
    llm_wrapper = module.Qwen4ExpForCausalLM.__init__
    vl_wrapper = module.Qwen4ExpForConditionalGeneration.__init__
    assert patch_qwen4_exp.apply_to_module(module) is False
    assert module.Qwen4ExpForCausalLM.__init__ is llm_wrapper
    assert module.Qwen4ExpForConditionalGeneration.__init__ is vl_wrapper

    module.Qwen4ExpForConditionalGeneration(
        vllm_config=_vllm_config(top_config)
    )
    module.Qwen4ExpForCausalLM(
        vllm_config=_vllm_config(top_config.text_config)
    )

    assert top_config.text_config.tie_word_embeddings is expected_text
    assert calls[-1] == ("llm", expected_text)


def test_qwen4_exp_validation_failure_does_not_partially_patch():
    class Qwen4ExpForCausalLM:
        def __init__(self, *, vllm_config, prefix=""):
            pass

    class Qwen4ExpForConditionalGeneration:
        def __init__(self, vllm_config, prefix="model"):
            pass

    llm_original = Qwen4ExpForCausalLM.__init__
    vl_original = Qwen4ExpForConditionalGeneration.__init__
    module = _module(
        patch_qwen4_exp.TARGET_MODULES[0],
        Qwen4ExpForCausalLM=Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration=Qwen4ExpForConditionalGeneration,
    )

    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_qwen4_exp.apply_to_module(module)
    assert Qwen4ExpForCausalLM.__init__ is llm_original
    assert Qwen4ExpForConditionalGeneration.__init__ is vl_original


@pytest.mark.parametrize(
    ("top_value", "text_value", "expected_text"),
    [
        (_MISSING, _MISSING, False),
        (_MISSING, True, True),
        (None, True, True),
        (False, True, False),
        (True, False, True),
    ],
)
def test_qwen3_vl_moe_two_wrappers_are_atomic_and_preserve_values(
    top_value: object, text_value: object, expected_text: bool
):
    calls: list[tuple[object, ...]] = []
    module = _qwen3_vl_moe_module(calls)
    top_config = SimpleNamespace(text_config=SimpleNamespace())
    if top_value is not _MISSING:
        top_config.tie_word_embeddings = top_value
    if text_value is not _MISSING:
        top_config.text_config.tie_word_embeddings = text_value

    assert patch_qwen3_vl_moe.apply_to_module(module) is True
    llm_wrapper = module.Qwen3MoeLLMForCausalLM.__init__
    vl_wrapper = module.Qwen3VLMoeForConditionalGeneration.__init__
    assert patch_qwen3_vl_moe.apply_to_module(module) is False
    assert module.Qwen3MoeLLMForCausalLM.__init__ is llm_wrapper
    assert module.Qwen3VLMoeForConditionalGeneration.__init__ is vl_wrapper

    module.Qwen3VLMoeForConditionalGeneration(
        vllm_config=_vllm_config(top_config)
    )
    module.Qwen3MoeLLMForCausalLM(
        vllm_config=_vllm_config(top_config.text_config)
    )

    assert top_config.text_config.tie_word_embeddings is expected_text
    assert calls[-1] == ("llm", expected_text)


def test_qwen3_vl_moe_validation_failure_does_not_partially_patch():
    class Qwen3MoeLLMForCausalLM:
        def __init__(self, *, vllm_config, prefix=""):
            pass

    class Qwen3VLMoeForConditionalGeneration:
        def __init__(self, vllm_config, prefix=""):
            pass

    llm_original = Qwen3MoeLLMForCausalLM.__init__
    vl_original = Qwen3VLMoeForConditionalGeneration.__init__
    module = _module(
        patch_qwen3_vl_moe.TARGET_MODULE,
        Qwen3MoeLLMForCausalLM=Qwen3MoeLLMForCausalLM,
        Qwen3VLMoeForConditionalGeneration=Qwen3VLMoeForConditionalGeneration,
    )

    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_qwen3_vl_moe.apply_to_module(module)
    assert Qwen3MoeLLMForCausalLM.__init__ is llm_original
    assert Qwen3VLMoeForConditionalGeneration.__init__ is vl_original


def test_callbacks_reject_wrong_exact_module():
    wrong = _module("vllm.model_executor.models.not_the_target")
    for patch_module in (
        patch_deepseek_v32_config,
        patch_gpt_oss_mlp_block,
        patch_qwen3_5_mamba_state_dtype,
        patch_qwen4_exp,
        patch_qwen3_vl,
        patch_qwen3_vl_moe,
    ):
        with pytest.raises(PatchCompatibilityError, match="expected module"):
            patch_module.apply_to_module(wrong)


def test_import_callbacks_do_not_touch_runtime_registry(
    monkeypatch: pytest.MonkeyPatch,
):
    PATCH_REGISTRY.reset_for_tests()

    def rocm_unquantized_gemm(layer, x, weight, bias=None):
        return weight

    module = _module(
        patch_gpt_oss_mlp_block.TARGET_MODULE,
        rocm_unquantized_gemm=rocm_unquantized_gemm,
        torch=SimpleNamespace(
            nn=SimpleNamespace(
                functional=SimpleNamespace(linear=lambda x, weight, bias=None: None)
            )
        ),
    )
    monkeypatch.setattr(patch_gpt_oss_mlp_block, "_use_nn_layout", lambda: False)
    assert PATCH_REGISTRY.snapshot() == ()
    patch_gpt_oss_mlp_block.apply_to_module(module)
    assert PATCH_REGISTRY.snapshot() == ()
