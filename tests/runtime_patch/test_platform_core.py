# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import logging
import os
import re
import sys
from types import ModuleType

import pytest

from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
from vllm_hcu.patch.platform.core_fix import (
    patch_envs,
    patch_hy_v3_reasoning_parser,
    patch_hy_v3_tool_parser,
    patch_import_utils,
    patch_nixl_utils,
)
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError
from vllm_hcu.patch.runtime_state import (
    PATCH_REGISTRY,
    PatchRegistry,
    PatchStatus,
)


@pytest.fixture(autouse=True)
def _reset_patch_registry():
    PATCH_REGISTRY.reset_for_tests()
    yield
    PATCH_REGISTRY.reset_for_tests()


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _clear_vllm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("VLLM_"):
            monkeypatch.delenv(name, raising=False)


def test_envs_allows_only_hcu_namespace_and_defaults_aiter_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _clear_vllm_environment(monkeypatch)
    logger = logging.getLogger("test.hcu.envs")

    def original_validate(hard_fail):
        raise AssertionError("the upstream validator should have been replaced")

    module = _module(
        patch_envs.TARGET_MODULE,
        environment_variables={
            "VLLM_KNOWN": lambda: "known",
            "VLLM_ROCM_USE_AITER_MOE": lambda: True,
        },
        validate_environ=original_validate,
        logger=logger,
    )

    assert patch_envs.apply(module) is True
    assert patch_envs.apply(module) is False
    getter = module.environment_variables["VLLM_ROCM_USE_AITER_MOE"]
    assert getter() is False
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    assert getter() is True

    monkeypatch.setenv("VLLM_HCU_FEATURE", "1")
    monkeypatch.setenv("VLLM_KNOWN", "1")
    module.validate_environ(hard_fail=True)

    monkeypatch.setenv("VLLM_NOT_HCU", "1")
    with pytest.raises(ValueError, match="VLLM_NOT_HCU"):
        module.validate_environ(hard_fail=True)
    monkeypatch.delenv("VLLM_NOT_HCU")
    monkeypatch.setenv("VLLM_HCU", "1")
    with caplog.at_level(logging.WARNING, logger="test.hcu.envs"):
        module.validate_environ(hard_fail=False)
    assert "VLLM_HCU" in caplog.text

    record = PATCH_REGISTRY.get(patch_envs.PATCH_ID)
    assert record is not None and record.status is PatchStatus.APPLIED


def test_envs_rejects_signature_drift_and_latches_failure():
    module = _module(
        patch_envs.TARGET_MODULE,
        environment_variables={"VLLM_ROCM_USE_AITER_MOE": lambda: True},
        validate_environ=lambda: None,
        logger=logging.getLogger("test.hcu.bad_envs"),
    )
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_envs.apply(module)
    record = PATCH_REGISTRY.get(patch_envs.PATCH_ID)
    assert record is not None and record.status is PatchStatus.FAILED


def test_import_utils_checks_all_three_deep_gemm_package_names():
    available: set[str] = set()
    probes: list[str] = []

    def has_module(module_name):
        probes.append(module_name)
        return module_name in available

    module = _module(
        patch_import_utils.TARGET_MODULE,
        _has_module=has_module,
        has_deep_gemm=lambda: False,
    )
    assert patch_import_utils.apply(module) is True

    assert module.has_deep_gemm() is False
    assert probes == ["deepgemm", "deep_gemm", "vllm.third_party.deep_gemm"]

    probes.clear()
    available.add("deepgemm")
    assert module.has_deep_gemm() is True
    assert probes == ["deepgemm"]

    probes.clear()
    available.clear()
    available.add("deep_gemm")
    assert module.has_deep_gemm() is True
    assert probes == ["deepgemm", "deep_gemm"]

    probes.clear()
    available.clear()
    available.add("vllm.third_party.deep_gemm")
    assert module.has_deep_gemm() is True
    assert probes == ["deepgemm", "deep_gemm", "vllm.third_party.deep_gemm"]


def test_apply_to_module_is_marker_idempotent_and_does_not_touch_registry():
    module = _module(
        patch_import_utils.TARGET_MODULE,
        _has_module=lambda module_name: module_name == "deepgemm",
        has_deep_gemm=lambda: False,
    )
    assert patch_import_utils.apply_to_module(module) is True
    assert patch_import_utils.apply_to_module(module) is False
    assert module.has_deep_gemm() is True
    assert PATCH_REGISTRY.get(patch_import_utils.PATCH_ID) is None


def test_apply_to_module_runs_inside_coordinator_without_reentrant_registry(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module(
        patch_import_utils.TARGET_MODULE,
        _has_module=lambda module_name: module_name == "deepgemm",
        has_deep_gemm=lambda: False,
    )
    monkeypatch.setitem(sys.modules, patch_import_utils.TARGET_MODULE, module)
    registry = PatchRegistry()
    coordinator = ExactImportCoordinator(registry=registry)

    registration = coordinator.register_callback(
        patch_import_utils.PATCH_ID,
        patch_import_utils.TARGET_MODULE,
        patch_import_utils.apply_to_module,
        targets=patch_import_utils.TARGETS,
    )

    assert registration.status == PatchStatus.APPLIED.value
    assert module.has_deep_gemm() is True
    record = registry.get(patch_import_utils.PATCH_ID)
    assert record is not None and record.status is PatchStatus.APPLIED
    assert record.targets == patch_import_utils.TARGETS
    assert PATCH_REGISTRY.get(patch_import_utils.PATCH_ID) is None


def test_nixl_utils_uses_nixl_for_hcu(
    monkeypatch: pytest.MonkeyPatch,
):
    probes: list[str] = []

    def find_spec(package_name):
        probes.append(package_name)
        return object()

    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    module = _module(
        patch_nixl_utils.TARGET_MODULE,
        _get_nixl_module_name=lambda name: (
            "rixl._bindings" if name == "nixlXferTelemetry" else "rixl._api"
        ),
        is_nixl_available=lambda: False,
    )

    assert patch_nixl_utils.apply(module) is True
    assert patch_nixl_utils.apply(module) is False
    assert module._get_nixl_module_name("NixlWrapper") == "nixl._api"
    assert (
        module._get_nixl_module_name("nixlXferTelemetry")
        == "nixl._bindings"
    )
    assert module.is_nixl_available() is True
    assert probes == ["nixl"]


def test_nixl_utils_rejects_signature_drift():
    module = _module(
        patch_nixl_utils.TARGET_MODULE,
        _get_nixl_module_name=lambda: "rixl._api",
        is_nixl_available=lambda: False,
    )

    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_nixl_utils.apply(module)


class _Tokenizer:
    def __init__(self, vocabulary: dict[str, int], suffix: str | None = None):
        self._vocabulary = vocabulary
        self.init_kwargs = {} if suffix is None else {"token_suffix": suffix}

    def get_vocab(self) -> dict[str, int]:
        return self._vocabulary


def test_reasoning_parser_uses_suffix_before_original_initialization():
    class FakeReasoningParser:
        @property
        def start_token(self):
            return "<think>"

        @property
        def end_token(self):
            return "</think>"

        def __init__(self, tokenizer, *args, **kwargs):
            vocabulary = tokenizer.get_vocab()
            self.start_token_id = vocabulary[self.start_token]
            self.end_token_id = vocabulary[self.end_token]

    module = _module(
        patch_hy_v3_reasoning_parser.TARGET_MODULE,
        HYV3ReasoningParser=FakeReasoningParser,
    )
    patch_hy_v3_reasoning_parser.apply(module)

    tokenizer = _Tokenizer({"<think:hcu>": 17, "</think:hcu>": 18}, ":hcu")
    parser = FakeReasoningParser(tokenizer)
    assert parser.suffix == ":hcu"
    assert parser.start_token == "<think:hcu>"
    assert parser.end_token == "</think:hcu>"
    assert (parser.start_token_id, parser.end_token_id) == (17, 18)


class _FakeToolParser:
    @staticmethod
    def _get_schema_options(arg_schema):
        if "type" in arg_schema:
            return [arg_schema]
        if "anyOf" in arg_schema:
            return arg_schema["anyOf"]
        return [{"type": "string"}]

    @property
    def vocab(self):
        return self.model_tokenizer.get_vocab()

    def __init__(self, tokenizer, tools=None):
        self.model_tokenizer = tokenizer
        self.tools = tools or []
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self._streaming_tool_name = None
        self._completed_args = {}
        self._current_arg_key = None
        self._current_arg_is_string = False
        self._streamed_json_len = 0
        self.tool_calls_start_token = "<tool_calls>"
        self.tool_calls_end_token = "</tool_calls>"
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_sep_token = "<tool_sep>"
        self.arg_key_start_token = "<arg_key>"
        self.arg_key_end_token = "</arg_key>"
        self.arg_value_start_token = "<arg_value>"
        self.arg_value_end_token = "</arg_value>"
        self.tool_call_regex = re.compile(
            rf"{self.tool_call_start_token}(.*?){self.tool_sep_token}"
            rf"(.*?){self.tool_call_end_token}",
            re.DOTALL,
        )
        self.tool_call_portion_regex = re.compile(
            rf"{self.tool_call_start_token}(.*?){self.tool_sep_token}(.*)", re.DOTALL
        )
        self.func_args_regex = re.compile(
            rf"{self.arg_key_start_token}(.*?){self.arg_key_end_token}\s*"
            rf"{self.arg_value_start_token}(.*?){self.arg_value_end_token}",
            re.DOTALL,
        )
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self._buffer = ""
        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError("missing tool-call boundary tokens")


def test_tool_parser_expands_union_type_and_uses_suffixed_tokens():
    module = _module(
        patch_hy_v3_tool_parser.TARGET_MODULE,
        HYV3ToolParser=_FakeToolParser,
        re=re,
    )
    patch_hy_v3_tool_parser.apply(module)

    assert _FakeToolParser._get_schema_options(
        {"type": ["string", "object"]}
    ) == [{"type": "string"}, {"type": "object"}]
    scalar = {"type": "integer", "minimum": 1}
    assert _FakeToolParser._get_schema_options(scalar) == [scalar]
    assert _FakeToolParser._get_schema_options(
        {"anyOf": [{"type": "null"}, {"type": "string"}]}
    ) == [{"type": "null"}, {"type": "string"}]

    suffix = ":hcu"
    vocabulary = {
        patch_hy_v3_tool_parser._with_suffix(token, suffix): index
        for index, token in enumerate(patch_hy_v3_tool_parser._TOKENS, start=10)
    }
    tokenizer = _Tokenizer(vocabulary, suffix)
    parser = _FakeToolParser(tokenizer)
    assert parser.model_tokenizer is tokenizer
    assert parser.suffix == suffix
    assert parser.tool_calls_start_token == "<tool_calls:hcu>"
    assert parser.arg_value_end_token == "</arg_value:hcu>"
    assert parser.tool_calls_start_token_id == vocabulary["<tool_calls:hcu>"]
    assert parser.tool_calls_end_token_id == vocabulary["</tool_calls:hcu>"]
    match = parser.tool_call_regex.findall(
        "<tool_call:hcu>weather<tool_sep:hcu>"
        "<arg_key:hcu>city</arg_key:hcu>"
        "<arg_value:hcu>Wuhan</arg_value:hcu></tool_call:hcu>"
    )
    assert match == [
        (
            "weather",
            "<arg_key:hcu>city</arg_key:hcu>"
            "<arg_value:hcu>Wuhan</arg_value:hcu>",
        )
    ]


def test_tool_parser_preserves_unsuffixed_behavior():
    module = _module(
        patch_hy_v3_tool_parser.TARGET_MODULE,
        HYV3ToolParser=_FakeToolParser,
        re=re,
    )
    patch_hy_v3_tool_parser.apply(module)
    vocabulary = {
        token: index
        for index, token in enumerate(patch_hy_v3_tool_parser._TOKENS)
    }
    parser = _FakeToolParser(_Tokenizer(vocabulary))
    assert parser.suffix == ""
    assert parser.tool_calls_start_token == "<tool_calls>"


def test_apply_rejects_wrong_callback_module():
    with pytest.raises(PatchCompatibilityError, match="expected module"):
        patch_import_utils.apply(ModuleType("not.vllm.import_utils"))
