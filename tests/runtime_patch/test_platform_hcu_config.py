# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Configuration contracts for maintained HCU runtime paths.

Legacy custom FlashAttention plumbing remains in production code pending its
unified cleanup, but it is intentionally not asserted as a supported mode here.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import inspect
import json
import math
import multiprocessing
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm_hcu.model_executor.layers.quantization import slimquant_facade
from vllm_hcu.patch.config import HcuFeatureConfig, get_hcu_config
from vllm_hcu.patch.platform.core_fix import (
    patch_compilation_config,
    patch_engine_args,
    patch_hcu_config,
    patch_slimquant_registry,
    patch_vllm_config,
)
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError


REPO = Path(__file__).resolve().parents[2]
TARGET_VLLM_ROOT = Path(
    os.environ.get("VLLM_V0251_SOURCE_ROOT", REPO.parent / "vllm_0251")
).resolve()
if not (TARGET_VLLM_ROOT / "vllm" / "__init__.py").is_file():
    raise RuntimeError(
        f"VLLM_V0251_SOURCE_ROOT does not contain vllm: {TARGET_VLLM_ROOT}"
    )

_TARGET_SOURCE_ASSERTION = r'''
import os as _vllm_hcu_os
from pathlib import Path as _VllmHcuPath
import vllm as _vllm_hcu_target
_vllm_hcu_root = _VllmHcuPath(
    _vllm_hcu_os.environ["VLLM_V0251_SOURCE_ROOT"]
).resolve()
_vllm_hcu_file = _VllmHcuPath(_vllm_hcu_target.__file__).resolve()
assert _vllm_hcu_file.is_relative_to(_vllm_hcu_root), (
    f"vllm resolved outside target root: {_vllm_hcu_file} not under {_vllm_hcu_root}"
)
'''


def _run_fresh_v0251(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "__disabled__"
    env["VLLM_V0251_SOURCE_ROOT"] = str(TARGET_VLLM_ROOT)
    env["PYTHONPATH"] = os.pathsep.join((str(TARGET_VLLM_ROOT), str(REPO)))
    return subprocess.run(
        [sys.executable, "-c", _TARGET_SOURCE_ASSERTION + code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@dataclass
class _KernelConfig:
    moe_backend: str = "auto"


def _make_arg_utils_module() -> ModuleType:
    module = ModuleType(patch_engine_args.TARGET_MODULE)

    @dataclass
    class EngineArgs:
        additional_config: dict[str, Any] = field(default_factory=dict)
        attention_backend: str | None = None
        attention_config: dict[str, Any] = field(default_factory=dict)
        all2all_backend: str = "allgather_reducescatter"
        moe_backend: str = "auto"
        kernel_config: _KernelConfig | dict[str, Any] = field(
            default_factory=_KernelConfig
        )
        speculative_config: dict[str, Any] | None = None

        def __post_init__(self) -> None:
            if isinstance(self.kernel_config, dict):
                self.kernel_config = _KernelConfig(**self.kernel_config)

        def create_engine_config(
            self,
            usage_context: object | None = None,
            headless: bool = False,
        ) -> object:
            del usage_context, headless
            return SimpleNamespace(
                additional_config=copy.deepcopy(self.additional_config),
                kernel_config=copy.deepcopy(self.kernel_config),
                parallel_config=SimpleNamespace(
                    all2all_backend=self.all2all_backend
                ),
            )

        @staticmethod
        def add_cli_args(parser: argparse.ArgumentParser):
            parser.add_argument(
                "--all2all-backend",
                choices=("allgather_reducescatter", "deepep_low_latency"),
                default="allgather_reducescatter",
            )
            return parser

        @classmethod
        def from_cli_args(cls, args: argparse.Namespace):
            attrs = [item.name for item in dataclasses.fields(cls)]
            return cls(
                **{
                    name: getattr(args, name)
                    for name in attrs
                    if hasattr(args, name)
                }
            )

    @dataclass
    class AsyncEngineArgs(EngineArgs):
        enable_log_requests: bool = False

    module.EngineArgs = EngineArgs
    module.AsyncEngineArgs = AsyncEngineArgs
    return module


def _child_sidecar(payload: object, queue: multiprocessing.Queue) -> None:
    queue.put(get_hcu_config(payload).to_dict())


def test_engine_args_legacy_keywords_are_removed_before_official_init() -> None:
    module = _make_arg_utils_module()
    original_signature = inspect.signature(module.EngineArgs.__init__)
    assert patch_engine_args.apply_to_module(module)

    args = module.EngineArgs(
        enable_lightly_cp=True,
        enable_lightly_cplb=True,
        enable_custom_sp=True,
        enable_multi_layers_mtp=True,
        moe_backend="dpsk_deep_gemm",
    )
    assert args.moe_backend == "auto"
    assert args.kernel_config.moe_backend == "auto"
    assert get_hcu_config(args) == HcuFeatureConfig(
        enable_lightly_cp=True,
        enable_lightly_cplb=True,
        enable_custom_sp=True,
        enable_multi_layers_mtp=True,
        moe_backend="dpsk_deep_gemm",
    )
    assert inspect.signature(module.EngineArgs.__init__) == original_signature
    assert {item.name for item in fields(module.EngineArgs)} == {
        "additional_config",
        "attention_backend",
        "attention_config",
        "all2all_backend",
        "moe_backend",
        "kernel_config",
        "speculative_config",
    }

    config = args.create_engine_config()
    assert get_hcu_config(config) == get_hcu_config(args)


def test_engine_args_normalizes_deepep_auto_and_extends_cli_choice() -> None:
    module = _make_arg_utils_module()
    patch_engine_args.apply_to_module(module)

    args = module.EngineArgs(all2all_backend="deepep_auto")
    assert args.all2all_backend == "deepep_low_latency"
    assert get_hcu_config(args).deepep_auto is True
    config = args.create_engine_config()
    assert config.parallel_config.all2all_backend == "deepep_low_latency"
    assert get_hcu_config(config).deepep_auto is True

    parser = module.EngineArgs.add_cli_args(argparse.ArgumentParser())
    parsed = parser.parse_args(["--all2all-backend", "deepep_auto"])
    assert parsed.all2all_backend == "deepep_auto"


@pytest.mark.parametrize(
    ("alias", "mode"),
    [
        ("FLASH_ATTN_CLASSIC", "classic"),
        ("flash_attn_cutlass", "cutlass"),
        ("flash_attn_varlen", "varlen"),
    ],
)
def test_engine_args_normalizes_hcu_flash_attention_aliases(
    alias: str, mode: str
) -> None:
    module = _make_arg_utils_module()
    patch_engine_args.apply_to_module(module)

    top_level = module.EngineArgs(attention_backend=alias)
    assert top_level.attention_backend == "FLASH_ATTN"
    assert get_hcu_config(top_level).hcu_flash_attn_mode == mode
    assert get_hcu_config(top_level.create_engine_config()).hcu_flash_attn_mode == mode

    nested = module.EngineArgs(attention_config={"backend": alias})
    assert nested.attention_config["backend"] == "FLASH_ATTN"
    assert get_hcu_config(nested).hcu_flash_attn_mode == mode


def test_async_engine_args_and_nested_dpsk_config_use_same_sidecar() -> None:
    module = _make_arg_utils_module()
    patch_engine_args.apply_to_module(module)
    args = module.AsyncEngineArgs(
        kernel_config={"moe_backend": "dpsk_deep_gemm"},
        enable_custom_sp=True,
    )
    assert args.kernel_config.moe_backend == "auto"
    assert get_hcu_config(args).moe_backend == "dpsk_deep_gemm"
    assert get_hcu_config(args).enable_custom_sp is True


def test_nested_speculative_multi_mtp_is_extracted_before_official_config() -> None:
    module = _make_arg_utils_module()
    patch_engine_args.apply_to_module(module)
    args = module.EngineArgs(
        speculative_config={
            "method": "mtp",
            "enable_multi_layers_mtp": True,
        }
    )
    assert args.speculative_config == {"method": "mtp"}
    assert get_hcu_config(args).enable_multi_layers_mtp is True
    assert pickle.loads(pickle.dumps(args.additional_config)) == args.additional_config

    args.speculative_config = {"enable_multi_layers_mtp": False}
    with pytest.raises(ValueError, match="sidecar and speculative_config"):
        args.create_engine_config()

    with pytest.raises(ValueError, match="conflicting enable_multi_layers_mtp"):
        module.EngineArgs(
            enable_multi_layers_mtp=False,
            speculative_config={"enable_multi_layers_mtp": True},
        )
    with pytest.raises(ValueError, match="sidecar and speculative_config"):
        module.EngineArgs(
            additional_config={
                "hcu": HcuFeatureConfig(
                    enable_multi_layers_mtp=False
                ).to_dict()
            },
            speculative_config={"enable_multi_layers_mtp": True},
        )


def test_positional_additional_config_is_merged_not_overwritten() -> None:
    module = _make_arg_utils_module()
    patch_engine_args.apply_to_module(module)
    original_additional = {
        "unrelated": {"keep": True},
        "hcu": HcuFeatureConfig(
            enable_lightly_cp=True,
            enable_custom_sp=True,
        ).to_dict(),
    }
    args = module.EngineArgs(original_additional, enable_lightly_cp=True)
    assert args.additional_config["unrelated"] == {"keep": True}
    assert get_hcu_config(args) == HcuFeatureConfig(
        enable_lightly_cp=True,
        enable_custom_sp=True,
    )


def test_engine_args_rejects_incompatible_target_signature() -> None:
    module = ModuleType(patch_engine_args.TARGET_MODULE)

    class BadEngineArgs:
        def __init__(self, additional_config=None, moe_backend="auto") -> None:
            pass

    module.EngineArgs = BadEngineArgs
    module.AsyncEngineArgs = BadEngineArgs
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_engine_args.apply_to_module(module)


def test_cli_registration_is_idempotent_and_dpsk_never_reaches_schema() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-backend", choices=["auto", "triton"], default="auto")
    patch_hcu_config.register_hcu_cli_args(parser)
    patch_hcu_config.register_hcu_cli_args(parser)

    parsed = vars(
        parser.parse_args(
            [
                "--enable-lightly-cp",
                "--enable-lightly-cplb",
                "--enable-custom-sp",
                "--moe-backend",
                "dpsk_deep_gemm",
            ]
        )
    )
    assert sum(action.dest == "enable_lightly_cp" for action in parser._actions) == 1

    module = _make_arg_utils_module()
    patch_engine_args.apply_to_module(module)
    args = module.EngineArgs.from_cli_args(argparse.Namespace(**parsed))
    assert args.moe_backend == "auto"
    assert get_hcu_config(args).moe_backend == "dpsk_deep_gemm"


def test_cli_registration_reports_conflicting_destination() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-backend", choices=["auto"])
    parser.add_argument("--enable-lightly-cp", type=str)
    with pytest.raises(PatchCompatibilityError, match="incompatible semantics"):
        patch_hcu_config.register_hcu_cli_args(parser)


def test_cli_omission_preserves_sidecar_and_explicit_flag_overrides() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-backend", choices=["auto"], default="auto")
    parser.add_argument("--additional-config", type=json.loads, default={})
    patch_hcu_config.register_hcu_cli_args(parser)
    module = _make_arg_utils_module()
    patch_engine_args.apply_to_module(module)

    sidecar_true = json.dumps(
        {"hcu": HcuFeatureConfig(enable_custom_sp=True).to_dict()}
    )
    omitted = module.EngineArgs.from_cli_args(
        parser.parse_args(["--additional-config", sidecar_true])
    )
    assert get_hcu_config(omitted).enable_custom_sp is True

    sidecar_false = json.dumps({"hcu": HcuFeatureConfig().to_dict()})
    explicit = module.EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--additional-config",
                sidecar_false,
                "--enable-custom-sp",
            ]
        )
    )
    assert get_hcu_config(explicit).enable_custom_sp is True


def _make_compilation_module() -> ModuleType:
    module = ModuleType(patch_compilation_config.TARGET_MODULE)

    class _CUDAGraphMode:
        def __init__(self, piecewise: bool = True) -> None:
            self.piecewise = piecewise

        def has_piecewise_cudagraphs(self) -> bool:
            return self.piecewise

    class CompilationConfig:
        def __init__(self) -> None:
            self.pass_config = SimpleNamespace(enable_sp=False)
            self.calls = 0
            self.sp_observed = False
            self.splitting_calls = 0
            self.splitting_ops = ["vllm::unified_mla_attention_with_output"]
            self.use_inductor_graph_partition = False
            self.cudagraph_mode = _CUDAGraphMode()

        def adjust_cudagraph_sizes_for_spec_decode(
            self,
            uniform_decode_query_len: int,
            tensor_parallel_size: int,
        ) -> tuple[int, int]:
            self.calls += 1
            self.sp_observed = self.pass_config.enable_sp
            return uniform_decode_query_len, tensor_parallel_size

        def set_splitting_ops_for_v1(
            self,
            all2all_backend: str,
            data_parallel_size: int = 1,
        ) -> tuple[str, int]:
            self.splitting_calls += 1
            return all2all_backend, data_parallel_size

    module.CompilationConfig = CompilationConfig
    return module


def test_compilation_custom_sp_adapter_preserves_feature_off_path() -> None:
    module = _make_compilation_module()
    patch_compilation_config.apply_to_module(module)
    config = module.CompilationConfig()

    assert config.adjust_cudagraph_sizes_for_spec_decode(4, 8) == (4, 8)
    assert config.calls == 1
    assert config.sp_observed is False
    assert config.pass_config.enable_sp is False

    vllm_config = SimpleNamespace(
        additional_config={
            "hcu": HcuFeatureConfig(enable_custom_sp=True).to_dict()
        },
        compilation_config=config,
    )
    patch_compilation_config.bind_hcu_config(vllm_config)
    assert config.adjust_cudagraph_sizes_for_spec_decode(4, 8) == (4, 8)
    assert config.calls == 2
    assert config.sp_observed is True
    assert config.pass_config.enable_sp is False


def test_compilation_adapter_splits_hcu_sparse_indexer_from_piecewise_graph() -> None:
    module = _make_compilation_module()
    patch_compilation_config.apply_to_module(module)
    config = module.CompilationConfig()

    assert config.set_splitting_ops_for_v1("allgather_reducescatter", 8) == (
        "allgather_reducescatter",
        8,
    )
    assert config.splitting_calls == 1
    assert config.splitting_ops[-1] == "vllm::hcu_sparse_attn_indexer"

    config.set_splitting_ops_for_v1("allgather_reducescatter", 8)
    assert config.splitting_ops.count("vllm::hcu_sparse_attn_indexer") == 1


def test_compilation_adapter_defers_to_inductor_unsafe_tags() -> None:
    module = _make_compilation_module()
    patch_compilation_config.apply_to_module(module)
    config = module.CompilationConfig()
    config.use_inductor_graph_partition = True

    config.set_splitting_ops_for_v1("allgather_reducescatter")
    assert "vllm::hcu_sparse_attn_indexer" not in config.splitting_ops


def test_compilation_adapter_skips_non_piecewise_cudagraphs() -> None:
    module = _make_compilation_module()
    patch_compilation_config.apply_to_module(module)
    config = module.CompilationConfig()
    config.cudagraph_mode.piecewise = False

    config.set_splitting_ops_for_v1("allgather_reducescatter")
    assert "vllm::hcu_sparse_attn_indexer" not in config.splitting_ops


def test_compilation_adapter_requires_finalized_splitting_ops_list() -> None:
    module = _make_compilation_module()
    patch_compilation_config.apply_to_module(module)
    config = module.CompilationConfig()
    config.splitting_ops = None

    with pytest.raises(PatchCompatibilityError, match="finalized as a list"):
        config.set_splitting_ops_for_v1("allgather_reducescatter")


def test_compilation_adapter_rejects_adjust_signature_drift() -> None:
    module = ModuleType(patch_compilation_config.TARGET_MODULE)

    class CompilationConfig:
        def adjust_cudagraph_sizes_for_spec_decode(self, query_len: int) -> None:
            pass

        def set_splitting_ops_for_v1(
            self,
            all2all_backend: str,
            data_parallel_size: int = 1,
        ) -> None:
            pass

    module.CompilationConfig = CompilationConfig
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_compilation_config.apply_to_module(module)


def test_compilation_adapter_rejects_splitting_signature_drift() -> None:
    module = ModuleType(patch_compilation_config.TARGET_MODULE)

    class CompilationConfig:
        def adjust_cudagraph_sizes_for_spec_decode(
            self,
            uniform_decode_query_len: int,
            tensor_parallel_size: int,
        ) -> None:
            pass

        def set_splitting_ops_for_v1(self, all2all_backend: str) -> None:
            pass

    module.CompilationConfig = CompilationConfig
    with pytest.raises(
        PatchCompatibilityError,
        match="set_splitting_ops_for_v1.*incompatible signature",
    ):
        patch_compilation_config.apply_to_module(module)


class _FakeHFConfig:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_text_config(self) -> object:
        return SimpleNamespace(name=self.text)


class _FakeModelConfig:
    def __init__(self, hf_config: _FakeHFConfig, *, enforce_eager: bool = True) -> None:
        self.hf_config = hf_config
        self.hf_text_config = hf_config.get_text_config()
        self.model_arch_config = self.get_model_arch_config()
        self.enforce_eager = enforce_eager
        self.use_mla = False

    def get_model_arch_config(self) -> str:
        return self.hf_text_config.name

    def verify_with_parallel_config(self, parallel_config: object) -> object:
        return parallel_config


class _FakeCompilationConfig:
    def __init__(self, sizes: list[int] | None = None) -> None:
        self.cudagraph_capture_sizes = sizes
        self.max_cudagraph_capture_size = None
        self.compile_sizes: list[int | str] | None = [
            "cudagraph_capture_sizes",
            999,
        ]
        self.post_init_calls = 0

    def post_init_cudagraph_sizes(self) -> None:
        self.post_init_calls += 1
        computed: list[int] = []
        for value in self.compile_sizes or []:
            if value == "cudagraph_capture_sizes":
                computed.extend(self.cudagraph_capture_sizes or [])
            else:
                computed.append(value)  # type: ignore[arg-type]
        self.compile_sizes = computed


def _make_vllm_module() -> ModuleType:
    module = ModuleType(patch_vllm_config.TARGET_MODULE)

    class FakeVllmConfig:
        def __init__(self, sizes: list[int] | None = None) -> None:
            self.additional_config: dict[str, Any] = {}
            self.model_config = _FakeModelConfig(_FakeHFConfig("old"))
            self.compilation_config = _FakeCompilationConfig(sizes)
            self.scheduler_config = SimpleNamespace(max_num_batched_tokens=64)
            self.speculative_config = SimpleNamespace(num_speculative_tokens=3)

        def with_hf_config(
            self,
            hf_config: _FakeHFConfig,
            architectures: list[str] | None = None,
        ) -> "FakeVllmConfig":
            del architectures
            updated = copy.copy(self)
            updated.model_config = copy.copy(self.model_config)
            updated.model_config.hf_config = hf_config
            # This deliberately emulates the stale upstream order.
            updated.model_config.model_arch_config = (
                updated.model_config.get_model_arch_config()
            )
            return updated

        def _set_cudagraph_sizes(self) -> str:
            if self.compilation_config.cudagraph_capture_sizes is None:
                self.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8, 16, 64]
            self.compilation_config.max_cudagraph_capture_size = max(
                self.compilation_config.cudagraph_capture_sizes
            )
            self.compilation_config.post_init_cudagraph_sizes()
            return "upstream-result"

        def _get_v2_model_runner_unsupported_features(self) -> list[str]:
            return []

        def _validate_v2_model_runner(self) -> None:
            return None

    module.VllmConfig = FakeVllmConfig
    module.ModelConfig = _FakeModelConfig
    return module


def test_vllm_adapter_refreshes_hf_text_config_and_arch_config() -> None:
    module = _make_vllm_module()
    patch_vllm_config.apply_to_module(module)
    config = module.VllmConfig()
    updated = config.with_hf_config(_FakeHFConfig("new"))
    assert updated.model_config.hf_text_config.name == "new"
    assert updated.model_config.model_arch_config == "new"


def test_request_cudagraph_buckets_and_feature_off_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _make_vllm_module()
    patch_vllm_config.apply_to_module(module)

    monkeypatch.setattr(
        patch_vllm_config, "_request_cudagraph_buckets_enabled", lambda: False
    )
    feature_off = module.VllmConfig()
    assert feature_off._set_cudagraph_sizes() == "upstream-result"
    assert feature_off.compilation_config.cudagraph_capture_sizes == [
        1,
        2,
        4,
        8,
        16,
        64,
    ]
    assert feature_off.compilation_config.post_init_calls == 1

    monkeypatch.setattr(
        patch_vllm_config, "_request_cudagraph_buckets_enabled", lambda: True
    )
    enabled = module.VllmConfig()
    assert enabled._set_cudagraph_sizes() == "upstream-result"
    assert enabled.compilation_config.cudagraph_capture_sizes == [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        40,
        48,
        56,
        64,
    ]
    assert enabled.compilation_config.compile_sizes == [
        *enabled.compilation_config.cudagraph_capture_sizes,
        999,
    ]
    assert enabled.compilation_config.post_init_calls == 2

    explicit = module.VllmConfig([2, 6])
    explicit._set_cudagraph_sizes()
    assert explicit.compilation_config.cudagraph_capture_sizes == [2, 6]
    assert explicit.compilation_config.post_init_calls == 1


def test_real_v0251_set_cudagraph_binds_custom_sp_before_first_adjustment() -> None:
    result = _run_fresh_v0251(
        "import json; from types import SimpleNamespace; "
        "import vllm.config.compilation as compilation_module; "
        "import vllm.config.vllm as vllm_module; "
        "from vllm.config.compilation import CUDAGraphMode,CompilationConfig; "
        "from vllm.v1.attention.backend import AttentionCGSupport; "
        "from vllm_hcu.patch.config import HcuFeatureConfig; "
        "from vllm_hcu.patch.platform.core_fix import ("
        "patch_compilation_config,patch_vllm_config); "
        "patch_compilation_config.apply(compilation_module); "
        "patch_vllm_config.apply(vllm_module); "
        "sizes=[1,2,3,4,5,6,7,8,9,10,12,16]; "
        "make=lambda enabled:object.__new__(vllm_module.VllmConfig); "
        "enabled=make(True); "
        "enabled.model_config=SimpleNamespace(enforce_eager=False); "
        "enabled.compilation_config=CompilationConfig(cudagraph_mode="
        "CUDAGraphMode.FULL,cudagraph_capture_sizes=list(sizes),"
        "max_cudagraph_capture_size=16); "
        "enabled.parallel_config=SimpleNamespace(tensor_parallel_size=4); "
        "enabled.scheduler_config=SimpleNamespace(max_num_seqs=8,"
        "max_num_batched_tokens=64); "
        "enabled.speculative_config=SimpleNamespace(num_speculative_tokens=1); "
        "enabled.performance_mode='balanced'; "
        "enabled.additional_config={'hcu':HcuFeatureConfig("
        "enable_custom_sp=True).to_dict()}; enabled._set_cudagraph_sizes(); "
        "initial_enabled=list(enabled.compilation_config.cudagraph_capture_sizes); "
        "enabled_mode=enabled.compilation_config."
        "resolve_cudagraph_mode_and_sizes(AttentionCGSupport.ALWAYS,None,"
        "uniform_decode_query_len=2,use_v2_model_runner=False,"
        "tensor_parallel_size=4); "
        "disabled=make(False); "
        "disabled.model_config=SimpleNamespace(enforce_eager=False); "
        "disabled.compilation_config=CompilationConfig(cudagraph_mode="
        "CUDAGraphMode.FULL,cudagraph_capture_sizes=list(sizes),"
        "max_cudagraph_capture_size=16); "
        "disabled.parallel_config=SimpleNamespace(tensor_parallel_size=4); "
        "disabled.scheduler_config=SimpleNamespace(max_num_seqs=8,"
        "max_num_batched_tokens=64); "
        "disabled.speculative_config=SimpleNamespace(num_speculative_tokens=1); "
        "disabled.performance_mode='balanced'; "
        "disabled.additional_config={'hcu':HcuFeatureConfig().to_dict()}; "
        "disabled._set_cudagraph_sizes(); "
        "initial_disabled=list(disabled.compilation_config.cudagraph_capture_sizes); "
        "disabled_mode=disabled.compilation_config."
        "resolve_cudagraph_mode_and_sizes(AttentionCGSupport.ALWAYS,None,"
        "uniform_decode_query_len=2,use_v2_model_runner=False,"
        "tensor_parallel_size=4); "
        "print(json.dumps({'initial_enabled':initial_enabled,"
        "'initial_disabled':initial_disabled,'enabled':enabled."
        "compilation_config.cudagraph_capture_sizes,'disabled':disabled."
        "compilation_config.cudagraph_capture_sizes,'enabled_sp_after':enabled."
        "compilation_config.pass_config.enable_sp,'disabled_sp_after':disabled."
        "compilation_config.pass_config.enable_sp,'enabled_mode':enabled_mode.name,"
        "'disabled_mode':disabled_mode.name}))"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "initial_enabled": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16],
        "initial_disabled": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16],
        "enabled": [4, 8, 12, 16],
        "disabled": [2, 4, 6, 8, 10, 12, 16],
        "enabled_sp_after": False,
        "disabled_sp_after": None,
        "enabled_mode": "FULL",
        "disabled_mode": "FULL",
    }


class _ValidationCompilation:
    pass


def _validation_config(feature_config: HcuFeatureConfig) -> object:
    return SimpleNamespace(
        additional_config={"hcu": feature_config.to_dict()},
        compilation_config=_ValidationCompilation(),
        model_config=SimpleNamespace(enforce_eager=True, use_mla=False),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        kv_transfer_config=None,
    )


def test_hcu_config_validation_binds_sidecar_without_upstream_fields() -> None:
    feature_config = HcuFeatureConfig(
        enable_lightly_cp=True,
        enable_lightly_cplb=True,
        enable_custom_sp=True,
        enable_multi_layers_mtp=True,
        moe_backend="dpsk_deep_gemm",
        hcu_flash_attn_mode="cutlass",
    )
    config = _validation_config(feature_config)
    assert patch_vllm_config.validate_and_update_hcu_config(config) == feature_config
    # CompilationConfig has no duplicate serialized sidecar; the process-local
    # binding is derived again from authoritative additional_config after IPC.
    assert "_vllm_hcu_feature_config" not in vars(config.compilation_config)
    assert config.kernel_config.moe_backend == "auto"
    assert not hasattr(config.parallel_config, "enable_lightly_cp")

    config.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="only supports eager"):
        patch_vllm_config.validate_and_update_hcu_config(config)
    config.model_config.enforce_eager = True
    config.parallel_config.decode_context_parallel_size = 2
    with pytest.raises(ValueError, match="DCP"):
        patch_vllm_config.validate_and_update_hcu_config(config)


@pytest.mark.parametrize("flash_attn_mode", ["cutlass", "varlen"])
def test_compilation_binding_is_recreated_after_pickle(
    flash_attn_mode: str,
) -> None:
    feature_config = HcuFeatureConfig(
        enable_custom_sp=True,
        hcu_flash_attn_mode=flash_attn_mode,
    )
    config = _validation_config(feature_config)
    patch_vllm_config.validate_and_update_hcu_config(config)
    restored = pickle.loads(pickle.dumps(config))
    assert "_vllm_hcu_feature_config" not in vars(restored.compilation_config)
    assert get_hcu_config(restored) == feature_config
    assert patch_vllm_config.validate_and_update_hcu_config(restored) == feature_config


def _make_quantization_module() -> ModuleType:
    module = ModuleType(patch_slimquant_registry.TARGET_MODULE)
    module.QUANTIZATION_METHODS = []
    module._CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}

    def register(name: str):
        def decorator(config_cls: type[QuantizationConfig]):
            module.QUANTIZATION_METHODS.append(name)
            module._CUSTOMIZED_METHOD_TO_QUANT_CONFIG[name] = config_cls
            return config_cls

        return decorator

    module.register_quantization_config = register
    return module


def test_slimquant_uses_public_registry_without_loading_concrete_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _make_quantization_module()

    def forbidden_import(name: str) -> ModuleType:
        raise AssertionError(f"concrete import during registration: {name}")

    monkeypatch.setattr(slimquant_facade.importlib, "import_module", forbidden_import)
    assert patch_slimquant_registry.apply_to_module(module)
    assert module.QUANTIZATION_METHODS == [
        "slimquant_marlin",
        "slimquant_compressed_tensors_marlin",
        "slimquant_w4a8",
    ]
    assert not patch_slimquant_registry.apply_to_module(module)

    facade = module._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["slimquant_w4a8"]
    assert (
        facade.override_quantization_method(
            {"quant_method": "slimquant_w4a8"}, None, hf_config=object()
        )
        == "slimquant_w4a8"
    )
    with pytest.raises(AssertionError, match="concrete import"):
        facade.get_supported_act_dtypes()


def test_slimquant_registry_rejects_provider_conflict() -> None:
    module = _make_quantization_module()
    module.QUANTIZATION_METHODS.append("slimquant_w4a8")
    module._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["slimquant_w4a8"] = object
    with pytest.raises(PatchCompatibilityError, match="already registered"):
        patch_slimquant_registry.apply_to_module(module)


def test_slimquant_marlin_inherits_v0251_compressed_tensors_constructor() -> None:
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors.compressed_tensors_marlin import (
        SlimQuantCompressedTensorsMarlinConfig,
    )

    assert inspect.signature(
        SlimQuantCompressedTensorsMarlinConfig.__init__
    ) == inspect.signature(CompressedTensorsConfig.__init__)
    config = SlimQuantCompressedTensorsMarlinConfig.from_config(
        {
            "config_groups": {},
            "format": "int-quantized",
            "ignore": [],
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
        }
    )
    assert config.target_scheme_map == {}
    assert config.ignore == []
    assert config.quant_format == "int-quantized"

    # v0.25.1's FusedMoE public symbol is a factory and may be wrapped by HCU;
    # quantization dispatch must use the target-owned RoutedExperts type.
    source = Path(
        "vllm_hcu/model_executor/layers/quantization/compressed_tensors/"
        "compressed_tensors_marlin.py"
    ).read_text(encoding="utf-8-sig")
    assert "isinstance(layer, RoutedExperts)" in source
    assert "isinstance(layer, FusedMoE)" not in source
    assert config.get_quant_method(torch.nn.Embedding(4, 4), "embed") is None

    from vllm_hcu.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe_marlin import (
        CompressedTensorsW8A8FP8MarlinMoEMethod,
        CompressedTensorsW8A8Int8MarlinMoEMethod,
    )

    target_prefix = (
        "self",
        "layer",
        "x",
        "topk_weights",
        "topk_ids",
        "shared_experts",
        "shared_experts_input",
    )
    for method in (
        CompressedTensorsW8A8FP8MarlinMoEMethod.apply,
        CompressedTensorsW8A8Int8MarlinMoEMethod.apply,
    ):
        parameters = tuple(inspect.signature(method).parameters)
        assert parameters[: len(target_prefix)] == target_prefix
        assert parameters[len(target_prefix) :] == ("i_q", "i_s")


def test_slimquant_fp8_moe_repack_preserves_fp8_without_widening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as moe_marlin,
    )

    method = object.__new__(
        moe_marlin.CompressedTensorsW8A8FP8MarlinMoEMethod
    )
    method.use_deepep = False

    def fp8_parameter(shape: tuple[int, ...]) -> torch.nn.Parameter:
        values = torch.linspace(-1.0, 1.0, math.prod(shape)).reshape(shape)
        return torch.nn.Parameter(
            values.to(torch.float8_e4m3fn), requires_grad=False
        )

    layer = SimpleNamespace(
        w13_weight=fp8_parameter((2, 128, 64)),
        w2_weight=fp8_parameter((2, 64, 64)),
    )
    expected_w13 = torch.stack(
        [
            moe_marlin.get_w8a8_int8_marlin_weights(weight)
            for weight in layer.w13_weight
        ]
    )
    expected_w2 = torch.stack(
        [
            moe_marlin.get_w8a8_int8_marlin_weights(weight)
            for weight in layer.w2_weight
        ]
    )

    def reject_widening(_tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("FP8 checkpoint weights must not widen through FP32")

    def reject_per_expert_stack(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise AssertionError("FP8 MoE weights must be repacked as a full tensor")

    monkeypatch.setattr(
        moe_marlin, "fp32_to_fp8_e4m3fn", reject_widening
    )
    monkeypatch.setattr(torch, "stack", reject_per_expert_stack)

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.dtype == torch.float8_e4m3fn
    assert layer.w2_weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(layer.w13_weight.float(), expected_w13.float())
    torch.testing.assert_close(layer.w2_weight.float(), expected_w2.float())


def test_slimquant_marlin_repack_supports_full_expert_tensor() -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as moe_marlin,
    )

    weight = torch.arange(16, dtype=torch.int8).reshape(2, 2, 4)

    actual = moe_marlin.get_w8a8_int8_marlin_weights(weight, k_tile=2)

    expected = torch.tensor(
        [
            [[0, 1, 4, 5], [2, 3, 6, 7]],
            [[8, 9, 12, 13], [10, 11, 14, 15]],
        ],
        dtype=torch.int8,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(4,), (1, 2, 3, 4)])
def test_slimquant_marlin_repack_rejects_non_matrix_weights(
    shape: tuple[int, ...],
) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as moe_marlin,
    )

    with pytest.raises(ValueError, match="Expected 2D or 3D weight"):
        moe_marlin.get_w8a8_int8_marlin_weights(torch.empty(shape))


def test_slimquant_marlin_repack_rejects_unaligned_k_dimension() -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe_marlin as moe_marlin,
    )

    with pytest.raises(AssertionError, match="K dimension must be divisible"):
        moe_marlin.get_w8a8_int8_marlin_weights(
            torch.empty((2, 3, 5)), k_tile=4
        )


class _HashableConfig:
    def compute_hash(self) -> str:
        return "fixed"


def _vllm_hash(additional_config: dict[str, Any]) -> str:
    config = SimpleNamespace(
        model_config=None,
        cache_config=None,
        parallel_config=None,
        scheduler_config=None,
        device_config=None,
        load_config=None,
        offload_config=None,
        attention_config=None,
        lora_config=None,
        speculative_config=None,
        structured_outputs_config=None,
        profiler_config=None,
        observability_config=_HashableConfig(),
        quant_config=None,
        compilation_config=None,
        kernel_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        additional_config=additional_config,
    )
    return VllmConfig.compute_hash(config)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, "cutlass"),
        ({"VLLM_HCU_USE_FLASH_ATTN": "1"}, "classic"),
        ({"VLLM_HCU_USE_FLASH_ATTN_UNIFIED": "1"}, "cutlass"),
        ({"VLLM_HCU_USE_FLASH_ATTN_VARLEN": "1"}, "varlen"),
    ],
)
def test_hcu_flash_attention_mode_is_finalized_before_config_hash(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected: str,
) -> None:
    for name in (
        "VLLM_HCU_USE_FLASH_ATTN",
        "VLLM_HCU_USE_FLASH_ATTN_UNIFIED",
        "VLLM_HCU_USE_FLASH_ATTN_VARLEN",
        "VLLM_HCU_USE_CUSTOM_FLASH_ATTN",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    config = _validation_config(HcuFeatureConfig())
    feature_config = patch_vllm_config.validate_and_update_hcu_config(config)

    assert feature_config.hcu_flash_attn_mode == expected
    assert get_hcu_config(config) == feature_config


def test_varlen_flash_attention_uses_64_token_cache_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm_hcu.platforms import envs as hcu_envs
    from vllm_hcu.platforms.hcu import HCUPlatform

    class _NoFullGraphs:
        @staticmethod
        def has_full_cudagraphs() -> bool:
            return False

    monkeypatch.setattr(hcu_envs, "VLLM_HCU_USE_PD_SPLIT", False)
    monkeypatch.setattr(
        hcu_envs,
        "VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE",
        128,
    )
    monkeypatch.setattr(hcu_envs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    config = _validation_config(
        HcuFeatureConfig(hcu_flash_attn_mode="varlen")
    )
    config.compilation_config.cudagraph_mode = _NoFullGraphs()
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.distributed_executor_backend = "uni"
    config.parallel_config.worker_cls = "auto"
    config.cache_config = SimpleNamespace(
        user_specified_block_size=False,
        block_size=None,
    )
    config.attention_config = SimpleNamespace(
        backend=AttentionBackendEnum.FLASH_ATTN
    )

    HCUPlatform.check_and_update_config(config)

    assert config.cache_config.block_size == 64


def test_cutlass_block_first_mooncake_defers_to_worker_capability_gates() -> None:
    config = _validation_config(HcuFeatureConfig(hcu_flash_attn_mode="cutlass"))
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="MooncakeConnector",
        kv_connector_extra_config={},
    )

    assert patch_vllm_config.validate_and_update_hcu_config(config) == (
        HcuFeatureConfig(hcu_flash_attn_mode="cutlass")
    )


def test_sidecar_changes_upstream_hash_and_crosses_serialization_boundaries() -> None:
    disabled = {"hcu": HcuFeatureConfig().to_dict()}
    enabled = {"hcu": HcuFeatureConfig(enable_custom_sp=True).to_dict()}
    assert _vllm_hash(disabled) != _vllm_hash(enabled)

    assert json.loads(json.dumps(enabled)) == enabled
    assert pickle.loads(pickle.dumps(enabled)) == enabled

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_child_sidecar,
        args=(SimpleNamespace(additional_config=enabled), queue),
    )
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 0
    assert queue.get(timeout=5) == enabled["hcu"]
