"""PCP provides the expert-parallel world for static-only HYV4 W4A8."""

from types import SimpleNamespace

import pytest

from tests.runtime_patch.test_glm52_pcp_config import (
    _make_pcp_config,
    _native_hyv4_mtp_config,
)
from vllm_hcu.patch.platform.core_fix import patch_parallel_config, patch_vllm_config


def _apply_parallel_config_patch(monkeypatch, module):
    cls = module.ParallelConfig
    current = cls._validate_parallel_config
    original = getattr(current, "__wrapped__", current)
    decorator = cls.__pydantic_decorators__.model_validators[
        "_validate_parallel_config"
    ]
    monkeypatch.setattr(cls, "_validate_parallel_config", original)
    monkeypatch.setattr(decorator, "func", original)
    monkeypatch.setattr(cls, patch_parallel_config._MARKER, False, raising=False)
    for name in (
        "__pydantic_core_schema__",
        "__pydantic_validator__",
        "__pydantic_serializer__",
        "__pydantic_complete__",
    ):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    assert patch_parallel_config.apply_to_module(module) is True


def config():
    value = _native_hyv4_mtp_config(
        _make_pcp_config, architecture="HYV4ForCausalLM", pp=1, tp=1, pcp=16
    )
    value.parallel_config.enable_eplb = True
    value.parallel_config.eplb_config = SimpleNamespace(num_redundant_experts=0)
    value.model_config.quantization = "compressed-tensors"
    value.additional_config["hcu"]["expert_map_path"] = "/tmp/static.json"
    return value


def test_exact_static_pcp16_native_mtp_contract_is_admitted():
    assert patch_vllm_config._validate_hcu_pcp_scope(config()) is True


@pytest.mark.parametrize(
    "path,value",
    [
        ("parallel_config.prefill_context_parallel_size", 8),
        ("parallel_config.tensor_parallel_size", 2),
        ("parallel_config.enable_eplb", False),
        ("parallel_config.eplb_config.num_redundant_experts", 16),
        ("model_config.quantization", "fp8"),
    ],
)
def test_other_static_modes_stay_closed(path, value):
    current = config()
    owner = current
    fields = path.split(".")
    for name in fields[:-1]:
        owner = getattr(owner, name)
    setattr(owner, fields[-1], value)
    with pytest.raises(ValueError):
        patch_vllm_config._validate_hcu_pcp_scope(current)


def test_dynamic_or_recording_pcp_eplb_is_still_rejected():
    for change in [
        {"expert_map_path": None},
        {"expert_map_record_path": "/tmp/record.json"},
    ]:
        current = config()
        current.additional_config["hcu"].update(change)
        with pytest.raises(ValueError):
            patch_vllm_config._validate_hcu_pcp_scope(current)


def test_upstream_parallel_config_counts_pcp_for_eplb(monkeypatch):
    import vllm.config.parallel as module

    monkeypatch.setattr(module.current_platform, "is_cuda_alike", lambda: True)
    _apply_parallel_config_patch(monkeypatch, module)
    eplb_config = module.EPLBConfig(num_redundant_experts=16)
    current = module.ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=16,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=eplb_config,
        all2all_backend="deepep_high_throughput",
        distributed_executor_backend="mp",
    )
    assert (
        current.enable_eplb
        and current.tensor_parallel_size == 1
        and current.data_parallel_size == 1
    )
    assert current.prefill_context_parallel_size == 16
    assert current.eplb_config.num_redundant_experts == 16
    assert current.eplb_config is eplb_config
    assert eplb_config.num_redundant_experts == 16


def test_upstream_single_rank_eplb_is_still_rejected(monkeypatch):
    import vllm.config.parallel as module

    monkeypatch.setattr(module.current_platform, "is_cuda_alike", lambda: True)
    _apply_parallel_config_patch(monkeypatch, module)
    with pytest.raises(ValueError, match="EPLB requires"):
        module.ParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=True,
            enable_eplb=True,
            distributed_executor_backend="mp",
        )


def test_upstream_pcp_eplb_still_requires_expert_parallel(monkeypatch):
    import vllm.config.parallel as module

    monkeypatch.setattr(module.current_platform, "is_cuda_alike", lambda: True)
    _apply_parallel_config_patch(monkeypatch, module)
    with pytest.raises(ValueError, match="enable_expert_parallel must be True"):
        module.ParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=16,
            enable_expert_parallel=False,
            enable_eplb=True,
            distributed_executor_backend="mp",
        )


def test_upstream_pcp_eplb_still_requires_cuda_like_platform(monkeypatch):
    import vllm.config.parallel as module

    monkeypatch.setattr(module.current_platform, "is_cuda_alike", lambda: False)
    _apply_parallel_config_patch(monkeypatch, module)
    with pytest.raises(ValueError, match="only supported on CUDA devices or ROCm"):
        module.ParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=16,
            enable_expert_parallel=True,
            enable_eplb=True,
            distributed_executor_backend="mp",
        )


def test_parallel_config_validator_supports_plain_dict(monkeypatch):
    import vllm.config.parallel as module

    _apply_parallel_config_patch(monkeypatch, module)
    current = module.ParallelConfig.__pydantic_validator__.validate_python({})
    assert isinstance(current, module.ParallelConfig)
    assert current.enable_eplb is False


def test_parallel_config_validator_rejects_invalid_parallel_sizes(monkeypatch):
    import vllm.config.parallel as module
    from pydantic_core import ArgsKwargs, ValidationError

    _apply_parallel_config_patch(monkeypatch, module)
    validator = module.ParallelConfig.__pydantic_validator__
    value = ArgsKwargs(
        (),
        {
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "prefill_context_parallel_size": "invalid",
            "enable_expert_parallel": True,
            "enable_eplb": True,
        },
    )
    with pytest.raises(ValidationError):
        validator.validate_python(value)


def test_pcp_eplb_preserves_communicator_initialization(monkeypatch):
    import vllm.config.parallel as module
    from vllm.distributed import nixl_utils

    monkeypatch.setattr(module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(nixl_utils, "is_nixl_available", lambda: False)
    _apply_parallel_config_patch(monkeypatch, module)
    current = module.ParallelConfig(
        prefill_context_parallel_size=16,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config={"num_redundant_experts": 16},
        distributed_executor_backend="mp",
    )
    assert current.eplb_config.communicator == "torch_gloo"


def test_pcp_eplb_rejects_negative_redundancy(monkeypatch):
    import vllm.config.parallel as module
    from pydantic_core import ValidationError

    monkeypatch.setattr(module.current_platform, "is_cuda_alike", lambda: True)
    _apply_parallel_config_patch(monkeypatch, module)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        module.ParallelConfig(
            prefill_context_parallel_size=16,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config={"num_redundant_experts": -1},
            distributed_executor_backend="mp",
        )


def test_pcp_eplb_preserves_remaining_parallel_checks(monkeypatch):
    import vllm.config.parallel as module

    monkeypatch.setattr(module.current_platform, "is_cuda_alike", lambda: True)
    _apply_parallel_config_patch(monkeypatch, module)
    eplb = module.EPLBConfig(num_redundant_experts=16)
    with pytest.raises(ValueError, match="must be divisible"):
        module.ParallelConfig(
            prefill_context_parallel_size=16,
            decode_context_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=eplb,
            distributed_executor_backend="mp",
        )
    assert eplb.num_redundant_experts == 16


def test_pcp_eplb_supports_pydantic_nested_dict_validation(monkeypatch):
    import vllm.config.parallel as module
    from pydantic import TypeAdapter

    monkeypatch.setattr(module.current_platform, "is_cuda_alike", lambda: True)
    _apply_parallel_config_patch(monkeypatch, module)
    current = TypeAdapter(module.ParallelConfig).validate_python(
        {
            "prefill_context_parallel_size": 16,
            "enable_expert_parallel": True,
            "enable_eplb": True,
            "eplb_config": {"num_redundant_experts": 16},
            "distributed_executor_backend": "mp",
        }
    )
    assert current.enable_eplb
    assert current.eplb_config.num_redundant_experts == 16


def test_parallel_config_validator_patch_is_idempotent(monkeypatch):
    import vllm.config.parallel as module

    _apply_parallel_config_patch(monkeypatch, module)
    validator = module.ParallelConfig.__pydantic_validator__
    assert patch_parallel_config.apply_to_module(module) is False
    assert module.ParallelConfig.__pydantic_validator__ is validator
