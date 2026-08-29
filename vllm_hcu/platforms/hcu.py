# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import importlib
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_hcu import _ensure_platform_plugin_ready

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig

import vllm_hcu.platforms.envs as henvs 

logger = init_logger(__name__)

_ensure_platform_plugin_ready()


def get_hcu_flash_attn_mode() -> str:
    """Resolve the HCU flash-attention sub-mode from serialized config."""

    try:
        from vllm.config import get_current_vllm_config_or_none
    except ImportError:
        # This module can be imported before vLLM finishes initializing its
        # config module.  Do not cache that temporary flagless default.
        return henvs.resolve_hcu_flash_attn_mode(None)

    from vllm_hcu.patch.config import get_hcu_config

    config = get_current_vllm_config_or_none()
    explicit_mode = (
        None if config is None else get_hcu_config(config).hcu_flash_attn_mode
    )
    # Do not cache the flagless default: VllmConfig may become available
    # after this early platform module is imported.
    return henvs.resolve_hcu_flash_attn_mode(explicit_mode)

@cache
def _load_hcu_management_api() -> ModuleType | None:
    """Load the optional device-management API without exposing vendor errors."""

    try:
        return importlib.import_module("amdsmi")
    except Exception:
        return None

try:
    import vllm._C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._C with %r", e)


@cache
def flash_attn_triton_available() -> bool:
    try:
        from importlib.util import find_spec

        if find_spec("flash_attn") is None:
            return False

        return True
    except ImportError:
        return False


@cache
def _get_gcn_arch_name() -> str:
    # Platform discovery also runs in CPU-only lint and contract jobs.  Do not
    # initialize the HIP runtime when no accelerator is available.
    if not torch.cuda.is_available():
        return ""
    try:
        GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    except (AssertionError, RuntimeError):
        # vLLM's ROCm platform bootstrap can make ``is_available`` report true
        # on a build host that has the driver stack but no usable device.
        return ""
    return GPU_ARCH.split(":")[0]


_ON_GFX93X = any(arch in _get_gcn_arch_name() for arch in ["gfx936", "gfx938"])
_ON_GFX938 = "gfx938" in _get_gcn_arch_name()


def on_gfx93x() -> bool:
    return _ON_GFX93X

def on_gfx938() -> bool:
    return _ON_GFX938

@cache
def _get_backend_priorities(
    use_mla: bool,
    use_sparse: bool,
) -> list[AttentionBackendEnum]:
    """Get HCU backend priorities; validation filters unavailable kernels."""
    if use_sparse:
        return [
            AttentionBackendEnum.FLASHMLA_SPARSE,
            AttentionBackendEnum.ROCM_AITER_MLA_SPARSE,
        ]

    if use_mla:
        return [
            AttentionBackendEnum.FLASHMLA,
            AttentionBackendEnum.TRITON_MLA,
        ]
    return [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TRITON_ATTN,
    ]


def register_attention_backends() -> None:
    # Pre-register all attention backends
    register_backend(
        AttentionBackendEnum.TRITON_ATTN,
        class_path="vllm_hcu.v1.attention.backends.triton_attn.HcuTritonAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_hcu.v1.attention.backends.flash_attn.HcuFlashAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA_SPARSE,
        class_path="vllm_hcu.v1.attention.backends.mla.flashmla_sparse.HcuFlashMLASparseBackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_hcu.v1.attention.backends.mla.flashmla.HcuFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.TRITON_MLA,
        class_path="vllm_hcu.v1.attention.backends.mla.triton_mla.HcuTritonMLABackend",
    )


class HCUPlatform(RocmPlatform):
    #这个地方会管理custom_ops
    _enum = PlatformEnum.ROCM
    device_name: str = "hip"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    ]

    supported_quantization: list[str] = [
        "awq",
        "awq_marlin",  # will be overwritten with awq
        "gptq",
        "gptq_marlin",  # will be overwritten with gptq
        "fp8",
        "deepseek_v4_fp8",
        "compressed-tensors",
        "fbgemm_fp8",
        "gguf",
        "quark",
        # "mxfp4",
        "mxfp8",
        "torchao",
        # "bitsandbytes",
        "modelopt",
        # "modelopt_fp4",
        "modelopt_mxfp8",
        "modelopt_mixed",
        "fp8_per_tensor",
        "fp8_per_block",
        "online",
        # "gpt_oss_mxfp4",
        "slimquant_w4a8",
        "slimquant_w4a8_marlin", 
        "slimquant_compressed_tensors_marlin",
    ]
    
    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]:
        valid_backends_priorities = []
        invalid_reasons = {}
        
        register_attention_backends()
        
        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            attn_selector_config.use_sparse,
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = invalid_reasons_i
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        if ( flash_attn_triton_available()
            and (dtype == torch.float16 or dtype == torch.bfloat16)
        ):
            logger.info_once("Using Flash Attention backend for ViT model.")
            return AttentionBackendEnum.FLASH_ATTN


        logger.info_once("Using Torch SDPA backend for ViT model.")
        return AttentionBackendEnum.TORCH_SDPA
    
    @classmethod
    def use_custom_allreduce(cls) -> bool:
        # We only enable custom allreduce for MI300 series
        # return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])
        return True
    
    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """Check whether all selected devices have a direct high-speed link."""

        api = _load_hcu_management_api()
        if api is None:
            logger.warning_once(
                "HCU management dependency is unavailable; "
                "custom all-reduce will be disabled."
            )
            return False

        initialized = False
        try:
            api.amdsmi_init()
            initialized = True
            all_handles = api.amdsmi_get_processor_handles()
            handles = [all_handles[index] for index in physical_device_ids]
            for index, handle in enumerate(handles):
                for peer_handle in handles[index + 1 :]:
                    link = api.amdsmi_topo_get_link_type(
                        handle,
                        peer_handle,
                    )
                    # Value 2 is the management API's direct high-speed-link type.
                    if link["hops"] != 1 or link["type"] != 2:
                        return False
            return True
        except Exception:
            logger.warning_once(
                "HCU topology detection failed; "
                "custom all-reduce will be disabled."
            )
            return False
        finally:
            if initialized:
                try:
                    api.amdsmi_shut_down()
                except Exception:
                    logger.warning_once("HCU management cleanup failed.")


    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        """Register HCU config adapters through vLLM's public platform hook."""
        from vllm_hcu.patch.import_coordinator import IMPORT_COORDINATOR
        from vllm_hcu.patch.platform.core_fix.patch_hcu_config import (
            pre_register_and_update,
        )

        IMPORT_COORDINATOR.drain_ready_callbacks()
        pre_register_and_update(parser)

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        """Arm HCU operators, then retain target ROCm platform defaults."""

        from vllm_hcu.patch.import_coordinator import IMPORT_COORDINATOR
        from vllm_hcu.patch.worker import prepare_worker_patches

        IMPORT_COORDINATOR.drain_ready_callbacks()
        prepare_worker_patches()
        super().apply_config_platform_defaults(vllm_config)

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Apply target v0.28 policy before narrow HCU selectors."""

        from vllm_hcu.patch.import_coordinator import IMPORT_COORDINATOR
        from vllm_hcu.patch.platform.core_fix.patch_vllm_config import (
            validate_and_update_hcu_config,
        )
        from vllm_hcu.patch.platform.framework_opt.patch_multiproc_executor import (
            select_hcu_multiproc_executor,
        )
        from vllm_hcu.patch.platform.framework_opt.patch_scheduler import (
            select_hcu_scheduler,
        )

        IMPORT_COORDINATOR.drain_ready_callbacks()
        validate_and_update_hcu_config(vllm_config)
        super().check_and_update_config(vllm_config)
        select_hcu_scheduler(vllm_config)
        select_hcu_multiproc_executor(vllm_config)

        # RocmPlatform resolves "auto" to the target Worker. Replace only that
        # exact selected implementation; preserve every explicit user choice.
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "vllm.v1.worker.gpu_worker.Worker":
            parallel_config.worker_cls = "vllm_hcu.v1.worker.HcuGPUWorker"


    @classmethod
    def supports_fp8(cls) -> bool:
        return on_gfx938()

# The registry stores lazy class paths, so this does not import HCU kernels.
# It does make explicit backend selection resolve to HCU before validation.
register_attention_backends()
