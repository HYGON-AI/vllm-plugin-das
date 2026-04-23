# SPDX-License-Identifier: Apache-2.0
import os
import torch
import regex as re
import vllm.envs as envs
from datetime import timedelta
from functools import cache, lru_cache, wraps
from typing import TYPE_CHECKING
from torch.distributed import PrefixStore, ProcessGroup
from torch.distributed.distributed_c10d import is_nccl_available
from vllm.logger import init_logger
from vllm.utils.torch_utils import cuda_device_count_stateless
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig

import vllm_hcu.platforms.envs as henvs 

logger = init_logger(__name__)

try:
    from amdsmi import (
        AmdSmiException,
        amdsmi_get_gpu_asic_info,
        amdsmi_get_processor_handles,
        amdsmi_init,
        amdsmi_shut_down,
        amdsmi_topo_get_link_type,
    )
except ImportError as e:
    logger.warning("Failed to import from amdsmi with %r", e)

try:
    import vllm._C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._C with %r", e)


# AMDSMI utils
# Note that NVML is not affected by `{CUDA/HIP}_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using AMDSMI is that it will not initialize CUDA


def with_amdsmi_context(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        amdsmi_init()
        try:
            return fn(*args, **kwargs)
        finally:
            amdsmi_shut_down()

    return wrapper

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
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    return GPU_ARCH.split(':')[0]


_ON_GFX938 = "gfx938" in _get_gcn_arch_name()


def on_gfx938() -> bool:
    return _ON_GFX938

@cache
def _get_backend_priorities(
    use_mla: bool,
    use_sparse: bool,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    # HCU platform prioritizes Triton-based backends since FlashAttention
    # and other CUDA-specific backends may not be available.
    if use_sparse:
        return [AttentionBackendEnum.ROCM_AITER_MLA_SPARSE]

    if use_mla:
        if henvs.VLLM_HCU_USE_FLASHMLA:
            return [
                AttentionBackendEnum.FLASHMLA,
            ]
        else:
            return [
                AttentionBackendEnum.TRITON_MLA,
            ]
    else:
        if henvs.VLLM_HCU_USE_FLASH_ATTN:
            return [
                AttentionBackendEnum.FLASH_ATTN,
            ]
        else:
            return [
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
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_hcu.v1.attention.backends.mla.flashmla.HcuFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.TRITON_MLA,
        class_path="vllm_hcu.v1.attention.backends.mla.triton_mla.HcuTritonMLABackend",
    )


class HCUPlatform(Platform):
    #这个地方会管理custom_ops
    _enum = PlatformEnum.ROCM
    device_name: str = "hip"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"

    @classmethod
    def import_kernels(cls) -> None:
        """Import ROCm-specific kernels."""
        super().import_kernels()

        import contextlib

        # Import ROCm-specific extension
        with contextlib.suppress(ImportError):
            import vllm._rocm_C  # noqa: F401

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
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        attn_selector_config = attn_selector_config._replace(block_size=None)

        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons = ["ImportError"]
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return selected_backend.get_path()

        # No selected backend or the selected backend is invalid,
        # so we try finding a valid backend.
        valid_backends_priorities, invalid_reasons = cls.get_valid_backends(
            device_capability=device_capability,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )
        reasons_str = (
            "{"
            + ", ".join(
                f"{backend.name}: [{', '.join(reasons)}]"
                for backend, reasons in invalid_reasons.items()
            )
            + "}"
        )
        config_str = attn_selector_config.__repr__()
        logger.debug_once(
            f"Some attention backends are not valid for {cls.device_name} with "
            f"{config_str}. Reasons: {reasons_str}."
        )
        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )

        # We have found some valid backends. Select the one with the
        # highest priority.
        sorted_indices = sorted(
            range(len(valid_backends_priorities)),
            key=lambda i: valid_backends_priorities[i][1],
        )
        selected_index = sorted_indices[0]
        selected_backend = valid_backends_priorities[selected_index][0]
        logger.info_once(
            "Using %s attention backend out of potential backends: %s.",
            selected_backend.name,
            "[" + ", ".join(f"'{b[0].name}'" for b in valid_backends_priorities) + "]",
            scope="local",
        )

        return selected_backend.get_path()

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
    @with_amdsmi_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        Query if the set of gpus are fully connected by xgmi (1 hop)
        """
        handles = [amdsmi_get_processor_handles()[i] for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        link_type = amdsmi_topo_get_link_type(handle, peer_handle)
                        # type is 2 for XGMI
                        if link_type["hops"] != 1 or link_type["type"] != 2:
                            return False
                    except AmdSmiException as error:
                        logger.error("AMD 1 hop XGMI detection failed.", exc_info=error)
                        return False
        return True


    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)
    
    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)


    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.config.compilation import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        is_eager_execution = compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        use_aiter_fused_moe = rocm_aiter_ops.is_fused_moe_enabled()
        use_aiter_rms_norm = rocm_aiter_ops.is_rmsnorm_enabled()
        use_aiter_fp8_linear = rocm_aiter_ops.is_linear_fp8_enabled()
        use_aiter_fused_se = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        #  Aiter rms norm perform best when CUDA Graph capture is enabled.
        if (
            use_aiter_rms_norm
            and not is_eager_execution
            and "-rms_norm" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+rms_norm")

        if use_aiter_fp8_linear and "-quant_fp8" not in compilation_config.custom_ops:
            compilation_config.custom_ops.append("+quant_fp8")

        if use_aiter_fused_se and "-grouped_topk" in compilation_config.custom_ops:
            logger.warning_once(
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled, which "
                "requires the 'grouped_topk' custom op. Overriding the "
                "user-provided '-grouped_topk'."
            )
            compilation_config.custom_ops.remove("-grouped_topk")
        # Ensure grouped_topk is always enabled when using AITER if
        # its not disabled by user
        if (
            use_aiter_fused_moe
            and "+grouped_topk" not in compilation_config.custom_ops
            and "-grouped_topk" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+grouped_topk")
        # Enable rotary embedding customop when using AITER if not disabled by user
        if (
            rocm_aiter_ops.is_enabled()
            and "+rotary_embedding" not in compilation_config.custom_ops
            and "-rotary_embedding" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+rotary_embedding")

        # Default dispatch to rocm's sparse_attn_indexer implementation
        compilation_config.custom_ops.append("+sparse_attn_indexer")

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config.compilation import CUDAGraphMode

        cache_config = vllm_config.cache_config
        compilation_config = vllm_config.compilation_config
        parallel_config = vllm_config.parallel_config
        # if cache_config and cache_config.block_size is None:
        #     cache_config.block_size = 64
        if compilation_config.cudagraph_mode.has_full_cudagraphs():
            # decode context parallel does not support full cudagraphs
            if parallel_config.decode_context_parallel_size > 1:
                logger.warning_once(
                    "Decode context parallel (DCP) is enabled, which is "
                    "incompatible with full CUDA graphs. "
                    "Overriding cudagraph_mode to PIECEWISE."
                )
                compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
            # prefill context parallel do not support full cudagraphs
            elif parallel_config.prefill_context_parallel_size > 1:
                logger.warning_once(
                    "Prefill context parallel (PCP) is enabled, which is "
                    "incompatible with full CUDA graphs. "
                    "Overriding cudagraph_mode to PIECEWISE."
                )
                compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

        if cache_config and not cache_config.user_specified_block_size:
            if (
                envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION and envs.VLLM_ROCM_USE_AITER
                # NOTE: This block has been deprecated
                # or get_env_variable_attn_backend()
                # == AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN
                # TODO: monitor https://github.com/vllm-project/vllm/pull/30396
                # to see how we can transition to the new way of selecting
                # attention backends
            ):
                cache_config.block_size = 64
                logger.warning(
                    "[ROCM_AITER_UNIFIED_ATTN]: Setting kv cache block size to 64."
                )
            elif henvs.VLLM_HCU_USE_FLASH_ATTN or henvs.VLLM_HCU_USE_FLASHMLA:
                cache_config.block_size = 64
                logger.warning(
                    "[HCU_FA/FlashMLA_UNIFIED_ATTN]: Setting kv cache block size to 64."
                )
            else:
                cache_config.block_size = 16

        if parallel_config.worker_cls == "auto":
            # parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
            parallel_config.worker_cls = "vllm_hcu.v1.worker.HcuGPUWorker"
            

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        # TODO: ROCm still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        pass


    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        return total_mem - free_mem

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa
        )

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True


    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype == torch.bfloat16:  # noqa: SIM102
            if not cls.has_device_capability(80):
                capability = cls.get_device_capability()
                gpu_name = cls.get_device_name()

                if capability is None:
                    compute_str = "does not have a compute capability"
                else:
                    version_str = capability.as_version_str()
                    compute_str = f"has compute capability {version_str}"

                raise ValueError(
                    "Bfloat16 is only supported on GPUs "
                    "with compute capability of at least 8.0. "
                    f"Your {gpu_name} GPU {compute_str}. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half."
                )

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on GPU."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from GPU to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return True
