# SPDX-License-Identifier: Apache-2.0
"""Step5 TP1/EP16 WASP128 integration with the DTK DeepGEMM nightly API.

Weights use explicit Pair32 packing. Dispatch, G1 and G2 share the same
128-row expert-slice contract; the ordinary and LL backends keep their layout.
"""
import os
from functools import lru_cache
from importlib import import_module

from vllm_hcu.platforms import envs as henvs


@lru_cache(maxsize=1)
def _nightly_api():
    try:
        dg = import_module("deepgemm")
        pack = import_module("deepgemm.group_gemm_weight_pack")
    except ImportError as exc:
        raise RuntimeError("Step5 WASP128 requires the DTK DeepGEMM Pair32 wheel") from exc
    gemm = getattr(dg, "m_grouped_fp8_gemm_nt_contiguous_wasp_pair32", None)
    packer = getattr(pack, "pack_fp8_weight_pair32", None)
    if not callable(gemm) or not callable(packer):
        raise RuntimeError("DeepGEMM is missing WASP128 GEMM or explicit Pair32 packing")
    return gemm, packer


def nightly_weight_packer(moe_config):
    """Qualify only the requested two-node, Channel-FP8 Step5 HT profile."""
    if not henvs.optional_custom_op_enabled(
        os.environ.get("HCU_STEP5_WASP128") == "1"
    ):
        return None
    from vllm.config import get_current_vllm_config_or_none
    from vllm_hcu.platforms.hcu import on_gfx938
    from vllm_hcu.runtime_compat.step5 import is_step5_owner

    if not is_step5_owner(moe_config):
        return None
    config = get_current_vllm_config_or_none()
    parallel = getattr(moe_config, "moe_parallel_config", None)
    launch = getattr(config, "parallel_config", None)
    if not (
        on_gfx938()
        and getattr(parallel, "use_deepep_ht_kernels", False)
        and parallel.tp_size == 1
        and parallel.ep_size == 16
        and moe_config.num_experts in (352, 368)
        and moe_config.hidden_dim_unpadded == 4096
        and moe_config.intermediate_size_per_partition_unpadded == 1536
        and getattr(launch, "data_parallel_size", None) == 16
        and getattr(launch, "data_parallel_size_local", None) == 8
        and os.environ.get("DEEP_EP_NORMAL_MNVL") == "1"
        and os.environ.get("VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE") != "1"
        and (getattr(launch, "enable_dbo", False)
             or int(os.environ.get("HCU_DBO_MIN_TOKENS", "0")) > 0)
        and all(os.environ.get(flag) == "1" for flag in (
            "HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT",
            "HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR",
            "HCU_STEP5_DEEPEP_FUSED_SCATTER",
        ))
    ):
        raise ValueError("Step5 WASP128 requires gfx938 TP1/DP16/EP16 MNVL with fused scatter and DBO")
    return _nightly_api()[1]


def grouped_fp8(a, b, out, experts):
    """Launch both Step5 GEMMs with BM128/BN256 (nightly mode 1002)."""
    shape = (a[0].size(1), out.size(1))
    if shape not in ((4096, 3072), (1536, 4096)):
        raise ValueError(f"Unsupported Step5 WASP128 GEMM shape: {shape}")
    weight = b[0]
    n, k = out.size(1), a[0].size(1)
    expected = (weight.size(0), k // 64, n // 32, 2, 4, 16, 16)
    if weight.size(0) not in (22, 23) or tuple(weight.shape) != expected:
        raise ValueError("Step5 EP16 WASP128 requires Pair32 weights for 22 or 23 local experts")
    if a[0].size(0) % 128 or experts.numel() != a[0].size(0):
        raise ValueError("Step5 WASP128 requires 128-row dispatch slices and matching expert IDs")
    from vllm.logger import init_logger

    init_logger("vllm.hcu.step5_wasp128").info_once(
        "EP16 WASP128 G1/G2 active through DeepGEMM Pair32 mode 1002", scope="process",
    )
    return _nightly_api()[0](a, b, out, experts, config={"MODE": 1002})
