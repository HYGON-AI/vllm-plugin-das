# SPDX-License-Identifier: Apache-2.0
"""Load the optional Step5 EP8 grouped FP8 GEMM operator."""

import hashlib
import json
import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.platforms.hcu import on_gfx938


@lru_cache(maxsize=1)
def _deepgemm():
    if not on_gfx938():
        return None
    try:
        return import_module("deepgemm")
    except ImportError:
        return None


def _deepgemm_grouped_fp8():
    return getattr(_deepgemm(), "m_grouped_fp8_gemm_nt_contiguous_step5_ep8", None)


@lru_cache(maxsize=1)
def _pair32_api():
    gemm = getattr(_deepgemm(), "m_grouped_fp8_gemm_nt_contiguous_wasp_pair32", None)
    if not callable(gemm):
        return None, None
    try:
        pack_module = import_module("deepgemm.group_gemm_weight_pack")
    except ImportError:
        return None, None
    pack = getattr(pack_module, "pack_fp8_weight_pair32", None)
    return (gemm, pack) if callable(pack) else (None, None)


def nightly_weight_packer(moe_config):
    """Select the explicit Pair32 packer only for the local EP8 HT path."""
    parallel = getattr(moe_config, "moe_parallel_config", None)
    if not (
        henvs.custom_ops_enabled()
        and getattr(parallel, "use_deepep_ht_kernels", False)
        and parallel.tp_size == 1
        and parallel.ep_size == 8
        and moe_config.num_experts == 352
        and os.environ.get("DEEP_EP_NORMAL_MNVL") != "1"
        and all(os.environ.get(flag) == "1" for flag in (
            "VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE",
            "HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT",
            "HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR",
            "HCU_STEP5_DEEPEP_FUSED_SCATTER",
        ))
    ):
        return None
    config = get_current_vllm_config_or_none()
    if not (
        config is not None
        and config.parallel_config.data_parallel_size_local == 8
        and (
            config.parallel_config.enable_dbo
            or int(os.environ.get("HCU_DBO_MIN_TOKENS", "0")) > 0
        )
    ):
        return None
    if callable(_deepgemm_grouped_fp8()):
        return None
    return _pair32_api()[1]


def available(moe_config=None):
    if callable(_deepgemm_grouped_fp8()):
        return True
    if _pair32_api()[0] is not None:
        return nightly_weight_packer(moe_config) is not None
    return bool(os.environ.get("HCU_STEP5_EP8_WASP128_LIBRARY"))


@lru_cache(maxsize=1)
def _operator():
    library=Path(os.environ["HCU_STEP5_EP8_WASP128_LIBRARY"])
    proof=json.loads(library.with_suffix('.json').read_text())
    if not proof.get('passed') or proof.get('torch') != torch.__version__:
        raise RuntimeError(f"Step5 EP8 WASP128 qualification does not match PyTorch: {library}")
    if hashlib.sha256(library.read_bytes()).hexdigest() != proof.get('sha256'):
        raise RuntimeError(f"Step5 EP8 WASP128 checksum mismatch: {library}")
    torch.ops.load_library(str(library))
    init_logger('vllm.hcu.step5_wasp128').info_once('EP8 WASP128 native G1/G2 active',scope='process')
    return torch.ops.step5_wasp128_probe.gemm


def grouped_fp8(a, b, out, experts):
    native = _deepgemm_grouped_fp8()
    pair32, _ = _pair32_api() if not callable(native) else (None, None)
    if callable(native) or pair32 is not None:
        init_logger("vllm.hcu.step5_wasp128").info_once(
            "EP8 WASP128 native G1/G2 active through DeepGEMM%s",
            " (Pair32 mode 1000)" if pair32 is not None else "",
            scope="process",
        )
        if callable(native):
            return native(a, b, out, experts)
        shape = (a[0].size(1), out.size(1))
        if shape not in ((4096, 3072), (1536, 4096)):
            raise ValueError(f"Unsupported Step5 WASP128 GEMM shape: {shape}")
        return pair32(a, b, out, experts, config={"MODE": 1000})

    bn = 384 if a[0].size(1) == 4096 else 256
    return _operator()(
        a[0], b[0], out, a[1].contiguous(), b[1].contiguous(),
        experts.contiguous(), 128, bn, 4,
    )
