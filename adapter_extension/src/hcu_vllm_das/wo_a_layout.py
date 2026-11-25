# SPDX-License-Identifier: Apache-2.0
"""Write native BF16 WO_A directly in the following linear's layout."""
from __future__ import annotations

import inspect
import logging
import os
from functools import lru_cache
from types import SimpleNamespace

import torch
from hcu_adapter.operator_contracts import supports_operator_contract
from vllm.models.deepseek_v41.amd import rocm
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as native

_O_PROJ = rocm.DeepseekV41ROCMAiterMLAAttention._o_proj
_CONSUMER = rocm.DeepseekV41ROCMAiterMLAAttention._wo_b_after_wo_a
_EINSUM = native.rocm_inv_rope_einsum
_CACHE = inspect.unwrap(native._get_cached_wo_a_bf16)


@lru_cache(maxsize=2)
def _cache_contract(function):
    if function is _CACHE:
        return True, False
    verified = (function is getattr(native, '_vllm_hcu_get_cached_wo_a_bf16', None)
                and getattr(function, '__wrapped__', None) is _CACHE
                and getattr(function, '_vllm_hcu_wo_a_nn_weight_layout_wrapper', False))
    return verified, verified


_BLAS_OVERRIDDEN = any(os.environ.get(name) for name in (
    'ROCBLAS_TENSILE_LIBPATH', 'ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH',
    'HIPBLASLT_TENSILE_LIBPATH', 'HIPBLASLT_TUNING_OVERRIDE_FILE'))
_SOURCE_MATCHES = (not _BLAS_OVERRIDDEN
                   and supports_operator_contract(native, 'dsv41_wo_a_bf16_consumer_layout')
                   and supports_operator_contract(torch, 'rocm_bf16_bmm_einsum_parity'))
_qualified_devices = set()
_LOG = logging.getLogger('vllm.' + __name__)
_CAPTURE_LOG = os.getenv('HCU_VLLM_WO_A_CAPTURE_LOG') == '1'
_logged_capture_devices = set()


def _clear_preflight():
    _qualified_devices.clear()
    _logged_capture_devices.clear()
    _qualify_layout.cache_clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_clear_preflight)


def preflight_admitted():
    """Include worker qualification in the compile-cache factor."""
    return bool(_qualified_devices)


def configured(config, device):
    from .environment import WO_A_TOKEN_MAJOR_SWITCH, _flag

    if not _flag(WO_A_TOKEN_MAJOR_SWITCH, default=True):
        return False

    from .quantization.ct_w4a8 import HcuDSV41CTW4A8Config
    from .adapter_api import adapter_module

    channel_fp8 = adapter_module('dsv41_channel_fp8').HcuDSV41ChannelFP8Config
    if _BLAS_OVERRIDDEN or not isinstance(
            config.quant_config, (HcuDSV41CTW4A8Config, channel_fp8)):
        return False
    parallel = config.parallel_config
    mc = config.model_config
    outer = mc.hf_config
    model = getattr(mc, 'hf_text_config', None)
    if model is None:
        model = getattr(outer, 'text_config', outer)
    return (_cache_contract(native._get_cached_wo_a_bf16)[0]
            and getattr(outer, 'model_type', None) in ('deepseek_v41', 'deepseek_v41_text')
            and getattr(model, 'model_type', None) in ('deepseek_v41', 'deepseek_v41_text')
            and getattr(model, 'hidden_size', None) == 5120
            and parallel.pipeline_parallel_size == 1
            and parallel.tensor_parallel_size == parallel.prefill_context_parallel_size == 1
            and parallel.data_parallel_size == 8 and parallel.enable_expert_parallel
            and qualify_wo_a_layout(device))


def qualify_wo_a_layout(device):
    if _BLAS_OVERRIDDEN:
        return False
    if _SOURCE_MATCHES:
        return True
    device = torch.device(device)
    if device.type != 'cuda':
        return False
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if getattr(props, 'gcnArchName', '').split(':')[0] != 'gfx938' or props.multi_processor_count != 64:
        return False
    cache = native._get_cached_wo_a_bf16
    valid, transposed = _cache_contract(cache)
    if not valid:
        return False
    return _qualify_layout(device, cache, transposed)


@lru_cache(maxsize=8)
@torch.inference_mode()
def _qualify_layout(device, cache, transposed):
    """Exercise the installed native projection and its consumer handoff."""
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            generator = torch.Generator(device=device).manual_seed(7004)
            weight = torch.randn((8192, 4096), dtype=torch.bfloat16,
                                 device=device, generator=generator)
            for layout in ('logical', 'nn') if transposed else ('logical',):
                physical_weight = weight if layout == 'logical' else weight.T.contiguous()
                owner = SimpleNamespace(
                    n_local_heads=64, n_local_groups=8, o_lora_rank=1024,
                    rotary_emb=None, rope_head_dim=64,
                    wo_a=SimpleNamespace(weight=physical_weight),
                    _wo_b_after_wo_a=lambda value: value)
                for rows in (1, 2, 8, 17, 32, 48, 64, 65, 128, 192, 255, 256):
                    for heads in (64, 128):
                        attn_out = torch.randn((rows, heads, 512), dtype=torch.bfloat16,
                                               device=device, generator=generator)
                        expected = _O_PROJ(owner, attn_out, None)
                        actual = _project(cache, owner.wo_a, attn_out)
                        if (expected.shape != actual.shape or expected.dtype != actual.dtype
                                or not torch.equal(expected.contiguous().view(torch.uint8),
                                                   actual.view(torch.uint8))):
                            _LOG.warning('HCU WO-A layout preflight differs from native projection')
                            return False
        except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as error:
            _LOG.warning('HCU WO-A layout preflight unavailable: %s', error)
            return False
    _qualified_devices.add(device)
    _LOG.info('HCU direct WO-A consumer layout admitted after native preflight')
    return True


def supported(owner, attn_out):
    # Out= BMM is an inference-only route. Autocast callers keep the native
    # einsum's own casting policy, even when the supplied tensors are BF16.
    if (not isinstance(attn_out, torch.Tensor)
            or not (_SOURCE_MATCHES or attn_out.device in _qualified_devices)
            or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
            or not attn_out.is_cuda or attn_out.dtype != torch.bfloat16
            or attn_out.ndim != 3 or not 0 < attn_out.shape[0] <= 256
            or attn_out.shape[1] not in (64, 128) or attn_out.shape[2] != 512
            or not attn_out.is_contiguous()
            or (owner.n_local_heads, owner.n_local_groups, owner.o_lora_rank) != (64, 8, 1024)
            or owner._wo_b_scale is not None):
        return False
    from .wo_a_token_major import owns_projection_wrapper

    project = rocm.DeepseekV41ROCMAiterMLAAttention._o_proj
    if ((project is not _O_PROJ and not owns_projection_wrapper(_O_PROJ))
            or rocm.DeepseekV41ROCMAiterMLAAttention._wo_b_after_wo_a is not _CONSUMER
            or rocm.rocm_inv_rope_einsum is not _EINSUM
            or native.rocm_inv_rope_einsum is not _EINSUM
            or getattr(owner._wo_b_after_wo_a, '__func__', None) is not _CONSUMER):
        return False
    cache_valid, cache_nn = _cache_contract(native._get_cached_wo_a_bf16)
    if not cache_valid:
        return False
    weight = owner.wo_a.weight
    cached = getattr(owner.wo_a, '_dsv4_wo_a_bf16', None)
    allowed_weights = ((8192, 4096), (4096, 8192)) if cache_nn else ((8192, 4096),)
    if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16
            or tuple(weight.shape) not in allowed_weights or not weight.is_contiguous()
            or weight.device != attn_out.device):
        return False
    if cached is not None and (not isinstance(cached, torch.Tensor)
            or cached.shape != (8, 1024, 4096) or cached.dtype != torch.bfloat16
            or cached.device != attn_out.device or not cached.is_contiguous()):
        return False
    properties = torch.cuda.get_device_properties(attn_out.device)
    return (getattr(properties, 'gcnArchName', '').split(':')[0] == 'gfx938'
)


def project(owner, attn_out):
    if not supported(owner, attn_out):
        raise ValueError('WO_A token-major output is outside the verified BF16 contract')
    output = _project(native._get_cached_wo_a_bf16, owner.wo_a, attn_out)
    if (_CAPTURE_LOG and attn_out.device not in _logged_capture_devices
            and torch.cuda.is_current_stream_capturing()):
        _logged_capture_devices.add(attn_out.device)
        _LOG.info('HCU token-major WO-A consumer layout captured')
    return output


def _project(cache, wo_a, attn_out):
    rows = attn_out.shape[0]
    inputs = attn_out[:, :64, :].reshape(rows, 8, 4096)
    weight = cache(wo_a, 8, 1024, 4096)
    result = torch.empty((rows, 8, 1024), dtype=attn_out.dtype, device=attn_out.device)
    torch.bmm(inputs.transpose(0, 1), weight.transpose(1, 2), out=result.transpose(0, 1))
    # This is a view. Native einsum produces group-major storage and flatten
    # must copy it; the following native linear now receives the same
    # byte-verified BF16 values in contiguous token-major storage.
    return result.flatten(1)
