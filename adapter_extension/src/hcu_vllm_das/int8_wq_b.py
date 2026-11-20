# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Share the owned FP4 attention's post-RMSNorm INT8 WQ-B operand."""
from __future__ import annotations

import hashlib
import inspect
from functools import partial
from importlib.metadata import PackageNotFoundError, version

import torch
from .adapter_api import adapter_module

HcuDSV41Int8LinearMethod = adapter_module('dsv41_int8_linear').HcuDSV41Int8LinearMethod
from vllm import _custom_ops as ops
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm import triton
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from vllm.models.deepseek_v41 import attention as common
from vllm.models.deepseek_v41.amd import rocm
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

_K, _MAIN_N, _INDEX_N = 1280, 32768, 4096
_CORE_WHEEL = '0.33.0+dtk2610.torch2130.2611101131.g7364da'
_LOG = init_logger(__name__)


def _source_matches(fn, digest):
    if not inspect.isfunction(fn) or hasattr(fn, '__wrapped__'):
        return False
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() == digest
    except (OSError, TypeError):
        return False


def _native_quant():
    quant = ops.scaled_int8_quant
    if hasattr(quant, '__wrapped__'):
        try:
            from hcu_adapter.bridges.vllm_input_quant_int8_lightop import native_quant_binding
        except ImportError:
            return quant
        return native_quant_binding(ops)
    return quant


def _reviewed_source():
    amd, base, indexer = (rocm.DeepseekV41ROCMAiterMLAAttention,
                          common.DeepseekV4Attention, common.DeepseekV4Indexer)
    kernel = triton.TritonInt8ScaledMMLinearKernel
    method = HcuDSV41Int8LinearMethod
    if (method.apply is not getattr(method, '_hcu_wq_b_int8_apply_v1', None)
            or method.process_weights_after_loading is not getattr(
                method, '_hcu_wq_b_int8_finalize_v1', None)):
        _LOG.warning_once('INT8 WQ-B adapter method contract unavailable')
        return False
    reviewed = (
        (amd.forward, '3152e366240c837a1b12c2f28094dad235fabe5e2743450c2b38bb6b9b88b5f8'),
        (amd._forward_csa2_full, '5c5e13336e9d76ea02f585d18a5e3935c11bf5998a42972980c7f89da93b7b6f'),
        (amd._forward_csa2_reindex, 'a55aa720527b95dd14effc4d9de6316bdd345a5cf447e1f16b5477643001cdcd'),
        (amd._split_qkv_and_norm, '4af899810969ad71d4fd5ea6de158bd7873234f7b1714581a9b6f8a131125c3f'),
        (base.forward, 'c8d93b381dd17f4e7a44794bd24045694ea199cdad457d810ac6dda6112225df'),
        (base._split_qkv_and_norm, 'b067814c475fd1c9b170bef2e30716f5543851f3355340f3c036731481e5efe8'),
        (base._prepare_and_attn, '7724460d5f00168783bf4b1ea67b9dfbb12dd302ee091afa9b73812e6f7c7470'),
        (indexer.forward, '9a7000f080675c697ee3ebd18e437122b7bdb8f4e5c3f87ae8576c147bd17a88'),
        (ColumnParallelLinear.forward, '308fabe081224a58eafc09a982aac11fb31ddc1d73ea5f5ef33085fca446a56c'),
        (ReplicatedLinear.forward, 'ca4826436841648b26a8b83c4026abadab32a29b86359261900dd936b13939d2'),
        (CompressedTensorsLinearMethod.apply, 'f2267a9c76789466651134de4df7a8ec75fb0928c0b8780c82443c07f62abe73'),
        (CompressedTensorsW8A8Int8.apply_weights, '7560e8139c47af3606304a13dd4aee4aaa5ec8f9a7a7f050d73e8001c575ec5f'),
        (kernel.apply_weights, 'c825e9cc457413d572e4bcffafd50d3256026d36e3ab83208947e7337371b0eb'),
        (kernel._get_layer_params, '748b270e83d959bdd74167e048c8df573806f6928930061aa08c06d1ab5489e1'),
        (triton._int8_scaled_mm_func, '90fac41461e287ff9da855adc7408073d459bc729ac5f0a8dd3e99413592b940'),
        (_native_quant(), '195d8e863eb49e6ba2e59f6f822845a956a722f12a7a3b185c279a84d6b1f2ab'),
    )
    for fn, digest in reviewed:
        if not _source_matches(fn, digest):
            _LOG.warning_once('INT8 WQ-B native source differs: %s', fn.__qualname__)
            return False
    return True


def _registered_ops():
    try:
        quant = torch.ops._C.dynamic_scaled_int8_quant.default._schema
        mm = torch.ops.vllm.int8_scaled_mm_func.default._schema
        return (
            tuple((a.name, str(a.real_type)) for a in quant.arguments) == (
                ('result', 'Tensor'), ('input', 'Tensor'), ('scale', 'Tensor'), ('azp', 'Optional[Tensor]')
            )
            and tuple((a.name, str(a.real_type)) for a in mm.arguments) == (
                ('qx', 'Tensor'), ('weight', 'Tensor'), ('x_scale', 'Tensor'),
                ('weight_scale', 'Tensor'), ('output_dtype', 'ScalarType'), ('bias', 'Optional[Tensor]')
            )
            and tuple(bool(a.alias_info and a.alias_info.is_write) for a in quant.arguments)
            == (True, False, True, True)
            and not any(a.alias_info for a in mm.arguments)
            and not quant.returns and tuple(str(r.real_type) for r in mm.returns) == ('Tensor',)
            and all(torch._C._dispatch_has_kernel_for_dispatch_key(schema.name, 'CUDA')
                    for schema in (quant, mm))
        )
    except (AttributeError, RuntimeError):
        return False


def configured_device(cfg):
    """Resolve the existing WQ-B operand-sharing policy."""
    if cfg.model_config.enforce_eager:
        return None
    return _configured_int8_device(cfg)


def _configured_int8_device(cfg):
    p = cfg.parallel_config
    if (not cfg.use_v2_model_runner
            or p.tensor_parallel_size != 1 or p.prefill_context_parallel_size != 1
            or p.decode_context_parallel_size != 1 or p.use_ubatching
            or not torch.version.hip or not current_platform.is_rocm()
            or current_platform.device_name != 'hcu'):
        return None
    device = torch.cuda.current_device()
    arch = getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '')
    if arch.split(':', 1)[0] != 'gfx938':
        return None
    try:
        supported = version('vllm') == _CORE_WHEEL and _reviewed_source() and _registered_ops()
    except (AttributeError, PackageNotFoundError):
        supported = False
    if not supported:
        _LOG.warning_once('INT8 WQ-B sharing disabled because the pinned native ABI differs')
        return None
    return device


def _layer_matches(layer, device, *, indexer):
    cls, n = (ReplicatedLinear, _INDEX_N) if indexer else (ColumnParallelLinear, _MAIN_N)
    if (type(layer) is not cls or getattr(layer.forward, '__func__', None) is not cls.forward
            or layer.has_bias or layer.bias is not None or layer.skip_bias_add
            or layer.return_bias is not indexer or layer.tp_size != 1 or layer.tp_rank != 0
            or layer.input_size != _K or layer.output_size != n
            or tuple(layer.output_partition_sizes) != (n,)):
        return False
    if not indexer and (layer.gather_output or layer.input_size_per_partition != _K
                        or layer.output_size_per_partition != n):
        return False
    method, scheme = layer.quant_method, getattr(layer, 'scheme', None)
    if (type(method) is not HcuDSV41Int8LinearMethod
            or getattr(method.apply, '__func__', None) is not HcuDSV41Int8LinearMethod.apply
            or type(scheme) is not CompressedTensorsW8A8Int8
            or scheme.strategy != 'channel' or scheme.is_static_input_scheme
            or not scheme.input_symmetric
            or getattr(scheme.apply_weights, '__func__', None) is not CompressedTensorsW8A8Int8.apply_weights):
        return False
    kernel = scheme.kernel
    cls = triton.TritonInt8ScaledMMLinearKernel
    if (type(kernel) is not cls or getattr(kernel.apply_weights, '__func__', None) is not cls.apply_weights
            or tuple(kernel.layer_param_names) != ('weight', 'weight_scale', 'input_scale',
                                                    'input_zero_point', 'azp_adj')
            or not kernel.config.is_channelwise or kernel.config.is_static_input_scheme
            or not kernel.config.input_symmetric or layer.input_scale is not None
            or layer.input_zero_point is not None or layer.azp_adj is not None):
        return False
    weight, scale = layer.weight, getattr(layer, 'weight_scale', None)
    witness = getattr(layer, '_hcu_int8_empty_witness', None)
    return (
        isinstance(weight, torch.Tensor) and isinstance(scale, torch.Tensor)
        and getattr(layer, '_hcu_int8_empty_state', None) == 'ready' and witness is not None
        and witness.scheme is scheme and witness.kernel is kernel and witness.weight is weight
        and weight.device == witness.device == device and weight.dtype == witness.dtype == torch.int8
        and tuple(weight.shape) == witness.shape == (_K, n)
        and weight.stride() == witness.stride == (1, _K)
        and weight.data_ptr() == witness.data_ptr
        and scale.device == device and scale.dtype == torch.float32
        and scale.shape == (n, 1) and scale.is_contiguous()
    )


def _full_decode_capture(cfg, owner, rows):
    # The pinned MRV2 FULL body records runtime NONE. PIECEWISE uses its own
    # marker, including the decoder-replay subgraph. Bounded replay can call
    # forward_kv in prefill and discard qr, so require decode-only metadata.
    compilation = cfg.compilation_config
    if (compilation.mode != CompilationMode.NONE
            or compilation.cudagraph_mode not in (CUDAGraphMode.FULL_AND_PIECEWISE,
                                                  CUDAGraphMode.FULL_DECODE_ONLY)
            or torch.compiler.is_compiling() or not is_forward_context_available()):
        return False
    context = get_forward_context()
    if (context.cudagraph_runtime_mode != CUDAGraphMode.NONE
            or context.ubatch_slices is not None or not isinstance(context.attn_metadata, dict)):
        return False
    meta = context.attn_metadata.get(owner.k_cache.prefix)
    return (type(meta) is DeepseekV32IndexerMetadata and meta.decode is not None
            and 0 < meta.num_decodes <= rows and meta.num_decode_tokens == rows
            and meta.num_prefills == meta.num_prefill_tokens == 0 and meta.prefill is None
            and torch.cuda.is_current_stream_capturing())


def _valid_pair(qr, scale):
    return (isinstance(qr, torch.Tensor) and qr.dtype == torch.int8 and qr.is_cuda
            and qr.ndim == 2 and 1 <= qr.shape[0] <= 256 and qr.shape[1] == _K
            and qr.is_contiguous() and isinstance(scale, torch.Tensor)
            and scale.device == qr.device and scale.dtype == torch.float32
            and scale.shape == (qr.shape[0], 1) and scale.is_contiguous())


def share(cfg, device, attention, qr, qr_scale):
    """Return the native pair or the local INT8 pair used by both consumers."""
    owner = attention.indexer
    if (qr_scale is not None or not isinstance(qr, torch.Tensor)
            or qr.dtype != torch.bfloat16 or not qr.is_cuda or qr.device.index != device
            or torch.cuda.current_device() != device
            or qr.ndim != 2 or not 1 <= qr.shape[0] <= 256 or qr.shape[1] != _K
            or attention.use_pcp or attention.q_lora_rank != _K
            or (attention.n_local_heads, attention.head_dim) != (64, 512)
            or (owner.n_head, owner.head_dim, owner.q_lora_rank) != (32, 128, _K)
            or not _full_decode_capture(cfg, owner, qr.shape[0])
            or not _layer_matches(attention.wq_b, qr.device, indexer=False)
            or not _layer_matches(owner.wq_b, qr.device, indexer=True)):
        return qr, qr_scale
    with torch.profiler.record_function('hcu::dsv41_int8_wq_b_quant'):
        payload, scale, azp = _native_quant()(qr.contiguous(), None, None, symmetric=True)
    if azp is not None or not _valid_pair(payload, scale) or payload.shape != qr.shape:
        raise RuntimeError('INT8 WQ-B native dynamic quant returned an invalid operand')
    _LOG.info_once('INT8 WQ-B shared operand captured')
    return payload, scale


def configure_direct_gemm(cfg):
    """Resolve the opt-in legacy GEMM for eager execution and graph capture."""
    device = _configured_int8_device(cfg)
    if device is None:
        return None
    try:
        from lightop.gemm_ops import hipblaslt_w8a8_gemm

        expected = ('a', 'b', 'scale_a', 'scale_b', 'm', 'n', 'k',
                    'transpose_flag', 'out_dtype', 'bias')
        if tuple(inspect.signature(hipblaslt_w8a8_gemm).parameters) == expected:
            from .environment import _flag

            gemm = hipblaslt_w8a8_gemm
            if _flag('HCU_DSV41_INT8_DIRECT_CAPTURE_LOG', default=False):
                gemm = partial(_log_direct_capture, gemm)
            return device, gemm
    except (ImportError, AttributeError, ValueError, TypeError):
        pass
    _LOG.warning_once('INT8 direct GEMM unavailable: unsupported LightOp API')
    return None


_logged_capture_shapes = set()


def _log_direct_capture(gemm, a, b, scale_a, scale_b, m, n, k, transpose_flag, out_dtype):
    result = gemm(a, b, scale_a, scale_b, m, n, k, transpose_flag, out_dtype)
    shape = (k, n)
    if result[0] is True and shape not in _logged_capture_shapes and torch.cuda.is_current_stream_capturing():
        _logged_capture_shapes.add(shape)
        _LOG.warning('INT8 direct GEMM captured K=%d N=%d', k, n)
    return result


def project_direct(binding, layer, qr, qr_scale):
    """Use the same GEMM for BF16 input and a shared INT8 operand."""
    device, gemm = binding
    if (torch.compiler.is_compiling() or not isinstance(qr, torch.Tensor)
            or not qr.is_cuda or qr.device.index != device or qr.ndim != 2
            # Legacy LightOp allocates its output on the current CUDA device.
            or torch.cuda.current_device() != device
            or not 1 <= qr.shape[0] <= 256 or qr.shape[1] != _K
            or not _layer_matches(layer, qr.device, indexer=False)):
        return None
    if _valid_pair(qr, qr_scale):
        return _run_direct_gemm(gemm, layer, qr, qr_scale, _K, _MAIN_N, torch.bfloat16)
    if qr.dtype == torch.bfloat16 and qr_scale is None:
        return _direct_gemm(gemm, layer, qr, _K, _MAIN_N)
    return None


def _direct_gemm(gemm, layer, x, k, n):
    payload, scale, azp = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    rows = x.shape[0]
    if (azp is not None or payload.shape != x.shape or payload.dtype != torch.int8
            or payload.device != x.device or not payload.is_contiguous()
            or scale.shape != (rows, 1) or scale.dtype != torch.float32
            or scale.device != x.device or not scale.is_contiguous()):
        raise RuntimeError('INT8 quantization returned an invalid operand')
    return _run_direct_gemm(gemm, layer, payload, scale, k, n, x.dtype)


def _run_direct_gemm(gemm, layer, payload, scale, k, n, dtype):
    rows = payload.shape[0]
    status, out = gemm(payload, layer.weight.t(), scale, layer.weight_scale,
                       rows, n, k, 'NT', dtype)
    if (status is not True or not isinstance(out, torch.Tensor)
            or out.shape != (rows, n) or out.dtype != dtype or out.device != payload.device):
        raise RuntimeError('INT8 direct GEMM returned an invalid result')
    return out


def bind_decoder_direct_gemm(binding, attention, moe):
    """Attach direct GEMM to the three audited CT kernel instances."""
    if binding is None:
        return
    shared = moe.shared_experts
    gate_up = getattr(shared, 'gate_up_proj', None)
    for layer, k, n in ((attention.fused_wqa_wkv, 5120, 1792),
                        (attention.wo_b, 8192, 5120), (gate_up, 5120, 4608)):
        if (layer is None or type(layer.quant_method) is not HcuDSV41Int8LinearMethod
                or layer.tp_size != 1 or layer.bias is not None):
            continue
        scheme = layer.scheme
        if type(scheme) is not CompressedTensorsW8A8Int8:
            continue
        kernel = scheme.kernel
        cls = triton.TritonInt8ScaledMMLinearKernel
        if (type(kernel) is cls and kernel.config.is_channelwise
                and not kernel.config.is_static_input_scheme and kernel.config.input_symmetric
                and getattr(kernel.apply_weights, '__func__', None) is cls.apply_weights):
            kernel.apply_weights = partial(_apply_decoder_direct, kernel.apply_weights,
                                           binding, k, n, layer)


def _apply_decoder_direct(original, binding, k, n, owner, layer, x, bias=None):
    device, gemm = binding
    if (layer is owner and bias is None and isinstance(x, torch.Tensor)
            and x.dtype == torch.bfloat16 and x.is_cuda and x.device.index == device
            and x.ndim == 2 and 1 <= x.shape[0] <= 256 and x.shape[1] == k
            and not torch.compiler.is_compiling() and torch.cuda.current_device() == device):
        weight, scale = layer.weight, layer.weight_scale
        witness = getattr(layer, '_hcu_int8_empty_witness', None)
        if (getattr(layer, '_hcu_int8_empty_state', None) == 'ready' and witness is not None
                and witness.scheme is layer.scheme and witness.kernel is layer.scheme.kernel
                and witness.weight is weight and weight.dtype == torch.int8
                and weight.device == scale.device == x.device and weight.shape == (k, n)
                and weight.stride() == (1, k) and scale.dtype == torch.float32
                and scale.shape in ((n,), (n, 1)) and scale.is_contiguous()
                and layer.input_scale is None and layer.input_zero_point is None
                and layer.azp_adj is None):
            return _direct_gemm(gemm, layer, x, k, n)
    return original(layer, x, bias)


def project(layer, qr, qr_scale, *, indexer):
    if not _valid_pair(qr, qr_scale):
        raise ValueError('INT8 WQ-B requires contiguous INT8 [M,1280] and FP32 [M,1]')
    if not _layer_matches(layer, qr.device, indexer=indexer):
        raise RuntimeError('INT8 WQ-B projection no longer matches its native contract')
    label = 'hcu::dsv41_int8_wq_b_index' if indexer else 'hcu::dsv41_int8_wq_b_main'
    with torch.profiler.record_function(label):
        return torch.ops.vllm.int8_scaled_mm_func.default(
            qr, layer.weight, qr_scale, layer.weight_scale, torch.bfloat16, None
        )
