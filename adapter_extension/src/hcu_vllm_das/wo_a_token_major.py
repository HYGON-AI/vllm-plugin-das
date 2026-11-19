# SPDX-License-Identifier: Apache-2.0
"""BF16 G8 WO-A layout for the validated vLLM 0.33 HCU eager path."""

from functools import lru_cache, wraps
from hashlib import sha256
import inspect
from logging import getLogger
import os
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

from .environment import _flag

_SWITCH = "HCU_VLLM_WO_A_TOKEN_MAJOR"
_CAPABILITY = "dsv41-wo-a-token-major"
_PHASE_ROWS = "_das_wo_a_eager_decode_rows"
_LOCK = RLock()
_publication = None
_log = getLogger(__name__)


@lru_cache(maxsize=1)
def _supported_default():
    from hcu_adapter.bridges.vllm import _installed_vllm_version, detect_hcu

    if _installed_vllm_version() != '0.33.0+dtk2610.torch2130.2611101131.g7364da':
        return False
    detected = detect_hcu()
    return detected.detected and detected.architecture == 'gfx938'


def _enabled():
    if _SWITCH in os.environ:
        return _flag(_SWITCH, default=False)
    return _supported_default()


def _phase_rows(batch, np):
    """Check V2 target-runner host data, independent of the SWA split heuristic."""
    rows, reqs = batch.num_tokens, batch.num_reqs
    if (not 2 <= rows <= 32 or not 0 < reqs <= rows or batch.pcp is not None
            or batch.num_reqs_after_padding != reqs
            or batch.num_tokens_after_padding != rows):
        return None
    phase = batch.is_prefilling_np
    prompt_lens = batch.prefill_len_np
    completed = batch.num_computed_prefill_tokens_np
    # Dummy input has zero prefill lengths and all-false phase bits. A one-token
    # short extend can have the same SWA metadata counters as a true decode.
    if (not isinstance(phase, np.ndarray) or phase.dtype != np.bool_
            or phase.shape != (reqs,) or phase.any()):
        return None
    if any(not isinstance(values, np.ndarray) or values.shape != (reqs,)
           or not np.issubdtype(values.dtype, np.integer)
           for values in (prompt_lens, completed)):
        return None
    if (prompt_lens <= 0).any() or (completed < prompt_lens).any():
        return None
    return rows


def _wrap_prepare(original, state_cls, swa_cls, eager, no_graph, np):
    @wraps(original)
    def prepare(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                attn_groups, kv_cache_config, for_capture=False, ubatch_idx=0,
                model_specific_attn_metadata=None):
        metadata = original(
            self, input_batch, cudagraph_mode, block_tables, slot_mappings,
            attn_groups, kv_cache_config, for_capture, ubatch_idx,
            model_specific_attn_metadata,
        )
        if (type(self) is state_cls and self.model_config.enforce_eager is True
                and self.vllm_config.compilation_config.mode is eager
                and self.vllm_config.compilation_config.cudagraph_mode is no_graph
                and cudagraph_mode is no_graph and not for_capture and ubatch_idx == 0
                and model_specific_attn_metadata is None and isinstance(metadata, dict)):
            rows = _phase_rows(input_batch, np)
            if rows is not None:
                for swa in metadata.values():
                    if (type(swa) is swa_cls and swa.num_decode_tokens == rows
                            and swa.num_decodes == input_batch.num_reqs
                            and swa.num_prefills == swa.num_prefill_tokens == 0):
                        setattr(swa, _PHASE_ROWS, rows)
        return metadata

    return prepare


def _token_major_bmm(torch, o, weight):
    result = torch.empty((o.shape[0], 8, 1024), dtype=o.dtype, device=o.device)
    with torch.no_grad():
        torch.bmm(o.transpose(0, 1), weight.transpose(1, 2), out=result.transpose(0, 1))
    return result


def _wrap_o_proj(original, attention_classes, swa_cls, cached_weight, torch,
                 get_context, has_context, no_graph):
    @wraps(original)
    def project(self, attn_out, positions):
        # This exact AMD reference path gets BF16 rows after inverse RoPE.
        # QuantizedActivation, other backends, and alternate weight/row layouts
        # continue to use the original projection including its WO-B dispatch.
        if (type(self) not in attention_classes or not isinstance(attn_out, torch.Tensor)
                or attn_out.ndim != 3 or not 2 <= attn_out.shape[0] <= 32
                or self.n_local_groups != 8 or self.o_lora_rank != 1024
                or attn_out.shape[1] != self.n_local_heads
                or self.n_local_heads * attn_out.shape[2] != 8 * 4096
                or not attn_out.is_cuda or attn_out.dtype != torch.bfloat16
                or not attn_out.is_contiguous() or torch.compiler.is_compiling()
                or not has_context()):
            return original(self, attn_out, positions)
        context = get_context()
        if (context.cudagraph_runtime_mode is not no_graph
                or not isinstance(context.attn_metadata, dict)
                or context.ubatch_slices is not None):
            return original(self, attn_out, positions)
        swa = context.attn_metadata.get(self.swa_cache_layer.prefix)
        rows = attn_out.shape[0]
        if (type(swa) is not swa_cls or getattr(swa, _PHASE_ROWS, None) != rows
                or attn_out.device.index != torch.cuda.current_device()
                or torch.cuda.is_current_stream_capturing()):
            return original(self, attn_out, positions)
        # Reuse vLLM's dequantized BF16 weight cache, including its convention
        # for already-BF16 weights carrying stale scale metadata.
        weight = cached_weight(self.wo_a, 8, 1024, 4096)
        if (weight.shape != (8, 1024, 4096) or weight.dtype != torch.bfloat16
                or weight.device != attn_out.device or not weight.is_contiguous()):
            return original(self, attn_out, positions)
        z = _token_major_bmm(torch, attn_out.view(rows, 8, 4096), weight)
        return self._wo_b_after_wo_a(z.flatten(1))

    return project


def _require_source(function, expected):
    # Pinned to the inspected vLLM 0.33 HCU wheel g7364da. An ABI match alone
    # cannot prove that another build preserved phase or inverse RoPE ownership.
    if (not inspect.isfunction(function) or hasattr(function, "__wrapped__")
            or sha256(inspect.getsource(function).encode()).hexdigest() != expected):
        raise RuntimeError(f"{_CAPABILITY} vLLM source differs: {function!r}")


def _check_fp4_inheritance(attention, fp4_attention, fp4_project=None):
    # Compare descriptors so a classmethod is checked against its actual owner.
    methods = ("forward", "_alloc_attn_out", "get_padded_num_q_heads",
               "forward_mqa", "_wo_b_after_wo_a")
    if fp4_project is None:
        fp4_project = inspect.getattr_static(attention, "_o_proj")
    if (fp4_attention.__bases__[0] is not attention
            or inspect.getattr_static(fp4_attention, "_o_proj") is not fp4_project
            or any(inspect.getattr_static(fp4_attention, name) is not
                   inspect.getattr_static(attention, name) for name in methods)):
        raise RuntimeError(f"{_CAPABILITY} FP4 attention inheritance changed")


def owns_projection_wrapper(original):
    if _publication is None:
        return False
    _, _, classes, wrapper, _ = _publication
    return (classes[0]._o_proj is wrapper
            and getattr(wrapper, '__wrapped__', None) is original)


def _publish():
    import numpy as np
    import torch
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.models.deepseek_v41.amd import rocm
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    if not (torch.version.hip and current_platform.is_rocm()
            and current_platform.device_name == "hcu"):
        raise RuntimeError(f"{_CAPABILITY} requires the HCU ROCm platform")
    attention = rocm.DeepseekV41ROCMAiterMLAAttention
    swa = rocm.DeepseekV41ROCMAiterSparseSWAMetadata
    state = DeepseekV41ModelState
    prepare = state.prepare_attn
    project = attention._o_proj
    cached_weight = ops._get_cached_wo_a_bf16
    for function, expected in (
        (prepare, "cc910be7e99eb90b31cda804d2e963d83992a4a7120212faf574b6f07433032e"),
        (state._build_attn_metadata, "c8e13dddcc49918cd14d3a08de9c80eb3206193b6860515055b43f23dade1fd9"),
        (DefaultModelState.prepare_attn, "155bae76181475d6589eead3957a8d5b8be6fbd8ab058416f40fec8e95c92579"),
        (project, "050ac04cf793582bac2c53ed88d0d8d25e102474962e77264530ee68ecda6825"),
        (attention.forward_mqa, "ce088bdb6d90b52eff2f3ef986aefd60db246ff76e25a4e36090ae633c802b0b"),
        (attention._wo_b_after_wo_a, "adbefe2f49bd0900d4c87bc015f605399260bd769bcd76a5bd592d1a19ae5f61"),
        (cached_weight, "1c9dd6d7149af49550ce9ce5290cfe723de55904dae3697bed821adf997455b8"),
        (ops.rocm_inv_rope_einsum, "9b4fe284921cedf43cb649c736a80f5d4612d70f24d5368a023b1545a4af55c3"),
    ):
        _require_source(function, expected)
    if (rocm.rocm_inv_rope_einsum is not ops.rocm_inv_rope_einsum
            or project.__globals__ is not vars(rocm)
            or "_o_proj" not in attention.__dict__
            or "prepare_attn" not in state.__dict__):
        raise RuntimeError(f"{_CAPABILITY} projection or metadata owner changed")
    # The FP4 policy also selects the owned target at PCP1. Its registry
    # entry is lazy, so resolve the same class here before patching the base.
    from .adapter_api import adapter_module

    fp4_selected = adapter_module('fp4_indexer').selected

    attention_classes = (attention,)
    fp4_project = None
    if fp4_selected():
        from .fp4_indexer import Fp4Attention, _FP4_OUTPUT_PROJECTION

        fp4_project = _FP4_OUTPUT_PROJECTION
        _check_fp4_inheritance(attention, Fp4Attention, fp4_project)
        attention_classes = (attention, Fp4Attention)
    prepare_wrapper = _wrap_prepare(
        prepare, state, swa, CompilationMode.NONE, CUDAGraphMode.NONE, np,
    )
    project_wrapper = _wrap_o_proj(
        project, attention_classes, swa, cached_weight, torch, get_forward_context,
        is_forward_context_available, CUDAGraphMode.NONE,
    )
    try:
        state.prepare_attn = prepare_wrapper
        attention._o_proj = project_wrapper
    except BaseException as error:
        raise ExtensionCommitError(f"{_CAPABILITY} publication interrupted") from error
    return state, prepare_wrapper, attention_classes, project_wrapper, fp4_project


def _check_health():
    if _publication is not None:
        state, prepare, attention_classes, project, fp4_project = _publication
        if (not _enabled() or state.prepare_attn is not prepare
                or attention_classes[0]._o_proj is not project):
            raise ExtensionCommitError(f"{_CAPABILITY} policy or binding changed; restart")
        if len(attention_classes) == 2:
            try:
                _check_fp4_inheritance(*attention_classes, fp4_project)
            except RuntimeError as error:
                raise ExtensionCommitError(f"{_CAPABILITY} FP4 binding changed; restart") from error


def _activate(context):
    global _publication
    with _LOCK:
        _check_health()
        if _publication is None and _enabled():
            from .extension import _require_vllm_033_release_line

            _require_vllm_033_release_line(context.vllm_version)
            _publication = _publish()
            _log.info("%s enabled for HCU vLLM V2 eager decode/target verification", _CAPABILITY)


def extension_spec():
    """Advertise WO-A when enabled on this process's startup stack."""
    enabled = _enabled()
    return ExtensionSpec(
        name=_CAPABILITY, api_version=1, priority=95,
        capabilities=frozenset({_CAPABILITY}) if enabled else frozenset(),
        activate=_activate, check_health=_check_health if enabled else None,
    )
