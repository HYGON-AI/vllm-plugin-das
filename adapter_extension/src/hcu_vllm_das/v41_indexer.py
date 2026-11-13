# SPDX-License-Identifier: Apache-2.0
"""Plugin-side binding for the adapter-owned V4.1 FP8 indexer reference.

The reference scoring, the packed-cache decoding and the graph contract moved to
`hcu_adapter.v41_fp8_indexer`: the SHUFFLE page layout, the FP32 scale tail and
the candidate-block selection are upstream vLLM definitions, and filling the
behaviour the AITER-off HCU path cannot take is adapter work. The plugin keeps
only what is genuinely private to it:

* the FP4 cache branch (LightOp preflight and execution), and
* the OOT publication of the class defined here.

The module attributes below stay the single patch point for the whole scoring
path. The adapter reads them through resolvers bound at import, so
`patch.object(v41_indexer, ...)` keeps reaching code that now lives in the
adapter, exactly as it did when this module owned the implementation.
"""
from __future__ import annotations

from hcu_adapter import v41_fp8_indexer as _impl
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config as current_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.utils.torch_utils import _resolve_layer_name
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    FP8_DTYPE,
    cp_gather_indexer_k_quant_cache_triton,
    fp8_mqa_logits_torch,
    rocm_aiter_sparse_attn_indexer as native_indexer,
)

from . import v41_indexer_graph
from .environment import _flag

FLAG = 'VLLM_HCU_DSV41_FP8_INDEXER_REFERENCE'
logger = init_logger(__name__)


def enabled():
    # HCU V41 FP8 indexer reference path is enabled by default on v0.33:
    # the native FP8 indexer rejects AITER-off HCU before execution, so the
    # reference path is the only usable decode route there. Set the flag to 0
    # to fall back to the native path explicitly.
    return _flag(FLAG, default=True)


def _check_health():
    from .v41_indexer_activation import check_health

    return check_health()


def register_splitting_op(config):
    """Historical import path; the contract itself is adapter-owned."""
    return _impl.register_splitting_op(config)


def graph_capture_allowed(config):
    """Historical import path; the contract itself is adapter-owned."""
    return _impl.graph_capture_allowed(config)


def _register_split_hook(config):
    return globals()['register_splitting_op'](config)


# `graph_allowed` is bound by `v41_indexer_graph`, which owns that patch point
# and is imported above; both modules bind under this one activation owner.
_impl.bind_runtime_resolvers(
    v41_indexer_graph.OWNER,
    current_config=lambda: globals()['current_config'](),
    forward_context=lambda: globals()['get_forward_context'](),
    candidate_select=lambda: globals()['select_candidate_blocks'],
    candidate_mask=lambda: globals()['apply_candidate_mask'],
    gather_kv=lambda: globals()['cp_gather_indexer_k_quant_cache_triton'],
    enabled=lambda: globals()['enabled'](),
    health=_check_health,
    register_split=lambda: _register_split_hook,
    # Three names the old module owned that a caller may still patch. The
    # adapter reads them through these resolvers, so the patch keeps reaching
    # the code that now lives there. Types and dtypes are deliberately NOT
    # resolvers: patching a metadata class or a dtype is not a supported seam.
    native_indexer=lambda: globals()['native_indexer'],
    mqa_logits=lambda: globals()['fp8_mqa_logits_torch'],
    layer_name=lambda: globals()['_resolve_layer_name'],
)

_api = _impl.build()
indexer = _api['indexer']
local_topk = _api['local_topk']
decode_logits = _api['decode_logits']
_select = _api['select']
# Historical private surface: keep every name the old module owned resolvable,
# so a replay that reaches into the scorer module keeps working unchanged.
_cache_flat = _api['cache_flat']
_cache_pages = _api['cache_pages']
_fake = _api['fake']


def register_op():
    _api['register_op']()


class HcuV41SparseAttnIndexer(_api['base']):
    """Plugin-published OOT class; only the FP4 branch is plugin-private.

    Everything else (admission, scoring, candidate-block ABI) is inherited from
    the adapter base, which is where the upstream behaviour belongs.
    """

    def __init__(self, *args, **kwargs):
        from hcu_adapter.dsv41_channel_fp8 import _is_v41
        from .v41_indexer_activation import admit_config

        cfg = current_config()
        if _is_v41(cfg.model_config.hf_config):
            admit_config(cfg)
        super().__init__(*args, **kwargs)

    def _fp4_preflight(self, cfg):
        from .fp4_indexer import preflight

        return preflight(cfg)

    def _fp4_execute(self, hidden_states, q_quant, weights):
        from .fp4_indexer_lightop import execute

        return execute(self, hidden_states, q_quant, weights)
