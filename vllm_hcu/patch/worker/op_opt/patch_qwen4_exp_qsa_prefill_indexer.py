# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route Qwen4Exp prefill-only QSA selection to a request-major scorer."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.v1.attention.ops.qsa_prefill_indexer import (
    qsa_select_paged_prefill,
)

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

logger = init_logger(f"vllm.{__name__}")

TARGET_MODULE = "vllm.models.qwen4_exp.amd.indexer_qsa"
PATCH_ID = "worker.op_opt.qwen4_exp.qsa.prefill_indexer"
_SELECT_TARGET = f"{TARGET_MODULE}.QSAIndexer._select"
TARGETS = (_SELECT_TARGET,)
_MARKER = "_vllm_hcu_qsa_prefill_indexer_applied"
_WRAPPER = "_vllm_hcu_qsa_prefill_indexer_select"
_RUNTIME_ENABLED = henvs.qsa_prefill_indexer_enabled()


def configure(enabled: bool) -> None:
    """Latch the master-gated policy serialized by the API process."""

    if not isinstance(enabled, bool):
        raise TypeError("QSA prefill-indexer worker policy must be bool")
    global _RUNTIME_ENABLED
    _RUNTIME_ENABLED = enabled


def _enabled() -> bool:
    return _RUNTIME_ENABLED


def _can_use_prefill_indexer(q: torch.Tensor, metadata: object) -> bool:
    num_tokens = int(getattr(metadata, "num_actual_tokens", 0))
    num_decodes = int(getattr(metadata, "num_decodes", -1))
    num_decode_tokens = int(getattr(metadata, "num_decode_tokens", -1))
    num_prefills = int(getattr(metadata, "num_prefills", -1))
    num_prefill_tokens = int(getattr(metadata, "num_prefill_tokens", -1))
    max_query_len = int(getattr(metadata, "max_query_len", 0))
    if not (
        num_tokens > 0
        and q.shape[0] == num_tokens
        and num_decodes == 0
        and num_decode_tokens == 0
        and num_prefills > 0
        and num_prefill_tokens == num_tokens
        and max_query_len > 0
    ):
        return False
    block_table = getattr(metadata, "block_table", None)
    query_start_loc = getattr(metadata, "query_start_loc", None)
    visible_blocks = getattr(metadata, "visible_blocks", None)
    logical_positions = getattr(metadata, "logical_positions", None)
    token_to_req = getattr(metadata, "token_to_req", None)
    sequence_lengths = getattr(metadata, "seq_lens", None)
    return (
        isinstance(block_table, torch.Tensor)
        and block_table.ndim == 2
        and block_table.shape[0] == num_prefills
        and isinstance(query_start_loc, torch.Tensor)
        and query_start_loc.shape == (num_prefills + 1,)
        and isinstance(visible_blocks, torch.Tensor)
        and visible_blocks.shape == (num_tokens,)
        and isinstance(logical_positions, torch.Tensor)
        and logical_positions.shape == (num_tokens,)
        and isinstance(token_to_req, torch.Tensor)
        and token_to_req.shape == (num_tokens,)
        and isinstance(sequence_lengths, torch.Tensor)
        and sequence_lengths.shape == (num_prefills,)
    )


def run_qsa_prefill_selection(indexer, q, metadata, out):
    """Run the specialized score kernel with vLLM's existing top-k/expand."""

    logger.info_once(
        "Qwen4Exp QSA routing: prefill-only selection uses the HCU "
        "request-major indexer",
        scope="local",
    )
    return qsa_select_paged_prefill(
        q,
        indexer.compressed_key_cache.kv_cache,
        metadata.block_table,
        metadata.query_start_loc,
        metadata.visible_blocks,
        metadata.logical_positions,
        metadata.token_to_req,
        metadata.seq_lens,
        indexer.token_topk,
        indexer.compress_ratio,
        metadata.max_query_len,
        out,
    )


def apply_to_module(module: ModuleType) -> bool:
    owner = load_exact_module(TARGET_MODULE, module)
    indexer_cls = require_class(owner, "QSAIndexer", _SELECT_TARGET)
    wrapped = ((indexer_cls, "_select", _SELECT_TARGET, _WRAPPER),)
    if already_applied(indexer_cls, _MARKER, wrapped):
        return False
    original_select = require_callable(indexer_cls, "_select", _SELECT_TARGET)
    require_exact_signature(
        original_select,
        _SELECT_TARGET,
        positional=("self", "q", "metadata", "out"),
    )

    @functools.wraps(original_select)
    def hcu_select(self, q, metadata, out):
        if _enabled() and _can_use_prefill_indexer(q, metadata):
            return run_qsa_prefill_selection(self, q, metadata, out)
        return original_select(self, q, metadata, out)

    setattr(hcu_select, _WRAPPER, True)
    setattr(indexer_cls, "_select", hcu_select)
    setattr(indexer_cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "configure",
    "run_qsa_prefill_selection",
]
