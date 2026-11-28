# SPDX-License-Identifier: Apache-2.0
"""Gather host-backed FP8-DS MLA prefill on gfx936 using Core and LightOp."""

import ast
from contextlib import contextmanager
from functools import cache
import hashlib
import inspect
import textwrap
from types import FunctionType

import torch

_GATHER_GROUP_SIZE = 512
_CORE_FP8_METADATA_ABI = "29b984d53657c5ed06d381236a2c34e2504964e160287424c956947dbd5a6fd7"


def bind_hisparse_fp8_metadata_builder(metadata_factory):
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadataBuilder

    original = FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode
    source = textwrap.dedent(inspect.getsource(original))
    node, = ast.parse(source).body
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if digest != _CORE_FP8_METADATA_ABI:
        raise RuntimeError("HiSparse FP8 metadata builder changed in Core")
    # Keep Core's request splitting and workspace logic. Substitute only the
    # vendor scheduler factory in a private function namespace.
    return FunctionType(
        original.__code__, dict(original.__globals__, get_mla_metadata=metadata_factory),
        original.__name__, original.__defaults__, original.__closure__,
    )


@cache
def get_compact_prefill_gather():
    try:
        from lightop.op import decode_gather_and_up_convert_with_indices
    except ImportError:
        return None
    from hcu_adapter._base import Missing

    op = decode_gather_and_up_convert_with_indices
    return op if callable(op) and not isinstance(op, Missing) else None


class HiSparseFP8PrefillGather:
    """Stage the layer's batch once and upconvert each Core workspace chunk."""

    def __init__(self, index_group):
        self.index_group = index_group
        self.staged = None
        self.plan = None

    def __call__(
        self,
        layer_index,
        source_cache,
        dst,
        block_table,
        workspace_starts,
        batch_size,
        attn_metadata,
        request_start,
    ):
        from vllm.utils.torch_utils import current_stream

        gather = get_compact_prefill_gather()
        if gather is None:
            raise RuntimeError("LightOp FP8-DS MLA prefill gather is unavailable")
        group = self.index_group
        if self.plan is None:
            # Core computes the resident row mapping on the forward stream.
            self.plan = group._prefill_gather_plan(layer_index, attn_metadata)
        plan = self.plan
        cache = group.cache(layer_index)
        pages = plan.block_table[request_start : request_start + batch_size]
        if (
            pages.shape[0] != block_table.shape[0]
            or workspace_starts.numel() != batch_size
        ):
            raise ValueError(
                "HiSparse prefill chunk geometry does not match its staging plan"
            )

        stream = group.prefill_stream
        assert stream is not None and cache.view is not None
        # This also orders reuse of the caller's BF16 workspace after the
        # previous chunk consumed it on the forward stream.
        stream.wait_stream(current_stream())
        with torch.cuda.stream(stream):
            if self.staged is None:
                self.staged = cache.runtime.gather_prefill_cache(
                    source_cache,
                    plan,
                    resident_cache=cache.view.cache,
                )

            rows = dst.shape[0]
            if rows:
                # Starts are rebased within the chunk, while plan page IDs
                # index the entire staged layer. Resolve both on the device.
                padded_rows = (
                    (rows + _GATHER_GROUP_SIZE - 1) // _GATHER_GROUP_SIZE
                ) * _GATHER_GROUP_SIZE
                row = torch.arange(padded_rows, device=dst.device, dtype=torch.int32)
                live = row.clamp_max(rows - 1)
                req = torch.bucketize(
                    live, workspace_starts[1:], right=True, out_int32=True
                )
                pos = live - workspace_starts[req]
                bs = plan.block_size
                slots = pages[req, pos // bs] * bs + pos % bs
                slots.masked_fill_(row >= rows, -1)
                slots = slots.view(-1, _GATHER_GROUP_SIZE)
                limits = (rows - row[::_GATHER_GROUP_SIZE]).clamp_max_(
                    _GATHER_GROUP_SIZE
                )
                remap = torch.empty_like(slots)
                gathered = dst.new_empty((*slots.shape, dst.shape[1]))
                # LightOp's ABI uses contiguous physical pages of 64 tokens.
                gather(self.staged.view(-1, 64, 656), slots, gathered, limits, remap, 1)
                torch.index_select(
                    gathered.flatten(0, 1),
                    0,
                    remap.flatten()[:rows],
                    out=dst,
                )
            ready = group.prefill_ready_events[layer_index]
            ready.record(stream)
        return ready


@contextmanager
def hisparse_fp8_prefill_scope(index_group):
    """Give one Core layer forward a private HiSparse FP8 staging lifetime."""
    namespace = vars(index_group)
    missing = object()
    previous = namespace.get("gather_fp8_prefill", missing)
    index_group.gather_fp8_prefill = HiSparseFP8PrefillGather(index_group)
    try:
        yield
    finally:
        if previous is missing:
            del index_group.gather_fp8_prefill
        else:
            index_group.gather_fp8_prefill = previous
