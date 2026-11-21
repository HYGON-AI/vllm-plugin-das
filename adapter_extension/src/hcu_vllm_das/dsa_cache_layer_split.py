# SPDX-License-Identifier: Apache-2.0
"""Main-source and SWA cache ownership for DeepSeek V4.1 eager PCP.

PR #370 (279aeee) and PR #243 (8c827d9) supply the owned-plus-scratch
design. Logical cache specs stay complete and groups match across ranks. Placement
and budget are in dsa_cache_layout, through the platform builder seam and the
Adapter worker's pre-allocation callback. Main sources use source-list owners;
SWA uses cyclic hidden-layer owners. Both write only on the owner rank.

Each remote read broadcasts into the scratch region already in the KV pool.
A budgeted trailing wire region holds a live-ID bitmap and packs only referenced
pages in bounded chunks, including cache record bytes. All ranks participate,
including ranks with zero query rows. Fresh prefill can reuse its page plan
and a main source that is still resident in scratch.
The current path is eager, PP/TP/DP/DCP=1, with fixed DSpark(5) verification.
PD requires the explicit owner-aware Plugin Mooncake connector on both ends.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import re

import torch
import torch.distributed as dist
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from .layer_split_config import layer_split_enabled

from .dsa_prefill_pages import active_prefill_pages

logger = init_logger('vllm.hcu_dsv41_layer_split')
def get_layer_shard_range(
    rank: int, shard_size: int, total_layers: int
) -> tuple[int, int]:
    """Contiguous ``[start, end)`` range owned by ``rank`` out of ``shard_size``.

    Split as evenly as possible; the first ``total_layers % shard_size``
    ranks own one extra item. Ported verbatim from PR #370.
    """
    if shard_size <= 0:
        raise ValueError(f'shard_size must be > 0; got {shard_size}')
    if total_layers < 0:
        raise ValueError(f'total_layers must be >= 0; got {total_layers}')
    if not 0 <= rank < shard_size:
        raise ValueError(f'rank must be in [0, {shard_size}); got {rank}')
    base = total_layers // shard_size
    rem = total_layers % shard_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def get_layer_owner(local_idx: int, shard_size: int, total_layers: int) -> int:
    """Rank that owns ``local_idx`` under the contiguous split above."""
    if not 0 <= local_idx < total_layers:
        raise ValueError(
            f'local_idx={local_idx} out of range for total_layers={total_layers}'
        )
    for rank in range(shard_size):
        start, end = get_layer_shard_range(rank, shard_size, total_layers)
        if start <= local_idx < end:
            return rank
    raise ValueError(
        f'Invalid local_idx={local_idx} for shard_size={shard_size}, '
        f'total_layers={total_layers}'
    )


def kv_source_owner_rank(
    kv_source_layer_id: int, kv_source_layer_ids: tuple[int, ...], pcp_size: int
) -> int:
    """PCP rank that owns ``kv_source_layer_id``'s main-KV cache.

    Shards across the ordered kv-source layer list itself (e.g. 4 entries
    for DeepSeek V4.1), not across all hidden layers; see module docstring.
    """
    source_index = kv_source_layer_ids.index(kv_source_layer_id)
    return get_layer_owner(source_index, pcp_size, len(kv_source_layer_ids))


def _prefix_owner_rank(
    prefix: str, kv_source_layer_ids: tuple[int, ...], pcp_size: int
) -> int:
    """PCP rank that owns the kv-source layer embedded in ``prefix``.

    Shared by every call site that only has a module prefix string to
    identify its layer (the indexer cache's own ``self.prefix``, the
    compressor's ``k_cache_prefix``, the indexer's ``k_cache.prefix``) —
    all of them equal ``f"{attention_prefix}.indexer.k_cache"`` or
    ``attention_prefix`` itself for a kv-source layer.
    """
    from vllm.model_executor.models.utils import extract_layer_index

    layer_id = extract_layer_index(prefix)
    return kv_source_owner_rank(layer_id, kv_source_layer_ids, pcp_size)


def indexer_cache_owner_rank(
    prefix: str, kv_source_layer_ids: tuple[int, ...], pcp_size: int
) -> int:
    """PCP rank that owns the indexer K cache living at ``prefix``.

    ``DeepseekV4IndexerCache`` is only ever constructed on a kv-source
    layer (see ``vllm/models/deepseek_v41/attention.py``), at
    ``f"{attention_prefix}.indexer.k_cache"``; its layer id parses out the
    same way the main-KV ownership check uses ``kv_source_layer_id``.
    """
    return _prefix_owner_rank(prefix, kv_source_layer_ids, pcp_size)


def swa_cache_owner_rank(prefix: str, pcp_size: int) -> int:
    """Cyclic layer ownership; independent of the four main-KV source IDs.

    The real native SWA groups interleave layers with stride seven. Cyclic
    ownership spreads each group's persistent pages over the PCP ranks.
    All planner, write, read and PD paths must use this same mapping.
    """
    match = re.search(r'(?:^|\.)layers\.(\d+)\.attn\.swa_cache$', prefix)
    if match is None or pcp_size < 1:
        raise ValueError(f'Invalid SWA layer ownership: {prefix}, PCP={pcp_size}')
    return int(match[1]) % pcp_size


def owner_swa_write_metadata(metadata, prefix):
    """Mask only non-owner writes; keep PCP gathers and query metadata intact.

    Native PCP gathers before inserting. Negative gathered slots suppress all
    cache stores, including the separate full-batch forward_kv pass used by
    decoder bounded replay. Never mutate metadata shared by other layers or
    by the original and replay forward contexts.
    """
    if _state is None or not isinstance(metadata, dict):
        return metadata
    width, _, rank = _state
    if swa_cache_owner_rank(prefix, width) == rank:
        return metadata
    meta = metadata.get(prefix)
    slots = getattr(meta, 'cache_slot_mapping', None)
    if not isinstance(slots, torch.Tensor):
        raise RuntimeError(f'DSA SWA gathered write slots missing for {prefix}')
    local = copy.copy(meta)
    local.cache_slot_mapping = torch.full_like(slots, -1)
    result = dict(metadata)
    result[prefix] = local
    return result


_state: tuple[int, tuple[int, ...], int] | None = None
_source_block_sizes: dict[str, int] = {}
_transport_regions: dict[str, tuple[int, int, int]] = {}


def register_transport_regions(regions):
    global _transport_regions
    if any(offset < 0 or offset % 8 or size <= 0 or count <= 0
           for offset, size, count in regions.values()):
        raise ValueError('Invalid independent KV transport region')
    _transport_regions = dict(regions)


def register_source_block_sizes(block_sizes: dict[str, int]) -> None:
    """Bind main/SWA planner block sizes (uncompressed tokens) on each worker.

    Do not infer these from cache.shape: ratio-2 caches contain half as many
    states as logical tokens, and indexer metadata has no block_size field.
    """
    if any(not isinstance(size, int) or size <= 0 for size in block_sizes.values()):
        raise ValueError('DSA source block sizes must be positive integers')
    global _source_block_sizes
    _source_block_sizes = dict(block_sizes)


def split_config(vllm_config) -> tuple[int, tuple[int, ...]] | None:
    if not layer_split_enabled(vllm_config):
        return None
    p = vllm_config.parallel_config
    ids = tuple(getattr(vllm_config.model_config.hf_config, 'kv_source_layer_ids', ()) or ())
    if p.prefill_context_parallel_size <= 1 or not ids:
        return None
    if ids != tuple(sorted(set(ids))):
        raise ValueError('DSA layer split requires unique ordered KV source layers')
    if (p.tensor_parallel_size != 1 or p.pipeline_parallel_size != 1
            or p.data_parallel_size != 1 or p.decode_context_parallel_size != 1):
        raise ValueError('DSA layer split requires TP/PP/DP/DCP=1')
    if not vllm_config.model_config.enforce_eager:
        raise ValueError('DSA layer split requires eager execution')
    if vllm_config.speculative_config is not None:
        from .pcp_dspark import validate_dspark_config
        validate_dspark_config(vllm_config)
    validate_layer_split_transfer(vllm_config)
    return p.prefill_context_parallel_size, ids


def validate_layer_split_transfer(config):
    transfer = config.kv_transfer_config
    if transfer is None:
        return
    if (getattr(transfer, 'kv_connector', None) != 'HcuLayerSplitMooncakeConnector'
            or getattr(transfer, 'kv_connector_module_path', None) != 'hcu_vllm_das.mooncake_layer_split'
            or getattr(transfer, 'kv_role', None) != 'kv_producer'):
        raise ValueError('DSA layer split requires standalone execution or the '
                         'owner-aware HcuLayerSplitMooncakeConnector producer')


def record_layer_split_state(vllm_config):
    global _state
    config = split_config(vllm_config)
    if config is None:
        _state = None
        register_source_block_sizes({})
        register_transport_regions({})
        return None
    width, ids = config
    state = (width, ids, get_pcp_group().rank_in_group)
    if not 0 <= state[2] < width:
        raise ValueError('DSA layer split PCP rank is outside the configured group')
    if _state != state:
        _state = state
        owned = [i for i in ids if kv_source_owner_rank(i, ids, width) == state[2]]
        logger.info('DSA layer split PCP rank %d/%d owns KV sources %s', state[2], width, owned)
    return state


def layer_split_state():
    return _state


def _is_owner(prefix: str) -> bool | None:
    if _state is None:
        return None
    width, ids, rank = _state
    return _prefix_owner_rank(prefix, ids, width) == rank


_COMPRESSOR_INSERT_CACHE_REVIEWED = (
    '9adfce223bd7284a896222164ab30321f746ea1b7c9af643b9d11dac0c236c04'
)
_INDEXER_PRODUCE_K_REVIEWED = (
    '92c8e8864a8e1ee3fcf6f555f1333f96cd1580e6f592a4b781f90eb47b022144'
)
_write_suppression_applied = False


def apply_kv_source_write_suppression() -> None:
    """Hash-pinned no-op guard on the two kv-source cache *write* paths.

    ``DeepseekCompressor.insert_cache`` and ``DeepseekV4Indexer._produce_k``
    are each called unconditionally on every PCP rank for a kv-source layer
    (native PCP's ``all_gather``-before-write makes every rank compute and
    commit an identical copy; see the module docstring). Both wrappers
    early-return before touching their cache tensor when this rank does not
    own the layer, so only the owner's allocation (placed by the layer-split cache builder)
    ever gets written.

    Deliberately does NOT add, skip, or reorder any collective call: both
    native bodies only reach collectives (``get_pcp_group().all_gather``)
    *before* this early-return point, in the caller
    (``DeepseekV4Attention._prepare_and_attn`` / ``forward_kv``), not inside
    either patched method. Every PCP rank still calls these two methods
    exactly once per forward step, in the same order; only the ownership
    check's *data* path differs, not the call/collective count — so this
    cannot by itself desynchronize the ranks the way adding a new
    conditional collective would.
    """
    global _write_suppression_applied
    from vllm.models.deepseek_v41.attention import DeepseekV4Indexer
    from vllm.models.deepseek_v41.compressor import DeepseekCompressor

    targets = (
        (DeepseekCompressor, 'insert_cache', _COMPRESSOR_INSERT_CACHE_REVIEWED),
        (DeepseekV4Indexer, '_produce_k', _INDEXER_PRODUCE_K_REVIEWED),
    )
    if _write_suppression_applied:
        if all(getattr(getattr(cls, name), '_hcu_layer_split_wrapper', False) for cls, name, _ in targets):
            return
        raise RuntimeError('DSA kv-source write suppression was replaced by someone else')
    natives = []
    for cls, name, expected_hash in targets:
        native = getattr(cls, name)
        actual = hashlib.sha256(inspect.getsource(native).encode()).hexdigest()
        if actual != expected_hash:
            raise RuntimeError(
                f'HCU DSA kv-source write source changed: {native.__qualname__}: {actual}'
            )
        natives.append(native)

    native_insert_cache, native_produce_k = natives

    def insert_cache(self, latent, positions, rotary_emb):
        # Invalidate on every PCP rank, including non-writers, or ranks could
        # diverge on whether the next read enters a collective.
        pages = active_prefill_pages()
        if pages is not None:
            pages.source_written(self.k_cache_prefix)
        if _is_owner(self.k_cache_prefix) is False:
            return
        return native_insert_cache(self, latent, positions, rotary_emb)

    def produce_k(self, latent, positions, rotary_emb):
        if _is_owner(self.k_cache.prefix) is False:
            return
        return native_produce_k(self, latent, positions, rotary_emb)

    insert_cache._hcu_layer_split_wrapper = True
    produce_k._hcu_layer_split_wrapper = True
    DeepseekCompressor.insert_cache = insert_cache
    DeepseekV4Indexer._produce_k = produce_k
    _write_suppression_applied = True


def _live_block_ids(metadata: dict, prefix: str, num_blocks: int) -> list[int]:
    """Resolve live physical IDs from the *root* request table and lengths.

    v0.33 root indexer seq_lens are uncompressed request lengths. Its decode
    lengths are compressed and prefill chunk lengths may be PCP-padded; neither
    is safe for deciding ownership of overlaid physical blocks. The main MLA
    metadata omits seq_lens, so use its paired indexer's root lengths, while
    retaining the MLA's OWN table and planner block size.

    This path is eager-only. Host metadata copies are intentional; do not turn
    an unknown metadata schema into a full-pool broadcast (that corrupts other
    groups). Block zero is the native null block and is never written.
    """
    if prefix.endswith('.swa_cache'):
        return _swa_live_block_ids(metadata[prefix], prefix, num_blocks)
    block_size = _source_block_sizes.get(prefix)
    if block_size is None:
        raise RuntimeError(f'DSA source block size was not registered: {prefix}')
    meta = metadata[prefix]
    table = getattr(meta, 'block_table', None)
    lengths_meta = meta
    lengths = getattr(meta, 'seq_lens', None)
    if lengths is None and not prefix.endswith('.indexer.k_cache'):
        lengths_meta = metadata.get(prefix + '.indexer.k_cache')
        lengths = getattr(lengths_meta, 'seq_lens', None)
    # Native PCP's zero-query placeholder deliberately omits block_table.
    # It still participates in the bitmap reduction, then receives the live
    # pages requested by the other ranks. Main MLA may retain a nominal
    # request row, so use its paired indexer's actual local token counts.
    if (getattr(lengths_meta, 'num_decodes', None) == 0
            and getattr(lengths_meta, 'num_prefills', None) == 0
            and getattr(lengths_meta, 'num_decode_tokens', None) == 0
            and getattr(lengths_meta, 'num_prefill_tokens', None) == 0):
        return []
    if not isinstance(table, torch.Tensor) or not isinstance(lengths, torch.Tensor):
        raise RuntimeError(f'DSA live block table/lengths missing for {prefix}')
    if table.ndim != 2 or lengths.ndim != 1:
        raise RuntimeError(f'DSA root block table/length shape changed for {prefix}')
    count = getattr(meta, 'num_reqs', None)
    if count is None:
        decodes = getattr(lengths_meta, 'num_decodes', None)
        prefills = getattr(lengths_meta, 'num_prefills', None)
        if decodes is None or prefills is None:
            raise RuntimeError(f'DSA live request count missing for {prefix}')
        count = decodes + prefills
    if not isinstance(count, int) or not 0 <= count <= min(table.shape[0], lengths.numel()):
        raise RuntimeError(f'DSA live request count invalid for {prefix}')
    if lengths_meta is not meta and (
        getattr(lengths_meta, 'num_decodes', -1)
        + getattr(lengths_meta, 'num_prefills', -1) != count
    ):
        raise RuntimeError(f'DSA paired indexer request count differs for {prefix}')
    table_cpu = table[:count].detach().cpu()
    lengths_cpu = lengths[:count].detach().cpu().tolist()
    selected = set()
    for row, length in enumerate(lengths_cpu):
        if not isinstance(length, int) or length < 0:
            raise RuntimeError(f'DSA sequence length invalid for {prefix}')
        used = (length + block_size - 1) // block_size
        if used > table.shape[1]:
            raise RuntimeError(f'DSA block table is shorter than sequence for {prefix}')
        for block_id in table_cpu[row, :used].tolist():
            if not isinstance(block_id, int) or not 0 <= block_id < num_blocks:
                raise RuntimeError(f'DSA live block ID out of range for {prefix}: {block_id}')
            if block_id != 0:
                selected.add(block_id)
    return sorted(selected)


def _swa_live_block_ids(meta, prefix, num_blocks):
    """Use the exact native read extents, never a full historical table prefix.

    Expired SWA IDs may already belong to other groups. Decode indices address
    physical token slots; prefill gathers [seq_len-gather_len, seq_len). The
    latter comes from native metadata, including PCP and replay window bounds,
    not an assumed fixed 128-row suffix of a possibly large prefill chunk.
    """
    block_size = _source_block_sizes.get(prefix)
    if block_size is None or getattr(meta, 'block_size', None) != block_size:
        raise RuntimeError(f'DSA SWA block size missing or inconsistent: {prefix}')
    nd = getattr(meta, 'num_decode_tokens', None)
    np = getattr(meta, 'num_prefill_tokens', None)
    if not isinstance(nd, int) or not isinstance(np, int) or nd < 0 or np < 0:
        raise RuntimeError(f'DSA SWA token counts invalid: {prefix}')
    selected = set()

    def add(block):
        if not isinstance(block, int) or not 0 <= block < num_blocks:
            raise RuntimeError(f'DSA SWA live block ID out of range: {prefix}: {block}')
        if block:
            selected.add(block)

    if nd:
        indices = getattr(meta, 'decode_swa_indices', None)
        lengths = getattr(meta, 'decode_swa_lens', None)
        # Native V41 emits [tokens, 1, window]; some backend views squeeze
        # the single KV-head dimension. No multi-head layout is supported.
        if isinstance(indices, torch.Tensor) and indices.ndim == 3 and indices.shape[1] == 1:
            indices = indices.squeeze(1)
        if (not isinstance(indices, torch.Tensor) or indices.ndim != 2
                or not isinstance(lengths, torch.Tensor) or lengths.ndim != 1
                or min(indices.shape[0], lengths.numel()) < nd):
            raise RuntimeError(f'DSA SWA decode read metadata missing: {prefix}')
        rows = indices[:nd].detach().cpu().tolist()
        for row, length in zip(rows, lengths[:nd].detach().cpu().tolist()):
            if not isinstance(length, int) or not 0 <= length <= len(row):
                raise RuntimeError(f'DSA SWA decode extent invalid: {prefix}')
            for slot in row[:length]:
                if not isinstance(slot, int) or slot < -1:
                    raise RuntimeError(f'DSA SWA decode slot invalid: {prefix}')
                if slot >= 0:
                    add(slot // block_size)
    if np:
        count = getattr(meta, 'num_prefills', None)
        start = getattr(meta, 'num_decodes', None)
        table = getattr(meta, 'block_table', None)
        lengths = getattr(meta, 'prefill_seq_lens', None)
        gathers = getattr(meta, 'prefill_gather_lens', None)
        if (not isinstance(count, int) or count <= 0
                or not isinstance(start, int) or start < 0
                or not isinstance(table, torch.Tensor) or table.ndim != 2
                or start + count > table.shape[0]
                or not isinstance(lengths, torch.Tensor) or lengths.ndim != 1
                or not isinstance(gathers, torch.Tensor) or gathers.ndim != 1
                or min(lengths.numel(), gathers.numel()) < count):
            raise RuntimeError(f'DSA SWA prefill read metadata missing: {prefix}')
        rows = table[start:start + count].detach().cpu().tolist()
        for row, end, size in zip(rows, lengths[:count].detach().cpu().tolist(),
                                  gathers[:count].detach().cpu().tolist()):
            if (not isinstance(end, int) or not isinstance(size, int)
                    or not 0 <= size <= end):
                raise RuntimeError(f'DSA SWA prefill extent invalid: {prefix}')
            if not size:
                continue
            first, last = (end - size) // block_size, (end + block_size - 1) // block_size
            if last > len(row):
                raise RuntimeError(f'DSA SWA block table shorter than read extent: {prefix}')
            for block in row[first:last]:
                add(block)
    return sorted(selected)


def broadcast_kv_source_cache(prefix: str, owner_rank: int, local_cache: torch.Tensor):
    """Make the requested owner pages visible in the local scratch view.

    The scheduler supplies the global union for a fresh text prefill. Other
    batches reduce a live-ID bitmap. Both paths preserve physical coordinates
    and leave null pages and holes untouched.
    """
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return local_cache
    if prefix not in metadata:
        raise RuntimeError(f'DSA layer split metadata missing for {prefix}')
    if local_cache.ndim < 2 or local_cache.numel() == 0:
        raise RuntimeError(f'DSA layer split cache is unbound for {prefix}')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('DSA layer split broadcast requires eager execution')
    prefill = active_prefill_pages()
    planned_ids = None if prefill is None else prefill.ids(prefix, metadata)
    is_main = not prefix.endswith(('.indexer.k_cache', '.swa_cache'))
    if planned_ids is not None and is_main and prefill.ready(prefix):
        return local_cache
    pcp = get_pcp_group()
    storage = local_cache.untyped_storage()
    element_size = local_cache.element_size()
    byte_stride = local_cache.stride(0) * element_size
    byte_offset = local_cache.storage_offset() * element_size
    page_bytes = element_size * (1 + sum(
        (size - 1) * stride
        for size, stride in zip(local_cache.shape[1:], local_cache.stride()[1:])
    ))
    num_blocks = local_cache.shape[0]
    if _transport_regions:
        if prefix not in _transport_regions:
            raise RuntimeError(f'DSA transport plan missing for {prefix}')
        bitmap_offset, wire_size, planned_blocks = _transport_regions[prefix]
        end = bitmap_offset + wire_size
        if planned_blocks != num_blocks or end > storage.nbytes():
            raise RuntimeError(f'DSA transport plan/view mismatch for {prefix}')
        if byte_offset + (num_blocks - 1) * byte_stride + page_bytes > bitmap_offset:
            raise RuntimeError(f'DSA cache overlaps transport region for {prefix}')
    else:
        bitmap_offset = num_blocks * byte_stride
        end = storage.nbytes()
    ids_offset = (bitmap_offset + num_blocks + 7) // 8 * 8
    wire_offset = ids_offset + num_blocks * 8
    capacity = end - wire_offset
    if capacity <= 0 or page_bytes > byte_stride:
        raise RuntimeError(f'DSA layer split transport region missing for {prefix}')
    if planned_ids is not None:
        # The scheduler plan is already the same on every PCP rank. Both the
        # page count and the exact IDs are available without a GPU readback.
        _broadcast_pages(local_cache, pcp, owner_rank, planned_ids, num_blocks,
                         byte_offset, byte_stride, page_bytes, wire_offset, capacity)
        prefill.finish_transfer(prefix)
        return local_cache
    bitmap = torch.empty(0, dtype=torch.uint8, device=local_cache.device)
    bitmap.set_(storage, bitmap_offset, (num_blocks,), (1,))
    bitmap.zero_()
    device_ids = torch.empty(0, dtype=torch.int64, device=local_cache.device)
    device_ids.set_(storage, ids_offset // 8, (num_blocks,), (1,))
    error = None
    try:
        local_ids = _live_block_ids(metadata, prefix, num_blocks)
    except RuntimeError as exc:
        error = exc
        local_ids = []
        bitmap[0] = 1  # Null-block slot carries a collective failure flag.
    if local_ids:
        device_ids[:len(local_ids)].copy_(torch.tensor(local_ids, dtype=torch.int64))
        bitmap.index_fill_(0, device_ids[:len(local_ids)], 1)
    dist.all_reduce(bitmap, op=dist.ReduceOp.MAX, group=pcp.device_group)
    bitmap_cpu = bitmap.cpu()
    if bitmap_cpu[0].item():
        raise RuntimeError(f'DSA invalid live-block metadata on a PCP rank for {prefix}') from error
    host_ids = bitmap_cpu.nonzero().flatten().to(torch.int64)
    pages = host_ids.numel()
    if pages:
        device_ids[:pages].copy_(host_ids)
        _broadcast_pages(local_cache, pcp, owner_rank, device_ids[:pages], num_blocks,
                         byte_offset, byte_stride, page_bytes, wire_offset, capacity)
    return local_cache


def _broadcast_pages(local_cache, pcp, owner_rank, device_ids, num_blocks,
                     byte_offset, byte_stride, page_bytes, wire_offset, capacity):
    """Send exact physical pages without touching local holes or the null page."""
    pages = device_ids.numel()
    if not pages:
        return
    storage = local_cache.untyped_storage()
    raw_pages = torch.empty(0, dtype=torch.uint8, device=local_cache.device)
    raw_pages.set_(storage, byte_offset, (num_blocks, page_bytes), (byte_stride, 1))
    # Very small pools may have less than one page of wire capacity.
    # Split the byte columns too, so a one-block cache remains valid.
    for column in range(0, page_bytes, capacity):
        width = min(page_bytes - column, capacity)
        rows_per_chunk = max(1, capacity // width)
        for start in range(0, pages, rows_per_chunk):
            rows = min(rows_per_chunk, pages - start)
            wire = torch.empty(0, dtype=torch.uint8, device=local_cache.device)
            wire.set_(storage, wire_offset, (rows, width), (width, 1))
            region = raw_pages[:, column:column + width]
            selected_ids = device_ids[start:start + rows]
            if pcp.rank_in_group == owner_rank:
                torch.index_select(region, 0, selected_ids, out=wire)
            pcp.broadcast(wire.view(-1), src=owner_rank)
            if pcp.rank_in_group != owner_rank:
                region.index_copy_(0, selected_ids, wire)
