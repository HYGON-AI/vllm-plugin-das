# SPDX-License-Identifier: Apache-2.0
"""Host page plans and receive-slot validity for a fresh LayerSplit prefill."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageGroup:
    slot: int
    block_size: int  # Uncompressed request tokens per physical page.
    num_blocks: int
    prefixes: tuple[str, ...]
    sliding_window: int | None = None


def _fresh_prefill_requests(output):
    scheduled = output.num_scheduled_tokens
    if (not scheduled or output.scheduled_cached_reqs.req_ids
            or output.scheduled_spec_decode_tokens or output.scheduled_encoder_inputs
            or getattr(output, 'has_sync_kv_loads', False)
            or getattr(output, 'kv_connector_metadata', None) is not None
            or len(output.scheduled_new_reqs) != len(scheduled)):
        return None
    requests = {req.req_id: req for req in output.scheduled_new_reqs}
    if requests.keys() != scheduled.keys():
        return None
    for rid, count in scheduled.items():
        req = requests[rid]
        if (type(count) is not int or count <= 0 or req.num_computed_tokens
                or req.replay_start or req.prompt_token_ids is None
                or req.mm_features or count > len(req.prompt_token_ids)):
            return None
    return requests


def _request_pages(req, group, start, end):
    if group.slot >= len(req.block_ids):
        return None
    first = start // group.block_size
    stop = (end + group.block_size - 1) // group.block_size
    active = req.block_ids[group.slot][first:stop]
    if len(active) != stop - first or any(type(i) is not int or not 0 < i < group.num_blocks
                                         for i in active):
        return None
    return active


def fresh_prefill_page_ids(output, groups):
    """Exact global page union for new text requests with no previous KV."""
    requests = _fresh_prefill_requests(output)
    if requests is None:
        return None
    result = {}
    for group in groups:
        selected = set()
        for rid, count in output.num_scheduled_tokens.items():
            active = _request_pages(requests[rid], group, 0, count)
            if active is None:
                return None
            selected.update(active)
        result[group.slot] = tuple(sorted(selected))
    return result


def fresh_swa_prefill_page_ids(output, groups, *, replay_window=None, keep_prompt_rows=False):
    """Global SWA page unions for full, first replay and later replay views.

    PCP partitions a fresh prompt into contiguous segments. Each segment reads
    its query rows plus up to window-1 preceding rows; the union for the full
    view is the entire prompt. At the replay cut, the first layer already wrote
    the full KV and reads the kept tail plus its history. Later layers only
    wrote the tail, and Core floors their reads at the replay boundary.
    """
    requests = _fresh_prefill_requests(output)
    if requests is None:
        return None
    result = {}
    for group in groups:
        window = group.sliding_window
        if not window or (replay_window is not None and window != replay_window):
            return None
        full, first_replay, later_replay = set(), set(), set()
        trimmed = False
        for rid, count in output.num_scheduled_tokens.items():
            req = requests[rid]
            active = _request_pages(req, group, 0, count)
            if active is None:
                return None
            full.update(active)
            if replay_window is None:
                continue
            params = req.sampling_params
            keep = keep_prompt_rows or (params is not None and (
                params.prompt_logprobs is not None or params.prompt_logprob_token_ids is not None))
            tail_start = 0 if keep else max(0, count - replay_window)
            trimmed |= tail_start > 0
            # All subranges use coordinates already validated by the full view.
            halo_page = max(0, tail_start - window + 1) // group.block_size
            first_replay.update(active[halo_page:])
            later_replay.update(active[tail_start // group.block_size:])
        # At the cut, Core can retain zero-query PCP segments after another
        # request's first live prefill row. Their gather still includes the
        # old window halo. Keep native selection at this one layer for batches
        # where that can happen. Later replay floors those empty spans to zero.
        first_is_exact = replay_window is not None and (len(requests) == 1 or not trimmed)
        result[group.slot] = (
            tuple(sorted(full)),
            tuple(sorted(first_replay)) if first_is_exact else None,
            tuple(sorted(later_replay)) if replay_window is not None else None,
        )
    return result


@dataclass
class PrefillPages:
    by_prefix: dict
    group_by_prefix: dict[str, int]
    received: dict[int, str] = field(default_factory=dict)
    swa_replay_by_prefix: dict = field(default_factory=dict)
    replay_layers: object | None = None

    def ids(self, prefix, metadata):
        replay = None if self.replay_layers is None else self.replay_layers.replay_batch
        if (prefix.endswith('.swa_cache') and replay is not None
                and metadata is replay.forward_context.attn_metadata):
            return self.swa_replay_by_prefix.get(prefix)
        return self.by_prefix.get(prefix)

    def ready(self, prefix):
        group = self.group_by_prefix.get(prefix)
        return group is not None and self.received.get(group) == prefix

    def finish_transfer(self, prefix):
        group = self.group_by_prefix.get(prefix)
        if group is not None:
            # Remote main KV and index K can share the same receive area on
            # some ranks. All ranks must make the SAME collective decision.
            self.received[group] = prefix

    def source_written(self, prefix):
        group = self.group_by_prefix.get(prefix)
        if group is not None and self.received.get(group) == prefix:
            del self.received[group]


_active = ContextVar('hcu_layer_split_prefill_pages', default=None)


def active_prefill_pages():
    return _active.get()


@contextmanager
def prefill_pages_scope(pages):
    token = _active.set(pages)
    try:
        yield
    finally:
        _active.reset(token)
