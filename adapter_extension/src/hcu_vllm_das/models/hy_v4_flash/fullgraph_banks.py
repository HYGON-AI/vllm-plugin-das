# SPDX-License-Identifier: Apache-2.0
"""Request-budget admission and ownership for paired Core FULL graph banks.

This module imports no framework. Core remains responsible for descriptors,
capture algorithms, PIECEWISE graphs and output storage.
"""
from contextlib import contextmanager
from numbers import Integral


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def derive_short_bound(indexers):
    limits = []
    for owner in indexers:
        budget = _integer(getattr(owner, "group_budget", None))
        ratio = _integer(getattr(owner, "ratio", None))
        model_limit = _integer(getattr(owner, "max_model_len", None))
        if (budget is None or budget < 0 or ratio is None or ratio <= 0
                or model_limit is None or model_limit <= 0):
            return None
        limits.append((budget + 1) * ratio)
    return min(limits) if limits else None


def budget_fits_short(limits, lookahead, bound):
    """Only CPU integer budgets are admitted; tensor conversion is forbidden."""
    lookahead, bound = _integer(lookahead), _integer(bound)
    if lookahead is None or lookahead < 0 or bound is None or bound <= 0:
        return False
    largest = 0
    try:
        values = iter(limits)
    except TypeError:
        return False
    for value in values:
        value = _integer(value)
        if value is None or value <= 0:
            return False
        largest = max(largest, value)
    return largest > 0 and largest + lookahead <= bound


@contextmanager
def _capture_bound(indexers, bound):
    changed = []
    try:
        for owner in indexers:
            if getattr(owner, "_hcu_fullgraph_capture_bound", None) is not None:
                raise RuntimeError("Flash indexer already has a capture-bound owner")
            owner._hcu_fullgraph_capture_bound = bound
            changed.append(owner)
        yield
    finally:
        for owner in changed:
            owner._hcu_fullgraph_capture_bound = None


class FlashFullGraphBanks:
    def __init__(self, wide, short, indexers, bound, full_mode, *, short_pool_factory):
        self._wide, self._short = wide, short
        self._indexers = tuple(indexers)
        limit = derive_short_bound(self._indexers)
        if limit is None or _integer(bound) is None or not 0 < bound <= limit:
            raise ValueError("Flash short graph bound exceeds the indexer contract")
        self._bound, self._full_mode = int(bound), full_mode
        self._short_pool_factory = short_pool_factory
        # These banks replay in request-dependent order. Sharing the allocator
        # pool can alias live output storage with the other bank's temporaries.
        self._renew_short_pool()
        if (list(wide._capture_descs.get(full_mode, ()))
                != list(short._capture_descs.get(full_mode, ()))):
            raise ValueError("Flash graph banks require identical FULL descriptors")
        # The original manager captures model-owned PIECEWISE wrappers once.
        short._capture_descs = {
            full_mode: list(short._capture_descs.get(full_mode, ()))
        }
        self._allow_short = False
        self._selected = None
        self._failed = False

    def __getattr__(self, name):
        return getattr(self._wide, name)

    def _renew_short_pool(self):
        pool = self._short_pool_factory()
        if pool is None or pool == self._wide.pool or pool == self._short.pool:
            raise ValueError("Flash short bank requires a fresh private graph pool")
        self._short.pool = pool

    def _healthy(self):
        if self._failed:
            raise RuntimeError("Flash graph-bank capture failed; restart the worker")

    @property
    def pool(self):
        return self._wide.pool

    @pool.setter
    def pool(self, value):
        self._healthy()
        if value == self._short.pool:
            raise ValueError("Flash graph banks must keep separate graph pools")
        self._wide.pool = value

    @property
    def _max_full_descs_to_capture(self):
        return self._wide._max_full_descs_to_capture

    @_max_full_descs_to_capture.setter
    def _max_full_descs_to_capture(self, value):
        self._healthy()
        self._wide._max_full_descs_to_capture = value
        # Core extrapolates one bank from its first two samples. Measure the
        # complete short bank instead; its bytes stay in capture_model's real
        # measured delta and must not be subtracted or extrapolated again.
        self._short._max_full_descs_to_capture = None

    @property
    def _capture_mem_samples(self):
        return self._wide._capture_mem_samples

    @_capture_mem_samples.setter
    def _capture_mem_samples(self, value):
        self._healthy()
        self._wide._capture_mem_samples = value
        self._short._capture_mem_samples = None

    @property
    def _capture_descs(self):
        result = dict(self._wide._capture_descs)
        if self._wide._capture_mem_samples is not None:
            result[self._full_mode] = list(
                self._wide._capture_descs.get(self._full_mode, ())
            )
            return result
        result[self._full_mode] = [
            *self._wide._capture_descs.get(self._full_mode, ()),
            *self._short._capture_descs.get(self._full_mode, ()),
        ]
        return result

    def needs_capture(self):
        self._healthy()
        return self._wide.needs_capture() or self._short.needs_capture()

    def capture(self, *args, **kwargs):
        self._healthy()
        self.use_wide_bank()
        try:
            if any(getattr(owner, "_hcu_fullgraph_capture_bound", None) is not None
                   for owner in self._indexers):
                raise RuntimeError("Flash indexer capture-bound owner changed")
            self._wide.capture(*args, **kwargs)
            with _capture_bound(self._indexers, self._bound):
                self._short.capture(*args, **kwargs)
        except BaseException:
            # Partial capture may hold graph-pool/KV references. Never repair
            # that state by falling back to one of the partially built banks.
            self._failed = True
            raise

    def set_request_limits(self, limits, lookahead):
        self._healthy()
        self._allow_short = budget_fits_short(limits, lookahead, self._bound)

    def use_wide_bank(self):
        self._healthy()
        self._allow_short = False
        self._selected = None

    def dispatch(self, *args, **kwargs):
        self._healthy()
        descriptor = self._wide.dispatch(*args, **kwargs)
        self._selected = (
            self._short if self._allow_short and descriptor.cg_mode == self._full_mode
            else self._wide
        )
        return descriptor

    def run_fullgraph(self, descriptor):
        self._healthy()
        if self._selected is None:
            raise RuntimeError("Flash graph-bank replay requires a Core dispatch")
        return self._selected.run_fullgraph(descriptor)

    def run_pw_graph(self, *args, **kwargs):
        self._healthy()
        return self._wide.run_pw_graph(*args, **kwargs)

    def captured_token_counts(self):
        self._healthy()
        return self._wide.captured_token_counts()

    def release_graphs(self):
        self._healthy()
        self.use_wide_bank()
        try:
            self._wide.release_graphs()
            self._short.release_graphs()
            # A released private pool can have allocator use_count zero. Never
            # reuse its handle when the public release/capture lifecycle repeats.
            self._renew_short_pool()
        except BaseException:
            self._failed = True
            raise
