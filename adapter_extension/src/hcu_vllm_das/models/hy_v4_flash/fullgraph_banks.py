# SPDX-License-Identifier: Apache-2.0
"""Request-budget admission and ownership for Core FULL graph banks.

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


def indexer_capacity_covers(indexers, max_model_len):
    """Allow Core auto-fit within one unchanged indexer constructor capacity."""
    effective = _integer(max_model_len)
    capacities = {
        _integer(getattr(owner, "max_model_len", None)) for owner in indexers
    }
    if effective is None or len(capacities) != 1 or None in capacities:
        return False
    capacity = capacities.pop()
    return 0 < effective <= capacity


def required_bound(limits, lookahead):
    """Only CPU integer budgets are admitted; tensor conversion is forbidden."""
    lookahead = _integer(lookahead)
    if lookahead is None or lookahead < 0:
        return None
    largest = 0
    try:
        values = iter(limits)
    except TypeError:
        return None
    for value in values:
        value = _integer(value)
        if value is None or value <= 0:
            return None
        largest = max(largest, value)
    return largest + lookahead if largest > 0 else None


def budget_fits_short(limits, lookahead, bound):
    bound = _integer(bound)
    required = required_bound(limits, lookahead)
    return bound is not None and bound > 0 and required is not None and required <= bound


def scoring_max_seq_len(indexer, default):
    """A scoring capacity never grants the short bank's unpruned semantics."""
    value = getattr(indexer, "_hcu_fullgraph_scoring_bound", None)
    if value is None:
        return default
    bound = _integer(value)
    limit = _integer(getattr(indexer, "max_model_len", None))
    if bound is None or limit is None or not 0 < bound <= limit:
        raise ValueError("Flash scoring graph bound exceeds the indexer capacity")
    if getattr(indexer, "_hcu_fullgraph_capture_bound", None) is not None:
        raise RuntimeError("Flash short and scoring capture owners conflict")
    return bound


@contextmanager
def _capture_bound(indexers, bound):
    changed = []
    try:
        for owner in indexers:
            if (getattr(owner, "_hcu_fullgraph_capture_bound", None) is not None
                    or getattr(owner, "_hcu_fullgraph_scoring_bound", None) is not None):
                raise RuntimeError("Flash indexer already has a capture-bound owner")
            owner._hcu_fullgraph_capture_bound = bound
            changed.append(owner)
        yield
    finally:
        for owner in changed:
            owner._hcu_fullgraph_capture_bound = None


@contextmanager
def _capture_scoring_bound(indexers, bound):
    changed = []
    try:
        for owner in indexers:
            if (getattr(owner, "_hcu_fullgraph_capture_bound", None) is not None
                    or getattr(owner, "_hcu_fullgraph_scoring_bound", None) is not None):
                raise RuntimeError("Flash indexer already has a capture-bound owner")
            value, limit = _integer(bound), _integer(getattr(owner, "max_model_len", None))
            if value is None or limit is None or not 0 < value <= limit:
                raise ValueError("Flash scoring graph bound exceeds the indexer capacity")
            owner._hcu_fullgraph_scoring_bound = value
            changed.append(owner)
        yield
    finally:
        for owner in changed:
            owner._hcu_fullgraph_scoring_bound = None


class FlashFullGraphBanks:
    def __init__(self, wide, short, indexers, bound, full_mode, *, short_pool_factory,
                 scoring_banks=()):
        if wide is short:
            raise ValueError("Flash graph banks require distinct manager owners")
        self._wide, self._short = wide, short
        self._indexers = tuple(indexers)
        limit = derive_short_bound(self._indexers)
        if limit is None or _integer(bound) is None or not 0 < bound <= limit:
            raise ValueError("Flash short graph bound exceeds the indexer contract")
        self._bound, self._full_mode = int(bound), full_mode
        self._scoring_banks = tuple(scoring_banks)
        previous = self._bound
        managers = [wide, short]
        model_limits = {int(owner.max_model_len) for owner in self._indexers}
        if self._scoring_banks and len(model_limits) != 1:
            raise ValueError("Flash scoring banks require one model limit")
        model_limit = min(model_limits)
        normalized = []
        for value, manager in self._scoring_banks:
            value = _integer(value)
            if (value is None or not previous < value < model_limit
                    or manager is None or any(manager is old for old in managers)):
                raise ValueError("Flash scoring banks require ordered distinct capacities and owners")
            normalized.append((value, manager))
            previous = value
            managers.append(manager)
        self._scoring_banks = tuple(normalized)
        self._private_managers = (short, *(manager for _, manager in self._scoring_banks))
        self._short_pool_factory = short_pool_factory
        # These banks replay in request-dependent order. Sharing the allocator
        # pool can alias live output storage with the other bank's temporaries.
        for manager in self._private_managers:
            if (list(wide._capture_descs.get(full_mode, ()))
                    != list(manager._capture_descs.get(full_mode, ()))):
                raise ValueError("Flash graph banks require identical FULL descriptors")
        self._renew_private_pools()
        # The original manager captures model-owned PIECEWISE wrappers once.
        for manager in self._private_managers:
            manager._capture_descs = {
                full_mode: list(manager._capture_descs.get(full_mode, ()))
            }
        self._request_bound = None
        self._selected = None
        self._failed = False

    def __getattr__(self, name):
        return getattr(self._wide, name)

    def _renew_private_pools(self):
        occupied = [self._wide.pool, *(manager.pool for manager in self._private_managers)]
        for manager in self._private_managers:
            pool = self._short_pool_factory()
            if pool is None or pool in occupied:
                raise ValueError("Flash bank requires a fresh private graph pool")
            manager.pool = pool
            occupied.append(pool)

    def _healthy(self):
        if self._failed:
            raise RuntimeError("Flash graph-bank capture failed; restart the worker")

    @property
    def pool(self):
        return self._wide.pool

    @pool.setter
    def pool(self, value):
        self._healthy()
        if any(value == manager.pool for manager in self._private_managers):
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
        # complete bounded banks instead; their bytes stay in capture_model's real
        # measured delta and must not be subtracted or extrapolated again.
        for manager in self._private_managers:
            manager._max_full_descs_to_capture = None

    @property
    def _capture_mem_samples(self):
        return self._wide._capture_mem_samples

    @_capture_mem_samples.setter
    def _capture_mem_samples(self, value):
        self._healthy()
        self._wide._capture_mem_samples = value
        for manager in self._private_managers:
            manager._capture_mem_samples = None

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
            *(desc for manager in self._private_managers
              for desc in manager._capture_descs.get(self._full_mode, ())),
        ]
        return result

    def needs_capture(self):
        self._healthy()
        return any(manager.needs_capture() for manager in (self._wide, *self._private_managers))

    def capture(self, *args, **kwargs):
        self._healthy()
        self.use_wide_bank()
        try:
            if any((getattr(owner, "_hcu_fullgraph_capture_bound", None) is not None
                    or getattr(owner, "_hcu_fullgraph_scoring_bound", None) is not None)
                   for owner in self._indexers):
                raise RuntimeError("Flash indexer capture-bound owner changed")
            self._wide.capture(*args, **kwargs)
            with _capture_bound(self._indexers, self._bound):
                self._short.capture(*args, **kwargs)
            for bound, manager in self._scoring_banks:
                with _capture_scoring_bound(self._indexers, bound):
                    manager.capture(*args, **kwargs)
        except BaseException:
            # Partial capture may hold graph-pool/KV references. Never repair
            # that state by falling back to one of the partially built banks.
            self._failed = True
            raise

    def set_request_limits(self, limits, lookahead):
        self._healthy()
        self._request_bound = required_bound(limits, lookahead)
        self._selected = None

    def use_wide_bank(self):
        self._healthy()
        self._request_bound = None
        self._selected = None

    def dispatch(self, *args, **kwargs):
        self._healthy()
        descriptor = self._wide.dispatch(*args, **kwargs)
        self._selected = self._wide
        if descriptor.cg_mode == self._full_mode and self._request_bound is not None:
            for bound, manager in ((self._bound, self._short), *self._scoring_banks):
                if self._request_bound <= bound:
                    self._selected = manager
                    break
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
            for manager in self._private_managers:
                manager.release_graphs()
            # A released private pool can have allocator use_count zero. Never
            # reuse its handle when the public release/capture lifecycle repeats.
            self._renew_private_pools()
        except BaseException:
            self._failed = True
            raise
