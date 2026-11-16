# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bounded, asynchronous cost calibration; never caches input validity."""

from collections import OrderedDict
from statistics import median
from threading import Lock


class _Entry:
    def __init__(self):
        self.choice = None
        self.reserved = 0
        self.samples = []
        self.uses = 0


class DispatchPolicy:
    def __init__(
        self, event_factory=None, max_entries=64, trials=3, refresh_calls=1024
    ):
        if max_entries < 1 or trials < 1 or refresh_calls < 1:
            raise ValueError(
                "cache size, trial count and refresh interval must be positive"
            )
        self.event_factory = event_factory
        self.max_entries = max_entries
        self.trials = trials
        self.refresh_calls = refresh_calls
        self.entries = OrderedDict()
        self.lock = Lock()

    def _event(self):
        if self.event_factory is not None:
            return self.event_factory()
        import torch

        return torch.cuda.Event(enable_timing=True)

    def _reserve(self, key):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                entry = self.entries[key] = _Entry()
                if len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
            self.entries.move_to_end(key)
            if entry.choice is not None and entry.uses >= self.refresh_calls:
                # Locality may change while all shapes stay the same.
                entry = self.entries[key] = _Entry()
            # query() is nonblocking. Never synchronize to make a decision.
            if (
                entry.choice is None
                and len(entry.samples) == self.trials
                and all(s[1].query() and s[3].query() for s in entry.samples)
            ):
                bf16 = median(s[0].elapsed_time(s[1]) for s in entry.samples)
                fp8 = median(s[2].elapsed_time(s[3]) for s in entry.samples)
                entry.choice = bf16 > 0 and fp8 < 0.9 * bf16
                entry.samples.clear()
            if entry.choice is not None:
                entry.uses += 1
                return entry, entry.choice, None
            if entry.reserved == self.trials:
                return entry, False, None
            sample_id = entry.reserved
            entry.reserved += 1
            return entry, None, sample_id

    def _reject(self, key, entry):
        with self.lock:
            if self.entries.get(key) is entry:
                entry.choice = False
                entry.samples.clear()

    def run(self, key, fp8, bf16, unsupported):
        entry, choice, sample_id = self._reserve(key)
        if choice is False:
            return bf16()
        if choice is True:
            try:
                # The callback checks this call's indices and counts again.
                return fp8()
            except unsupported:
                self._reject(key, entry)
                return bf16()

        events = [self._event() for _ in range(4)]
        bf16_output = None

        def measure(fn, begin, end):
            begin.record()
            output = fn()
            end.record()
            return output

        try:
            # Alternate order to reduce bias from cache warming / clock drift.
            if sample_id % 2 == 0:
                bf16_output = measure(bf16, events[0], events[1])
                measure(fp8, events[2], events[3])
            else:
                measure(fp8, events[2], events[3])
                bf16_output = measure(bf16, events[0], events[1])
        except unsupported:
            self._reject(key, entry)
            return bf16() if bf16_output is None else bf16_output
        except Exception:
            with self.lock:
                if self.entries.get(key) is entry:
                    del self.entries[key]
            raise
        with self.lock:
            if self.entries.get(key) is entry and entry.choice is None:
                entry.samples.append(events)
        # Calibration is optional performance work; return the BF16 result.
        return bf16_output


policy = DispatchPolicy()


def dispatch_key(q, kv, indices, sink, stream):
    # Bucket only the KV extent (4K rows); do not retain tensors or addresses.
    return (
        q.device.index,
        stream,
        tuple(q.shape),
        (kv.shape[0] + 4095) // 4096,
        tuple(q.stride()),
        tuple(kv.stride()),
        tuple(indices.stride()),
        None if sink is None else tuple(sink.stride()),
    )
