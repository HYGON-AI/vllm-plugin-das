# SPDX-License-Identifier: Apache-2.0
"""CPU guard proofs for graph/eager selection without GPU initialization."""

import importlib.util
import types
import unittest
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "step5_hot_context_subject",
    Path(__file__).parents[2] / "vllm_hcu/v1/step5_hot_context.py",
)
_subject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_subject)
conservative_seq_upper_bound = _subject.conservative_seq_upper_bound
needs_context_eager = _subject.needs_context_eager
check_hot_graph_context = _subject.check_hot_graph_context


class ContextGuardTests(unittest.TestCase):
    def runner(self, computed, pending=()):
        return types.SimpleNamespace(
            _hsl_enabled=True,
            _hsl_exact=np.asarray(computed, dtype=np.int32),
            _hsl_pending=(np.arange(len(pending)), np.asarray(pending), True)
            if len(pending)
            else None,
        )

    def batch(self, indices, queries):
        return types.SimpleNamespace(
            idx_mapping_np=np.asarray(indices), num_scheduled_tokens=np.asarray(queries)
        )

    def test_short_queries_do_not_wait_for_rejection_correction(self):
        r = self.runner([65536, 69000], [4, 4])
        b = self.batch([0, 1], [4, 4])
        self.assertEqual(conservative_seq_upper_bound(r, b), 69008)
        self.assertFalse(needs_context_eager(r, b, False, 8736))

    def test_context_boundary_and_long_context(self):
        for computed, expected in [(69880, False), (69881, True), (999992, True)]:
            r = self.runner([computed], [4])
            b = self.batch([0], [4])
            self.assertEqual(needs_context_eager(r, b, False, 8736), expected)

    def test_last_large_prefill_is_conservatively_eager(self):
        r = self.runner([65536], [32768])
        b = self.batch([0], [4])
        self.assertTrue(needs_context_eager(r, b, False, 8736))

    def test_reused_slot_cannot_underestimate(self):
        r = self.runner([10, 900000], [32768])
        b = self.batch([0, 1], [4, 4])
        self.assertTrue(needs_context_eager(r, b, False, 8736))

    def test_dummy_and_missing_mirror(self):
        r = self.runner([999999], [4])
        b = self.batch([0], [4])
        self.assertFalse(needs_context_eager(r, b, True, 8736))
        r._hsl_enabled = False
        self.assertTrue(needs_context_eager(r, b, False, 8736))

    def test_exact_context_assertion_prevents_unsafe_replay(self):
        full = types.SimpleNamespace(name="FULL")
        none = types.SimpleNamespace(name="NONE")
        short = types.SimpleNamespace(seq_lens_cpu=np.asarray([65536, 69640]))
        long = types.SimpleNamespace(seq_lens_cpu=np.asarray([1000000]))
        check_hot_graph_context(short, full, True, 8736)
        check_hot_graph_context(long, none, True, 8736)
        check_hot_graph_context(long, full, False, 8736)
        with self.assertRaises(RuntimeError):
            check_hot_graph_context(long, full, True, 8736)

    def test_disabled_guard_preserves_original_behavior(self):
        r = self.runner([1000000], [4])
        b = self.batch([0], [4])
        self.assertFalse(needs_context_eager(r, b, False, 0))


if __name__ == "__main__":
    unittest.main()
