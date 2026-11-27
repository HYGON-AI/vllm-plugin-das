"""Actual Plugin initialize after Core auto-fit; Core/device are explicit doubles."""

import os
from types import SimpleNamespace as NS
import unittest

import test_hyv4_flash_middle_fullgraph_runner as existing


class AutoFitRunnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = existing.RunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.runner = self.fixture.runner

    def test_legal_auto_fit_installs_banks_without_rewriting_owner_capacity(self):
        # Actual Core RPC synchronization is qualified in the separate source
        # audit. This fixture begins at the resulting initialize boundary.
        self.runner.max_model_len = 991488
        self.runner.initialize_kv_cache()
        banks = self.runner.cudagraph_manager
        self.assertIsInstance(banks, self.fixture.banks.FlashFullGraphBanks)
        self.assertEqual(len(self.fixture.constructed), 3)
        self.assertEqual([o.max_model_len for o in self.fixture.owners],
                         [1048576, 1048576])
        self.assertIs(banks._wide, self.fixture.constructed[0])
        self.assertEqual(banks._scoring_banks,
                         ((16384, self.fixture.constructed[2]),))

    def test_capacity_predicate_accepts_only_homogeneous_positive_limits(self):
        covers = self.fixture.banks.indexer_capacity_covers
        cases = [
            ([1048576, 1048576], 1048576, True),
            ([1048576, 1048576], 991488, True),
            ([991488, 991488], 991488, True),
            ([1048576], 16384, True),
            ([1048576, 991488], 991488, False),
            ([991488], 1048576, False),
            ([], 991488, False),
        ]
        invalid = (None, 0, -1, True, False, 1048576.0, "1048576")
        cases.extend(([value], 1, False) for value in invalid)
        cases.extend(([1048576], value, False) for value in invalid)
        for capacities, effective, expected in cases:
            with self.subTest(capacities=capacities, effective=effective):
                owners = [NS(max_model_len=value) for value in capacities]
                self.assertIs(covers(owners, effective), expected)
        self.assertFalse(covers([NS()], 1))

    def test_heterogeneous_constructor_capacities_keep_core_wide(self):
        self.runner.max_model_len = 991488
        self.fixture.owners[1].max_model_len = 991488
        self.runner.initialize_kv_cache()
        self.assertIs(type(self.runner.cudagraph_manager), self.fixture.CoreManager)
        self.assertEqual(len(self.fixture.constructed), 1)

    def test_invalid_effective_limits_refuse_without_comparison_errors(self):
        for value in (None, 0, -1, True, False, 1048576.0, "991488"):
            with self.subTest(value=value):
                self.runner.max_model_len = value
                self.runner.initialize_kv_cache()
                self.assertIs(type(self.runner.cudagraph_manager), self.fixture.CoreManager)
        self.assertEqual(self.fixture.properties_calls, [])

    def test_effective_growth_and_short_bound_refuse(self):
        for value in (1048577, 2048, 2047, 1):
            with self.subTest(value=value):
                self.runner.max_model_len = value
                self.runner.initialize_kv_cache()
                self.assertIs(type(self.runner.cudagraph_manager), self.fixture.CoreManager)
        self.assertEqual(self.fixture.properties_calls, [])

    def test_invalid_or_empty_owners_keep_core_wide(self):
        original = self.fixture.owners[:]
        for value in (None, 0, -1, True, 1048576.0, "1048576"):
            with self.subTest(value=value):
                for owner in original:
                    owner.max_model_len = value
                self.runner.initialize_kv_cache()
                self.assertIs(type(self.runner.cudagraph_manager), self.fixture.CoreManager)
        self.fixture.owners.clear()
        self.runner.initialize_kv_cache()
        self.assertIs(type(self.runner.cudagraph_manager), self.fixture.CoreManager)

    def test_auto_fit_exact_middle_limit_does_not_duplicate_wide(self):
        self.runner.max_model_len = 16384
        self.runner.initialize_kv_cache()
        banks = self.runner.cudagraph_manager
        self.assertIsInstance(banks, self.fixture.banks.FlashFullGraphBanks)
        self.assertEqual(len(self.fixture.constructed), 2)
        self.assertEqual(banks._scoring_banks, ())
        self.assertEqual([o.max_model_len for o in self.fixture.owners],
                         [1048576, 1048576])
        self.assertEqual(self.fixture.properties_calls, [])

    def test_auto_fit_below_middle_retains_only_short_and_wide(self):
        self.runner.max_model_len = 8192
        self.runner.initialize_kv_cache()
        banks = self.runner.cudagraph_manager
        self.assertIsInstance(banks, self.fixture.banks.FlashFullGraphBanks)
        self.assertEqual(banks._scoring_banks, ())
        self.assertEqual(len(self.fixture.constructed), 2)

    def test_auto_fit_master_off_keeps_two_independent_banks(self):
        os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = "off"
        self.runner.max_model_len = 991488
        self.runner.initialize_kv_cache()
        banks = self.runner.cudagraph_manager
        self.assertEqual(banks._scoring_banks, ())
        self.assertIsNot(banks._wide.pool, banks._short.pool)
        self.assertEqual(len(self.fixture.constructed), 2)
        self.assertTrue(self.fixture.runner_api.is_qualified_config(
            self.fixture.cfg, breakable=False))

    def test_auto_fit_capture_and_budget_handoff_preserve_three_owners(self):
        self.runner.max_model_len = 991488
        self.runner.initialize_kv_cache()
        banks = self.runner.cudagraph_manager
        banks.capture("model")
        self.assertEqual(len({id(m.pool) for m in self.fixture.constructed}), 3)
        for budgets, dummy, expected in (
            ([2042], False, "short"),
            ([2043, 100], False, "middle"),
            ([16378], False, "middle"),
            ([16379, 1], False, "wide"),
            ([100], True, "wide"),
        ):
            with self.subTest(budgets=budgets, dummy=dummy):
                self.runner.gather_batch_req_state(dict(budgets=budgets), dummy)
                self.assertEqual(banks.run_fullgraph(banks.dispatch()),
                                 expected + "-owned-output")
        for owner in self.fixture.owners:
            self.assertEqual(owner.max_model_len, 1048576)
            self.assertIsNone(owner._hcu_fullgraph_capture_bound)
            self.assertIsNone(owner._hcu_fullgraph_scoring_bound)

    def test_profile_then_auto_fit_initialize_keeps_frozen_middle_policy(self):
        self.runner.initialize_kv_cache()
        previous = self.runner.cudagraph_manager
        previous.capture("profile-model")
        previous.release_graphs()
        self.runner.max_model_len = 991488
        self.runner.initialize_kv_cache()
        current = self.runner.cudagraph_manager
        self.assertIsNot(current, previous)
        self.assertEqual(len(self.fixture.constructed), 6)
        self.assertEqual(len(current._scoring_banks), 1)
        self.assertEqual(self.fixture.properties_calls, ["cuda:0"])
        self.assertIsNot(current._short.pool, previous._short.pool)
        self.assertEqual([o.max_model_len for o in self.fixture.owners],
                         [1048576, 1048576])


if __name__ == "__main__":
    unittest.main()
