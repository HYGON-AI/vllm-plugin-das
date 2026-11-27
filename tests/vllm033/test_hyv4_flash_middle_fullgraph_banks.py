"""Real control module + explicit Core doubles. No Torch or device math."""

import ast
import importlib.util
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash"


def load_api():
    spec = importlib.util.spec_from_file_location(
        "middle_banks_under_test", SOURCE / "fullgraph_banks.py"
    )
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_AND_PIECEWISE = 3


def owner(**changes):
    return NS(
        **dict(
            dict(
                group_budget=511,
                ratio=4,
                max_model_len=1048576,
                _hcu_fullgraph_capture_bound=None,
                _hcu_fullgraph_scoring_bound=None,
            ),
            **changes,
        )
    )


class Manager:
    """Descriptor/capture/pool double only. Actual module controls all routing."""

    def __init__(self, name, owners):
        (self.name, self.owners) = (name, owners)
        self._capture_descs = {Mode.FULL: [6, 12, 384], Mode.PIECEWISE: [128]}
        self._capture_mem_samples = None
        self._max_full_descs_to_capture = None
        self.pool = object()
        self.graphs = {}
        self.calls = []
        self.fail_capture = False
        self.fail_release = False

    def capture(self, *args, **kwargs):
        self.calls.append(
            (
                "capture",
                tuple(
                    (
                        (o._hcu_fullgraph_capture_bound, o._hcu_fullgraph_scoring_bound)
                        for o in self.owners
                    )
                ),
                args,
                kwargs,
            )
        )
        if self.fail_capture:
            raise RuntimeError("capture witness")
        ds = self._capture_descs[Mode.FULL]
        if self._max_full_descs_to_capture is not None:
            ds = ds[: self._max_full_descs_to_capture]
        for d in ds:
            self.graphs[d] = object()
            if self._capture_mem_samples is not None:
                self._capture_mem_samples.append(d)

    def needs_capture(self):
        return not self.graphs

    def dispatch(self, *args, **kwargs):
        assert self.name == "wide", "only wide owns descriptor dispatch"
        self.calls.append(("dispatch", args, kwargs))
        return NS(cg_mode=kwargs.get("mode", Mode.FULL), key=12)

    def run_fullgraph(self, d):
        assert d.key in self.graphs, "missing real Core graph"
        self.calls.append(("full", d))
        return self.name + "-owned-output"

    def run_pw_graph(self, *a, **k):
        return self.name + "-pw"

    def captured_token_counts(self):
        return sorted(self.graphs)

    def release_graphs(self):
        self.calls.append(("release",))
        if self.fail_release:
            raise RuntimeError("release witness")
        self.graphs.clear()


class BankTests(unittest.TestCase):
    def setUp(self):
        self.api = load_api()
        self.owners = [owner(), owner()]
        (self.wide, self.short, self.middle) = [
            Manager(n, self.owners) for n in ("wide", "short", "middle")
        ]

    def banks(self, extra=True, **kwargs):
        options = dict(short_pool_factory=object)
        if extra:
            options["scoring_banks"] = ((16384, self.middle),)
        options.update(kwargs)
        return self.api.FlashFullGraphBanks(
            self.wide, self.short, self.owners, 2048, Mode.FULL, **options
        )

    def select(self, b, limits, lookahead=6, mode=Mode.FULL):
        b.set_request_limits(limits, lookahead)
        d = b.dispatch(mode=mode)
        return b.run_fullgraph(d) if mode == Mode.FULL else b.run_pw_graph("model", {})

    def test_required_budget_is_one_pass_cpu_integral(self):
        seen = []

        def limits():
            for x in [100, 4352]:
                seen.append(x)
                yield x

        self.assertEqual(self.api.required_bound(limits(), 6), 4358)
        self.assertEqual(seen, [100, 4352])

        class CpuInt(int):
            pass

        self.assertEqual(self.api.required_bound([CpuInt(2**40)], 6), 2**40 + 6)

    def test_unknown_inputs_never_coerce_tensor_like(self):

        class TensorLike:
            def __int__(self):
                raise AssertionError("device conversion")

            __index__ = __int__

        for limits, look in [
            ([], 6),
            ([0], 6),
            ([-1], 6),
            ([True], 6),
            ([1.0], 6),
            ([TensorLike()], 6),
            (None, 6),
            ([1], True),
            ([1], -1),
        ]:
            with self.subTest(limits=limits, look=look):
                self.assertIsNone(self.api.required_bound(limits, look))

    def test_short_and_middle_budget_boundary_plus_minus_six(self):
        b = self.banks()
        b.capture("model")
        for boundary in (2048, 16384):
            for delta in range(-6, 7):
                budget = boundary + delta
                expected = (
                    "short"
                    if budget + 6 <= 2048
                    else "middle"
                    if budget + 6 <= 16384
                    else "wide"
                )
                with self.subTest(boundary=boundary, delta=delta):
                    self.assertEqual(
                        self.select(b, [budget]), expected + "-owned-output"
                    )

    def test_mixed_budgets_mapping_order_and_long_fallback(self):
        b = self.banks()
        b.capture("model")
        for limits, expected in [
            ([100, 4352], "middle"),
            ([4352, 100], "middle"),
            ([100, 33000], "wide"),
            ([1048576], "wide"),
            ([True], "wide"),
        ]:
            self.assertEqual(self.select(b, iter(limits)), expected + "-owned-output")

    def test_no_extra_keeps_existing_two_banks(self):
        b = self.banks(extra=False)
        b.capture("model")
        self.assertEqual(self.select(b, [2042]), "short-owned-output")
        self.assertEqual(self.select(b, [2043]), "wide-owned-output")

    def test_piecewise_and_dummy_reset_to_original_wide(self):
        b = self.banks()
        b.capture("model")
        self.select(b, [4352])
        self.assertEqual(self.select(b, [4352], mode=Mode.PIECEWISE), "wide-pw")
        b.use_wide_bank()
        self.assertEqual(b.run_fullgraph(b.dispatch()), "wide-owned-output")

    def test_new_request_limits_require_fresh_dispatch(self):
        b = self.banks()
        b.capture("model")
        d = b.dispatch()
        b.set_request_limits([4352], 6)
        with self.assertRaisesRegex(RuntimeError, "dispatch"):
            b.run_fullgraph(d)

    def test_capture_contexts_and_output_ownership_are_distinct(self):
        b = self.banks()
        mark = object()
        b.capture(mark, aux=True)
        self.assertEqual(self.wide.calls[0][1], ((None, None), (None, None)))
        self.assertEqual(self.short.calls[0][1], ((2048, None), (2048, None)))
        self.assertEqual(self.middle.calls[0][1], ((None, 16384), (None, 16384)))
        for m in (self.wide, self.short, self.middle):
            self.assertEqual(m.calls[0][2:], ((mark,), {"aux": True}))
        self.assertIn(Mode.PIECEWISE, self.wide._capture_descs)
        self.assertNotIn(Mode.PIECEWISE, self.short._capture_descs)
        self.assertNotIn(Mode.PIECEWISE, self.middle._capture_descs)
        for limits, expected in [
            ([10], "short"),
            ([4352], "middle"),
            ([33000], "wide"),
            ([4352], "middle"),
        ]:
            self.assertEqual(self.select(b, limits), expected + "-owned-output")
        self.assertTrue(
            all(
                (
                    o._hcu_fullgraph_capture_bound
                    is o._hcu_fullgraph_scoring_bound
                    is None
                    for o in self.owners
                )
            )
        )

    def test_private_pools_and_global_setter_never_alias(self):
        b = self.banks()
        pools = [m.pool for m in (self.wide, self.short, self.middle)]
        self.assertEqual(len({id(p) for p in pools}), 3)
        for pool in pools[1:]:
            with self.assertRaises(ValueError):
                b.pool = pool

    def test_duplicate_or_reused_factory_pool_rejected(self):
        pool = object()
        with self.assertRaises(ValueError):
            self.banks(short_pool_factory=lambda: pool)

    def test_profile_samples_only_wide_but_charges_all_extra_graphs(self):
        b = self.banks()
        samples = []
        b._max_full_descs_to_capture = 2
        b._capture_mem_samples = samples
        b.capture("profile")
        self.assertEqual(samples, [6, 12])
        self.assertEqual(len(self.short.graphs), 3)
        self.assertEqual(len(self.middle.graphs), 3)
        self.assertEqual(len(b._capture_descs[Mode.FULL]), 3)
        self.assertIsNone(self.short._capture_mem_samples)
        self.assertIsNone(self.middle._capture_mem_samples)
        self.assertIsNone(self.middle._max_full_descs_to_capture)
        b._capture_mem_samples = None
        self.assertEqual(len(b._capture_descs[Mode.FULL]), 9)
        self.assertEqual(b.captured_token_counts(), [6, 12])

    def test_release_clears_all_and_renews_all_private_pools(self):
        b = self.banks()
        b.capture("model")
        before = [self.short.pool, self.middle.pool]
        self.select(b, [4352])
        b.release_graphs()
        self.assertFalse(any((m.graphs for m in (self.wide, self.short, self.middle))))
        self.assertNotIn(self.short.pool, before)
        self.assertNotIn(self.middle.pool, before)
        with self.assertRaises(RuntimeError):
            b.run_fullgraph(NS(cg_mode=Mode.FULL, key=12))
        b.capture("fresh")
        self.assertEqual(self.select(b, [4352]), "middle-owned-output")

    def test_partial_middle_capture_poison_and_restore(self):
        b = self.banks()
        self.middle.fail_capture = True
        with self.assertRaisesRegex(RuntimeError, "capture witness"):
            b.capture("bad")
        self.assertTrue(
            all((o._hcu_fullgraph_scoring_bound is None for o in self.owners))
        )
        for call in (b.dispatch, b.release_graphs, b.needs_capture):
            with self.assertRaises(RuntimeError):
                call()

    def test_release_failure_poison(self):
        b = self.banks()
        b.capture("model")
        self.middle.fail_release = True
        with self.assertRaisesRegex(RuntimeError, "release witness"):
            b.release_graphs()
        with self.assertRaises(RuntimeError):
            b.dispatch()

    def test_constructor_retains_short_guard_and_rejects_bad_extra(self):
        with self.assertRaises(ValueError):
            self.api.FlashFullGraphBanks(
                self.wide,
                self.short,
                self.owners,
                16384,
                Mode.FULL,
                short_pool_factory=object,
            )
        for entries in [
            ((2048, self.middle),),
            ((1048576, self.middle),),
            ((True, self.middle),),
            ((16384, self.wide),),
            ((32768, self.middle), (16384, Manager("m2", self.owners))),
        ]:
            with self.subTest(entries=entries):
                with self.assertRaises(ValueError):
                    self.banks(scoring_banks=entries)

    def test_mismatched_descriptors_rejected(self):
        self.middle._capture_descs[Mode.FULL] = [6]
        with self.assertRaises(ValueError):
            self.banks()

    def test_scoring_banks_reject_mixed_model_limits(self):
        self.owners[1].max_model_len = 32768
        with self.assertRaises(ValueError):
            self.banks()

    def test_wide_short_must_be_different_managers(self):
        with self.assertRaises(ValueError):
            self.api.FlashFullGraphBanks(
                self.wide,
                self.wide,
                self.owners,
                2048,
                Mode.FULL,
                short_pool_factory=object,
            )

    def test_core_missing_selected_graph_fails_instead_of_wide_fallback(self):
        b = self.banks()
        b.capture("model")
        self.middle.graphs.clear()
        with self.assertRaisesRegex(AssertionError, "missing real Core graph"):
            self.select(b, [4352])

    def test_independent_contexts_restore_on_partial_owner_error(self):
        self.owners[1]._hcu_fullgraph_capture_bound = 2048
        with self.assertRaises(RuntimeError):
            with self.api._capture_scoring_bound(self.owners, 16384):
                pass
        self.assertIsNone(self.owners[0]._hcu_fullgraph_scoring_bound)
        self.assertEqual(self.owners[1]._hcu_fullgraph_capture_bound, 2048)

    def test_contexts_are_mutually_exclusive(self):
        with self.api._capture_scoring_bound(self.owners, 16384):
            with self.assertRaises(RuntimeError):
                with self.api._capture_bound(self.owners, 2048):
                    pass
            for o in self.owners:
                self.assertEqual(self.api.scoring_max_seq_len(o, 53), 16384)
        with self.api._capture_bound(self.owners, 2048):
            with self.assertRaises(RuntimeError):
                with self.api._capture_scoring_bound(self.owners, 16384):
                    pass

    def test_scoring_capacity_retains_default_and_rejects_corruption(self):
        o = self.owners[0]
        self.assertEqual(self.api.scoring_max_seq_len(o, 53), 53)
        for invalid in (0, -1, True, 1048577):
            o._hcu_fullgraph_scoring_bound = invalid
            with self.assertRaises((ValueError, RuntimeError)):
                self.api.scoring_max_seq_len(o, 53)

    def test_production_forward_middle_never_unpruned_with_metadata53(self):
        tree = ast.parse((SOURCE / "compressed_indexer.py").read_text())
        f = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "forward"
            )
        )
        branch = next(
            (
                n
                for n in f.body
                if isinstance(n, ast.If)
                and any(
                    (
                        isinstance(c, ast.Call)
                        and isinstance(c.func, ast.Name)
                        and (c.func.id == "unpruned_indices")
                        for c in ast.walk(n)
                    )
                )
            )
        )
        expr = compile(ast.Expression(branch.test), "<actual-unpruned-guard>", "eval")
        for score, maximum, expected in [
            (16384, 53, False),
            (None, 2048, True),
            (None, 2049, False),
        ]:
            o = owner(_hcu_fullgraph_scoring_bound=score)
            self.assertEqual(eval(expr, {"self": o, "max_seq_len": maximum}), expected)

    def test_forward_uses_capacity_only_after_original_unpruned_return(self):
        tree = ast.parse((SOURCE / "compressed_indexer.py").read_text())
        f = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "forward"
            )
        )
        calls = [
            n
            for n in ast.walk(f)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and (n.func.id == "scoring_max_seq_len")
        ]
        self.assertEqual(len(calls), 1)
        unpruned = next(
            (
                n
                for n in ast.walk(f)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and (n.func.id == "unpruned_indices")
            )
        )
        self.assertGreater(calls[0].lineno, unpruned.lineno)

    def test_capture_none_metadata53_executes_actual_scalar_capacity_assignments(self):
        tree = ast.parse((SOURCE / "compressed_indexer.py").read_text())
        f = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "forward"
            )
        )

        def assignment(name):
            return next(
                (
                    n
                    for n in f.body
                    if isinstance(n, ast.Assign)
                    and any(
                        (isinstance(t, ast.Name) and t.id == name for t in n.targets)
                    )
                )
            )

        branch = next(
            (
                n
                for n in f.body
                if isinstance(n, ast.If)
                and any(
                    (
                        isinstance(c, ast.Call)
                        and isinstance(c.func, ast.Name)
                        and (c.func.id == "unpruned_indices")
                        for c in ast.walk(n)
                    )
                )
            )
        )
        for mode, score, short, expected_max, unpruned, width in [
            (Mode.NONE, 16384, None, 53, False, 4096),
            (Mode.FULL, 16384, None, 1048576, False, 4096),
            (Mode.NONE, None, 2048, 2048, True, None),
            (Mode.NONE, None, None, 53, True, None),
            (Mode.FULL, None, None, 1048576, False, 262144),
        ]:
            ns = dict(
                self=owner(
                    _hcu_fullgraph_scoring_bound=score,
                    _hcu_fullgraph_capture_bound=short,
                ),
                km=NS(max_seq_len=53),
                get_forward_context=lambda: NS(cudagraph_runtime_mode=mode),
                CUDAGraphMode=Mode,
                POOL_PAGE_SIZE=64,
                cdiv=lambda a, b: -(-a // b),
                scoring_max_seq_len=self.api.scoring_max_seq_len,
            )

            def execute(node):
                exec(
                    compile(
                        ast.fix_missing_locations(
                            ast.Module(body=[node], type_ignores=[])
                        ),
                        "<actual-forward-scalars>",
                        "exec",
                    ),
                    ns,
                )

            execute(assignment("max_seq_len"))
            self.assertEqual(ns["max_seq_len"], expected_max)
            self.assertEqual(
                eval(
                    compile(
                        ast.Expression(branch.test), "<actual-forward-guard>", "eval"
                    ),
                    ns,
                ),
                unpruned,
            )
            if not unpruned:
                execute(assignment("score_max_seq_len"))
                execute(assignment("width"))
                self.assertEqual(ns["width"], width)


if __name__ == "__main__":
    unittest.main()
