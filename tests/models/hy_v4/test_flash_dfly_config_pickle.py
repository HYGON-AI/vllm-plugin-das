# SPDX-License-Identifier: Apache-2.0
"""Stdlib source/helper regressions; not a full vLLM engine qualification."""
import ast
from copy import deepcopy
from functools import partial
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import pickle
import sys
from types import ModuleType
import unittest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4_flash/dfly_config.py"


class DFlyConfigPickleTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(SOURCE.read_text(encoding="utf8"))
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_copy_derived_dfly_config")
        # Only execute the actual stdlib helper AST, not a mocked ModelConfig.
        self.module = ModuleType("_hy4_dfly_pickle_source_helper")
        self.module.deepcopy = deepcopy
        sys.modules[self.module.__name__] = self.module
        self.addCleanup(sys.modules.pop, self.module.__name__, None)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"),
             self.module.__dict__)

    def test_config_uses_module_partial_instead_of_local_lambda(self):
        overrides = [kw.value for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                     for kw in n.keywords if kw.arg == "hf_overrides"]
        self.assertEqual(len(overrides), 1)
        self.assertEqual(ast.unparse(overrides[0]),
                         "partial(_copy_derived_dfly_config, derived)")

    def test_actual_helper_partial_roundtrips_with_spawn_pickler(self):
        derived = {"architectures": ["HYV4DFlyModel"], "nested": {"layers": [1]}}
        override = partial(self.module._copy_derived_dfly_config, derived)
        restored = pickle.loads(ForkingPickler.dumps(override))
        self.assertEqual(restored({"irrelevant": "target"}), derived)

    def test_override_returns_independent_deepcopies(self):
        derived = {"nested": {"layers": [1]}}
        override = partial(self.module._copy_derived_dfly_config, derived)
        first, second = override(None), override(None)
        first["nested"]["layers"].append(2)
        self.assertEqual(second, derived)
        self.assertIsNot(second["nested"], derived["nested"])

    def test_previous_local_lambda_reproduces_pickle_failure(self):
        derived = {"layers": [1]}
        previous = lambda _: deepcopy(derived)
        with self.assertRaises((AttributeError, pickle.PicklingError)):
            ForkingPickler.dumps(previous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
