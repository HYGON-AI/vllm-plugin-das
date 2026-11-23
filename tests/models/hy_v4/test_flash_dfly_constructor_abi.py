# SPDX-License-Identifier: Apache-2.0
"""Portable legacy constructor source/ABI regression, not engine qualification.

Legacy HYV4DFlyModel is currently undecorated: the positional call is not an
independently reproduced legacy crash. Keywords preserve that ABI and prepare
for the installed compile port. Optional Core tests execute its unmodified
constructor-wrapper AST with explicit config/mode doubles, not the complete
decorator or a real Core VllmConfig. No Torch/plugin imports are needed.
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_hcu/models/hy_v4_flash/nvidia/dfly.py"
CORE_DECORATORS = ROOT.parent / "vllm-core-033/vllm/compilation/decorators.py"


class DFlyConstructorABITests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
        owner = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "HYV4DFlyForCausalLM"
        )
        init = next(
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        calls = [
            n
            for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "HYV4DFlyModel"
        ]
        self.assertEqual(len(calls), 1)
        self.call = calls[0]

    def invoke_source_call(self, constructor, config, prefix, prefix_helper):
        return eval(
            compile(ast.Expression(self.call), str(SOURCE), "eval"),
            {
                "HYV4DFlyModel": constructor,
                "vllm_config": config,
                "prefix": prefix,
                "maybe_prefix": prefix_helper,
            },
        )

    def test_source_call_uses_explicit_config_and_prefix_keywords(self):
        self.assertEqual(self.call.args, [])
        keywords = {kw.arg: ast.unparse(kw.value) for kw in self.call.keywords}
        self.assertEqual(
            keywords,
            {
                "vllm_config": "vllm_config",
                "prefix": "maybe_prefix(prefix, 'model')",
            },
        )

    def test_actual_call_expression_preserves_keyword_only_abi_and_identity(self):
        config, prefix_result = object(), object()
        for prefix in ("", "draft"):
            with self.subTest(prefix=prefix):
                calls = []

                def keyword_only(*, vllm_config, prefix):
                    calls.append((vllm_config, prefix))
                    return prefix_result

                def prefix_helper(received, name):
                    self.assertEqual((received, name), (prefix, "model"))
                    return prefix_result

                result = self.invoke_source_call(
                    keyword_only, config, prefix, prefix_helper
                )
                self.assertIs(result, prefix_result)
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0][0], config)
                self.assertIs(calls[0][1], prefix_result)

    def test_keyword_call_remains_compatible_with_plain_legacy_signature(self):
        class PlainLegacy:
            def __init__(self, vllm_config, prefix):
                self.config, self.prefix = vllm_config, prefix

        config = object()
        result = self.invoke_source_call(
            PlainLegacy, config, "draft", lambda p, n: p + "." + n
        )
        self.assertIs(result.config, config)
        self.assertEqual(result.prefix, "draft.model")

    def test_old_positional_form_is_rejected_by_keyword_only_contract(self):
        calls = []

        def keyword_only(*, vllm_config, prefix):
            calls.append((vllm_config, prefix))

        with self.assertRaises(TypeError):
            keyword_only(object(), "model")
        self.assertEqual(calls, [])

    def core_wrapper_receiver(self):
        if not CORE_DECORATORS.is_file():
            self.skipTest("optional pinned Core source tree unavailable")
        tree = ast.parse(CORE_DECORATORS.read_text(encoding="utf-8-sig"))
        decorator = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_support_torch_compile"
        )
        wrapper = next(
            n
            for n in decorator.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )

        class ConfigABIDouble:
            compilation_config = SimpleNamespace(mode="NONE")

        config = ConfigABIDouble()
        calls = []

        def old_init(self, vllm_config: ConfigABIDouble, prefix: str):
            calls.append((self, vllm_config, prefix))

        namespace = {
            "_T": object,
            "VllmConfig": ConfigABIDouble,
            "Any": Any,
            "inspect": inspect,
            "old_init": old_init,
            "get_current_vllm_config": lambda: config,
            "CompilationMode": SimpleNamespace(
                NONE="NONE", STOCK_TORCH_COMPILE="STOCK"
            ),
            "use_aot_compile": True,
            "enable_if": None,
            "_should_ignore_torch_compile": lambda cls: False,
        }
        # The exact nested wrapper body is compiled, not an imitation of its
        # positional-annotation loop. NONE exits before any compiler branch.
        exec(
            compile(
                ast.Module(body=[wrapper], type_ignores=[]),
                str(CORE_DECORATORS),
                "exec",
            ),
            namespace,
        )
        receiver = type("CoreWrapperReceiver", (), {"__init__": namespace["__init__"]})
        return receiver, config, calls

    def test_actual_core_wrapper_source_accepts_production_keyword_call(self):
        receiver, config, calls = self.core_wrapper_receiver()
        result = self.invoke_source_call(
            receiver, config, "draft", lambda p, n: p + "." + n
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], result)
        self.assertIs(calls[0][1], config)
        self.assertEqual(calls[0][2], "draft.model")
        self.assertTrue(result.do_not_compile)

    def test_actual_core_wrapper_source_rejects_old_str_positional_argument(self):
        receiver, config, calls = self.core_wrapper_receiver()
        with self.assertRaisesRegex(TypeError, "positional argument of type 'str'"):
            receiver(config, "draft.model")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
