# SPDX-License-Identifier: Apache-2.0
"""Static contract comparisons must surface drift without claiming equivalence."""
import ast
import sys
import unittest

sys.dont_write_bytecode = True
from upgrade_preflight import compare_contracts, constants, signature_checks


def compare(old, new, symbol="route"):
    return compare_contracts(ast.parse(old), ast.parse(new), symbol)


class ContractTests(unittest.TestCase):
    def test_formatted_fstring_targets_require_manual_review(self):
        for expression in ['f"vllm.{NAME!s}"', 'f"vllm.{NAME!r}"',
                           'f"vllm.{NAME!a}"', 'f"vllm.{NAME:.6}"',
                           'f"vllm.{NAME:>{WIDTH}}"', 'f"vllm.{NAME:}"']:
            with self.subTest(expression=expression):
                values = constants(ast.parse(
                    'NAME = "routers"\nWIDTH = 8\nTARGET_MODULE = ' + expression))
                self.assertNotIn("TARGET_MODULE", values)

    def test_unformatted_fstring_targets_remain_resolvable(self):
        values = constants(ast.parse(
            'NAME = "router"\nTARGET_MODULE = f"vllm.{NAME}"'))
        self.assertEqual(values["TARGET_MODULE"], "vllm.router")

    @unittest.skipIf(sys.version_info < (3, 12), "PEP 695 requires Python 3.12")
    def test_generic_type_parameters_are_callable_contracts(self):
        for old, new in [
            ("T: int", "T: str"),
            ("T: (int, str)", "T: (int, bytes)"),
            ("*Ts", "**Ts"),
        ]:
            with self.subTest(old=old, new=new):
                result = compare(f"def route[{old}](): return 1",
                                 f"def route[{new}](): return 1")
                self.assertEqual(result["status"], "changed_review_required")
                self.assertEqual(result["changed_fields"], ["type_params_ast"])
                same = compare(f"def route[{old}](): return 1",
                               f"def route[{old}]():\n    # comment\n    return 1\n")
                self.assertEqual(same["status"], "unchanged_static_only")

    def test_unevaluable_reassignments_invalidate_old_constants(self):
        for assignment in ["TARGET_MODULE = choose_module()",
                           "TARGET_MODULE: str = choose_module()",
                           "TARGET_MODULE = MISSING",
                           "TARGET_MODULE += suffix",
                           "TARGET_MODULE, other = choose_pair()"]:
            with self.subTest(assignment=assignment):
                values = constants(ast.parse('TARGET_MODULE = "vllm.router"\n' + assignment))
                self.assertNotIn("TARGET_MODULE", values)

    def test_annotation_only_preserves_binding_and_literal_reassignment_updates(self):
        values = constants(ast.parse('''TARGET_MODULE = "vllm.router"
TARGET_MODULE: str
COPY = TARGET_MODULE
TARGET_MODULE = "vllm.fast"
'''))
        self.assertEqual(values["COPY"], "vllm.router")
        self.assertEqual(values["TARGET_MODULE"], "vllm.fast")

    def test_zero_argument_guard_uses_default_empty_positional(self):
        guard = ast.parse('require_exact_signature(fn, "vllm.ops.ready")')
        target = ast.parse('def ready(): return 1')
        checks = signature_checks(guard, {}, target, "vllm.ops")
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["expected_names"], [])
        self.assertEqual(checks[0]["status"], "names_match_only")

    def test_variadic_guard_names_follow_signature_order(self):
        guard = ast.parse('''require_exact_signature(
            fn, "vllm.ops.route", positional=("self",),
            var_positional="args", keyword_only=("enabled",),
            var_keyword="kwargs")''')
        target = ast.parse('def route(self, *args, enabled=True, **kwargs): pass')
        checks = signature_checks(guard, {}, target, "vllm.ops")
        self.assertEqual(checks[0]["expected_names"],
                         ["self", "args", "enabled", "kwargs"])
        self.assertEqual(checks[0]["status"], "names_match_only")

    def test_same_names_different_default(self):
        result = compare("def route(x, pad=False): return x",
                         "def route(x, pad=True): return x")
        self.assertIn("parameters", result["changed_fields"])
        self.assertFalse(result["baseline"]["parameters"][1]["default_ast"] ==
                         result["target"]["parameters"][1]["default_ast"])
        self.assertEqual(result["status"], "changed_review_required")

    def test_parameter_kind_and_requiredness(self):
        result = compare("def route(x, pad=False): return x",
                         "def route(x, *, pad): return x")
        parameter = result["target"]["parameters"][1]
        self.assertEqual(parameter["kind"], "keyword_only")
        self.assertFalse(parameter["has_default"])
        self.assertIn("parameters", result["changed_fields"])

    def test_body_change_even_when_signature_matches(self):
        result = compare("def route(x): return x + 1", "def route(x): return x * 2")
        self.assertEqual(result["changed_fields"], ["body_sha256"])

    def test_decorator_annotation_and_async_changes(self):
        result = compare("def route(x: int) -> int: return x",
                         "@wrapper\nasync def route(x: float) -> float: return x")
        self.assertEqual(set(result["changed_fields"]),
                         {"parameters", "returns_ast", "decorators_ast", "is_async"})

    def test_missing_or_ambiguous_symbol_is_unresolved(self):
        for target in ("def other(x): return x",
                       "def route(x): return x\ndef route(x): return x + 1"):
            with self.subTest(target=target):
                result = compare("def route(x): return x", target)
                self.assertEqual(result["status"], "symbol_unresolved")

    def test_formatting_and_docstring_are_not_semantic_acceptance(self):
        result = compare("def route(x): return x",
                         'def route(x):\n    "Doc only"\n    # comment\n    return x\n')
        self.assertEqual(result["status"], "unchanged_static_only")
        self.assertEqual(result["changed_fields"], [])


if __name__ == "__main__":
    unittest.main()
