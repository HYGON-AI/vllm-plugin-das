# SPDX-License-Identifier: Apache-2.0
"""Static contract comparisons must surface drift without claiming equivalence."""
import ast
import sys
import unittest

sys.dont_write_bytecode = True
from upgrade_preflight import compare_contracts, signature_checks


def compare(old, new, symbol="route"):
    return compare_contracts(ast.parse(old), ast.parse(new), symbol)


class ContractTests(unittest.TestCase):
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
