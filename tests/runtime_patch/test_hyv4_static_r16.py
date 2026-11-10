# SPDX-License-Identifier: Apache-2.0
"""Portable checks of the two real R16 support predicates, without GPU imports."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PATHS = (
    "vllm_hcu/model_executor/layers/quantization/compressed_tensors_w4a8_moe.py",
    "vllm_hcu/patch/platform/core_fix/patch_vllm_config.py",
)


def predicate(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == "_hcu_static_eplb"
            or isinstance(t, ast.Name) and t.id == "static_pcp"
            for t in node.targets
        ):
            return compile(ast.Expression(node.value), path, "eval")
    raise AssertionError("Static EPLB predicate missing: " + path)


def fixture(redundant):
    return dict(
        parallel=NS(
            enable_eplb=True, enable_expert_parallel=True,
            _vllm_hcu_expert_map_path="map.json",
            eplb_config=NS(num_redundant_experts=redundant),
            pipeline_parallel_size=1, tensor_parallel_size=1,
            prefill_context_parallel_size=16, data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        moe_parallel=NS(all2all_backend="deepep_high_throughput", pcp_size=16,
                        ep_size=16, tp_size=1, dp_size=1, use_ep=True),
        moe=NS(moe_backend="deep_gemm"),
        feature_cfg=NS(expert_map_path="map.json", expert_map_record_path=None,
                       eplb_disable_rearrange=False),
        target=NS(quantization="compressed-tensors"),
    )


class StaticR16PredicateTests(unittest.TestCase):
    def test_flag_and_redundancy_matrix(self):
        for path in PATHS:
            expr = predicate(path)
            for flag in (None, "0", "1", "true", " 1"):
                with patch.dict(os.environ):
                    if flag is None:
                        os.environ.pop("HY4_EXPERIMENTAL_STATIC_R16", None)
                    else:
                        os.environ["HY4_EXPERIMENTAL_STATIC_R16"] = flag
                    for redundant in (None, -1, 0, 1, 15, 16, 32, 48):
                        with self.subTest(path=path, flag=flag, redundant=redundant):
                            self.assertEqual(
                                bool(eval(expr, {}, fixture(redundant))),
                                redundant == 0 or redundant == 16 and flag == "1",
                            )

    def test_quantization_rejects_other_routes_and_topologies(self):
        controls = (
            ("parallel", "enable_eplb", False),
            ("parallel", "enable_expert_parallel", False),
            ("parallel", "_vllm_hcu_expert_map_path", None),
            ("parallel", "tensor_parallel_size", 2),
            ("moe_parallel", "all2all_backend", "deepep_low_latency"),
            ("moe_parallel", "pcp_size", 8),
            ("moe_parallel", "ep_size", 8),
            ("moe_parallel", "tp_size", 2),
            ("moe_parallel", "dp_size", 2),
            ("moe_parallel", "use_ep", False),
            ("moe", "moe_backend", "triton"),
        )
        with patch.dict(os.environ, HY4_EXPERIMENTAL_STATIC_R16="1"):
            expr = predicate(PATHS[0])
            for group, name, value in controls:
                env = fixture(16)
                setattr(env[group], name, value)
                with self.subTest(group=group, name=name, value=value):
                    self.assertFalse(eval(expr, {}, env))

    def test_config_rejects_other_parallel_and_mapping_contracts(self):
        controls = (
            ("parallel", "enable_eplb", False),
            ("parallel", "pipeline_parallel_size", 2),
            ("parallel", "tensor_parallel_size", 2),
            ("parallel", "prefill_context_parallel_size", 8),
            ("parallel", "data_parallel_size", 2),
            ("parallel", "decode_context_parallel_size", 2),
            ("feature_cfg", "expert_map_path", None),
            ("feature_cfg", "expert_map_record_path", "record.json"),
            ("feature_cfg", "eplb_disable_rearrange", True),
            ("target", "quantization", None),
        )
        with patch.dict(os.environ, HY4_EXPERIMENTAL_STATIC_R16="1"):
            expr = predicate(PATHS[1])
            for group, name, value in controls:
                env = fixture(16)
                setattr(env[group], name, value)
                with self.subTest(group=group, name=name, value=value):
                    self.assertFalse(eval(expr, {}, env))


if __name__ == "__main__":
    unittest.main()
