# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _get_supported_kv_cache_dtypes(relative_path: str) -> list[str]:
    source = (ROOT / relative_path).read_text()
    module = ast.parse(source)

    for node in ast.walk(module):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "supported_kv_cache_dtypes":
                    return [ast.literal_eval(elt) for elt in node.value.elts]
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "supported_kv_cache_dtypes"
            ):
                return [ast.literal_eval(elt) for elt in node.value.elts]

    raise AssertionError(f"supported_kv_cache_dtypes not found in {relative_path}")


def test_hcu_mla_backends_declare_fp8_e5m2_support() -> None:
    flashmla_dtypes = _get_supported_kv_cache_dtypes(
        "vllm_hcu/v1/attention/backends/mla/flashmla.py"
    )
    triton_mla_dtypes = _get_supported_kv_cache_dtypes(
        "vllm_hcu/v1/attention/backends/mla/triton_mla.py"
    )

    assert "fp8_e5m2" in flashmla_dtypes
    assert "fp8_e5m2" in triton_mla_dtypes
