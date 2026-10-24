# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Static reachability guards for the phase-2 V4.1 integration boundary.

These tests intentionally read source rather than importing a model or native
extension.  They prove whether the real V4.1 entrypoint can reach the HCU
adapter under the phase-2 whitelist.
"""

from pathlib import Path
import ast

from tests.fixtures.vllm_source import resolve_target_vllm_root


ROOT = Path(__file__).resolve().parents[2]
HCU_V4_MODEL = ROOT / "vllm_hcu/models/deepseek_v4.py"
HCU_V4_OPS = ROOT / "vllm_hcu/v1/attention/ops/deepseek_v4_ops/__init__.py"
HCU_REGISTRY = ROOT / "vllm_hcu/models/__init__.py"
MODULE_EXCHANGE = ROOT / "vllm_hcu/patch/module_exchange.py"
EXTERNAL_V41_ATTENTION = (
    resolve_target_vllm_root() / "vllm" / "models" / "deepseek_v41" / "attention.py"
)


def _source(path: Path) -> str:
    assert path.is_file(), f"expected source evidence is missing: {path}"
    return path.read_text()


def test_real_v41_entrypoint_has_ratio1_shared_k_contract_outside_hcu_class():
    source = _source(EXTERNAL_V41_ATTENTION)
    assert "class DeepseekV4Attention" in source
    assert "class DeepseekV4Indexer" in source
    assert "compress_ratio not in (0, 1, 2)" in source
    assert "owns_k" in source
    assert "index_k_cache" in source
    assert "indexer_k_norm_rope_store" in source
    assert "fused_indexer_q_rope_quant" in source


def test_hcu_class_is_v4_legacy_and_not_a_v41_replacement():
    source = _source(HCU_V4_MODEL)
    assert "class DeepseekV4Attention" in source
    assert "if self.compress_ratio == 4:" in source
    assert "compress_ratio=self.compress_ratio" in source
    # The HCU class has the V4 local compressor/indexer construction, not V4.1
    # owns_k/shared-source construction.
    assert "owns_k" not in source
    assert "index_source_layer_ids" not in source


def test_hcu_q_import_is_external_v4_ops_reexport_not_phase2_adapter():
    tree = ast.parse(_source(HCU_V4_OPS))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm.v1.attention.ops.deepseek_v4_ops"
        for alias in node.names
    }
    assert "fused_indexer_q_rope_quant" in imported
    assert "fused_indexer_q_rope_quant" not in {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_hcu_registry_and_exchange_have_no_v41_attention_replacement():
    registry = _source(HCU_REGISTRY)
    exchange = _source(MODULE_EXCHANGE)
    assert '"DeepseekV41ForCausalLM"' not in registry
    assert "vllm.models.deepseek_v41.attention" not in exchange
    assert "vllm_hcu.model_executor.layers.deepseek_v4_attention" not in exchange


def test_phase2_blocker_is_not_satisfied_by_removing_fp4_guard():
    source = _source(ROOT / "vllm_hcu/model_executor/layers/sparse_attn_indexer.py")
    assert "Fp4QTuple" in source
    assert "rocm_aiter_sparse_attn_indexer_native" in source
    # FP4 now has a distinct reader entrypoint; it must not silently use the
    # old FP8-only route when the tuple ABI is selected.
    reader = _source(ROOT / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py")
    assert "_hcu_fp4_sparse_attn_indexer_native" in reader
    assert "lightop_fp4_decode" in reader
