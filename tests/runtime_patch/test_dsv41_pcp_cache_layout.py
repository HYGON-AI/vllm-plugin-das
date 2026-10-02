# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for DeepSeek-V4.1 PCP cache materialization.

These tests pin the row layouts the V4.1 cache writers depend on: the manager's
expanded slot mapping, the global token order used for compressor pooling, and
the mapping between them.  They run without a PCP process group by exercising
the single-rank identity paths and the local row arithmetic directly.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.attention import pcp


class _Layout:
    """Minimal stand-in for the manager metadata the helpers read."""

    def __init__(
        self,
        *,
        world_size: int,
        local_num_tokens: int,
        global_num_tokens: int,
        restore_idx: torch.Tensor,
        has_global_prefill: bool = True,
    ) -> None:
        self.pcp_world_size = world_size
        self.pcp_has_global_prefill = has_global_prefill
        self.pcp_local_num_tokens = local_num_tokens
        self.pcp_global_num_tokens = global_num_tokens
        self.pcp_restore_idx = restore_idx


def test_restore_pcp_rows_to_global_is_identity_without_pcp() -> None:
    """A non-PCP step must not touch the tensor at all."""

    layout = _Layout(
        world_size=1,
        local_num_tokens=3,
        global_num_tokens=3,
        restore_idx=torch.arange(3, dtype=torch.int64),
    )
    tensor = torch.arange(12, dtype=torch.float32).view(3, 4)

    restored = pcp.restore_pcp_rows_to_global(tensor, layout)

    assert restored is tensor


def test_globalize_pcp_slot_mapping_is_identity_without_pcp() -> None:
    """PCP=1 keeps the builder's mapping untouched."""

    layout = _Layout(
        world_size=1,
        local_num_tokens=2,
        global_num_tokens=2,
        restore_idx=torch.arange(2, dtype=torch.int64),
    )
    slots = torch.tensor([5, 7], dtype=torch.int64)

    assert pcp.globalize_pcp_slot_mapping(slots, layout) is slots


def test_globalize_requires_a_partitioned_prefill() -> None:
    """Global slot restoration is meaningless on a replicated decode step."""

    layout = _Layout(
        world_size=2,
        local_num_tokens=2,
        global_num_tokens=2,
        restore_idx=torch.arange(2, dtype=torch.int64),
        has_global_prefill=False,
    )

    with pytest.raises(AssertionError, match="prefill"):
        pcp.globalize_pcp_slot_mapping(
            torch.tensor([1, 2], dtype=torch.int64), layout
        )


def test_restore_rejects_a_tensor_wider_than_the_local_rows() -> None:
    """A rank-local tensor must never exceed its padded row count."""

    layout = _Layout(
        world_size=2,
        local_num_tokens=2,
        global_num_tokens=4,
        restore_idx=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    )
    tensor = torch.zeros((3, 4), dtype=torch.float32)

    with pytest.raises(AssertionError, match="exceeds"):
        pcp.restore_pcp_rows_to_global(tensor, layout)


def test_restore_rejects_a_restore_map_of_the_wrong_length() -> None:
    """The restore map has to describe exactly the global token count."""

    layout = _Layout(
        world_size=2,
        local_num_tokens=2,
        global_num_tokens=5,
        restore_idx=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    )

    with pytest.raises(AssertionError, match="global token count"):
        pcp.restore_pcp_rows_to_global(
            torch.zeros((2, 3), dtype=torch.float32), layout
        )


def test_pcp_global_accessors_require_the_manager_layout() -> None:
    """Global accessors must fail closed when the layout is incomplete."""

    layout = _Layout(
        world_size=2,
        local_num_tokens=2,
        global_num_tokens=3,
        restore_idx=torch.arange(3, dtype=torch.int64),
    )

    with pytest.raises(AssertionError, match="positions"):
        pcp.pcp_global_positions(layout)
    with pytest.raises(AssertionError, match="query offsets"):
        pcp.pcp_global_query_start_loc(layout)
    with pytest.raises(AssertionError, match="token-to-request"):
        pcp.pcp_global_token_to_req_indices(layout)


def test_pcp_cache_writers_reuse_upstream_with_global_request_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pooling sees contiguous requests; all writers see global slots transiently."""

    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_pcp as adapter

    layout = SimpleNamespace(
        pcp_world_size=2,
        pcp_has_global_prefill=True,
        pcp_local_num_tokens=2,
        pcp_global_num_tokens=4,
        pcp_restore_idx=torch.tensor([0, 2, 1, 3]),
        pcp_global_positions=torch.tensor([0, 1, 2, 3]),
        pcp_global_query_start_loc=torch.tensor([0, 4]),
        pcp_global_token_to_req_indices=torch.tensor([0, 0, 0, 0]),
    )
    local_slots = torch.tensor([10, 30, 20, 40])
    metadata = {
        name: SimpleNamespace(
            **vars(layout),
            slot_mapping=local_slots.clone(),
            block_size=16,
            query_start_loc=torch.tensor([0, 2]),
            token_to_req_indices=torch.tensor([0, 0]),
        )
        for name in ("state", "kv", "indexer", "swa")
    }

    class Group:
        world_size = 2
        rank_in_group = 0

        def all_gather(self, tensor, dim=0):
            assert dim == 0
            torch.testing.assert_close(tensor[:, 0], torch.tensor([2.0, 4.0]))
            return torch.tensor([[2.0], [4.0], [3.0], [5.0]])

    monkeypatch.setattr(pcp, "get_pcp_group", Group)
    monkeypatch.setattr(adapter, "_step_metadata", lambda: metadata)
    monkeypatch.setenv("VLLM_HCU_DSV4_PCP_EXPERIMENTAL", "1")
    observed = []

    def check_slots(name):
        torch.testing.assert_close(
            metadata[name].slot_mapping, torch.tensor([10, 20, 30, 40])
        )
        observed.append(name)

    class Compressor:
        def forward(self, kv_score, positions):
            name = "state" if self.state_cache is not None else "kv"
            check_slots(name)
            torch.testing.assert_close(
                kv_score[:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0])
            )
            torch.testing.assert_close(positions, layout.pcp_global_positions)
            if self.state_cache is not None:
                torch.testing.assert_close(
                    metadata["state"].query_start_loc,
                    layout.pcp_global_query_start_loc,
                )
                torch.testing.assert_close(
                    metadata["state"].token_to_req_indices,
                    layout.pcp_global_token_to_req_indices,
                )
            return kv_score

        def insert_cache(self, latent, positions, rotary_emb):
            check_slots("kv")
            torch.testing.assert_close(positions, layout.pcp_global_positions)
            torch.testing.assert_close(
                latent[:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0])
            )

    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return None

    class Indexer:
        def _produce_k(self, latent, positions, rotary_emb):
            check_slots("indexer")
            torch.testing.assert_close(positions, layout.pcp_global_positions)

    class Attention:
        def _prepare_and_attn(
            self,
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_weights,
            positions,
            attn_out,
        ):
            return None

        def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
            assert attn_metadata["swa"].slot_mapping.numel() == 0
            return q

    compressor_module = ModuleType(adapter.COMPRESSOR_MODULE)
    compressor_module.DeepseekCompressor = Compressor
    compressor_module.CompressorMetadataBuilder = Builder
    attention_module = ModuleType(adapter.TARGET_MODULE)
    attention_module.DeepseekV4Attention = Attention
    attention_module.DeepseekV4Indexer = Indexer
    monkeypatch.setattr(adapter.importlib, "import_module", lambda _: compressor_module)
    assert adapter.apply_to_module(attention_module)

    compressor = Compressor()
    compressor.state_cache = SimpleNamespace(prefix="state")
    compressor.k_cache_prefix = "kv"
    score = torch.tensor([[2.], [4.]])
    positions = torch.tensor([0, 2])
    latent = compressor.forward(score, positions)
    compressor.insert_cache(latent, positions, None)
    compressor.state_cache = None
    compressor.forward(score, positions)
    indexer = Indexer()
    indexer.k_cache = SimpleNamespace(prefix="indexer")
    indexer._produce_k(latent, positions, None)

    def insert_global_kv(kv, cache, slots, global_positions, *args):
        torch.testing.assert_close(kv[:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
        torch.testing.assert_close(slots, torch.tensor([10, 20, 30, 40]))
        torch.testing.assert_close(global_positions, layout.pcp_global_positions)
        observed.append("swa")

    monkeypatch.setattr(
        torch.ops._C, "fused_deepseek_v4_kv_rope_insert", insert_global_kv,
        raising=False,
    )
    attention = Attention()
    attention.swa_cache_layer = SimpleNamespace(
        prefix="swa", kv_cache=torch.empty(1, dtype=torch.uint8)
    )
    attention.rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(0))
    attention.kv_mxfp8 = False
    q = torch.ones(2, 1, 1)
    assert attention._fused_qnorm_rope_kv_insert(q, score, positions, metadata) is q
    assert observed == ["state", "kv", "kv", "indexer", "swa"]
    for item in metadata.values():
        torch.testing.assert_close(item.slot_mapping, local_slots)
    torch.testing.assert_close(metadata["state"].query_start_loc, torch.tensor([0, 2]))
    torch.testing.assert_close(
        metadata["state"].token_to_req_indices, torch.tensor([0, 0])
    )
    with pytest.raises(RuntimeError, match="writer failed"):
        with adapter._global_cache_metadata(metadata["state"], layout):
            raise RuntimeError("writer failed")
    torch.testing.assert_close(metadata["state"].slot_mapping, local_slots)


def test_pcp_adapter_helpers_are_not_self_recursive() -> None:
    """A helper that returns its own name would recurse until the stack dies.

    An automated edit once rewrote ``_globalize_slots``' body into a call to
    itself, which only surfaced inside the model warmup on a real run.  This
    guards the adapter's helper bodies statically.
    """

    import ast
    from pathlib import Path

    import vllm_hcu.patch.worker.core_fix.patch_deepseek_v41_pcp as adapter

    source = Path(adapter.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Return):
                continue
            value = statement.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if value.func.id == node.name:
                    offenders.append(node.name)
    assert offenders == [], f"self-recursive adapter helpers: {offenders}"


def test_swa_builder_bounds_validity_without_losing_expanded_slots() -> None:
    """SWA validity must come from the local-width view, not the expansion.

    ``run/20260924_104811`` crashed at
    ``sparse_swa.py: is_valid_token.copy_(slot_mapping >= 0)`` because the
    builder sized its 8192-token buffer from ``max_num_batched_tokens`` but
    PCP handed it an 8208-token expanded mapping.  A local-width view fixes
    the width; the returned metadata must still carry the expanded mapping so
    the PCP cache writer can globalize every token.  This pins both halves of
    that contract statically, independent of the HCU runtime.
    """

    import ast
    from pathlib import Path

    # Parse the source directly: importing the backend pulls in upstream
    # jit_warmup symbols absent from every installed vLLM build here.
    source_path = (
        Path(__file__).parents[2]
        / "vllm_hcu/v1/attention/backends/mla/sparse_swa.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    build = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build"
    )

    # `is_valid_token.copy_(local_slot_mapping >= 0)`: the compared tensor is
    # the localized view, never the raw expanded mapping.
    validity_sources: list[str] = []
    for node in ast.walk(build):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "copy_"
        ):
            for argument in node.args:
                if isinstance(argument, ast.Compare):
                    compared = argument.left
                    if isinstance(compared, ast.Name):
                        validity_sources.append(compared.id)
    assert validity_sources == ["local_slot_mapping"], (
        "SWA validity must derive from the localized slot view, "
        f"got {validity_sources}"
    )

    # The returned metadata keeps the expanded mapping for cache writers.
    returned_keywords = {
        keyword.arg
        for node in ast.walk(build)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DeepseekSparseSWAMetadata"
        for keyword in node.keywords
    }
    assert "slot_mapping" in returned_keywords
