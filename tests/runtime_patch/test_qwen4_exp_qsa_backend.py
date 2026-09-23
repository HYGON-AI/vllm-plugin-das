# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import logging
from types import ModuleType

import pytest
import torch

from vllm_hcu.platforms import envs as hcu_envs
from vllm_hcu.patch.worker.op_opt import (
    patch_qwen4_exp_qsa_flash_attn as qsa_patch,
)
from vllm_hcu.v1.attention.backends import qsa


def _triton_mqa(*args, **kwargs):
    del args, kwargs
    return "triton-mqa"


def _triton_sparse(*args, **kwargs):
    del args, kwargs
    return "triton-sparse"


def _backend(monkeypatch: pytest.MonkeyPatch):
    return qsa.get_qsa_kernel_backend(
        triton_mqa_paged=_triton_mqa,
        triton_sparse_gqa_paged_attn=_triton_sparse,
    )


@pytest.fixture(autouse=True)
def _reset_qsa_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qsa, "_legacy_qsa_cutlass_warning_emitted", False)
    for name in hcu_envs.hcu_vllm_environment_variables:
        hcu_envs.__dict__.pop(name, None)
    yield
    for name in hcu_envs.hcu_vllm_environment_variables:
        hcu_envs.__dict__.pop(name, None)


def test_custom_ops_master_gate_forces_triton_and_ignores_invalid_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_QSA_BACKEND", "invalid")

    backend = _backend(monkeypatch)

    assert backend.name == qsa.QSA_BACKEND_TRITON
    assert backend.mqa_paged_score is _triton_mqa
    assert backend.sparse_gqa_paged_attn is _triton_sparse


def test_qsa_backend_defaults_to_cutlass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.delenv("VLLM_HCU_QSA_BACKEND", raising=False)
    cutlass_mqa = object()
    cutlass_sparse = object()
    monkeypatch.setattr(
        qsa,
        "_load_flash_qsa_kernels",
        lambda: (cutlass_mqa, cutlass_sparse),
    )

    backend = _backend(monkeypatch)

    assert backend.name == qsa.QSA_BACKEND_CUTLASS
    assert backend.mqa_paged_score is cutlass_mqa
    assert backend.sparse_gqa_paged_attn is cutlass_sparse


@pytest.mark.parametrize(
    ("configured", "loader", "expected_name"),
    [
        ("triton", None, qsa.QSA_BACKEND_TRITON),
        ("cutlass", "flash", qsa.QSA_BACKEND_CUTLASS),
        ("boltops", "boltops", qsa.QSA_BACKEND_BOLTOPS),
    ],
)
def test_explicit_qsa_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    loader: str | None,
    expected_name: str,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_QSA_BACKEND", configured)
    selected_mqa = object()
    selected_sparse = object()
    if loader == "flash":
        monkeypatch.setattr(
            qsa,
            "_load_flash_qsa_kernels",
            lambda: (selected_mqa, selected_sparse),
        )
    elif loader == "boltops":
        monkeypatch.setattr(
            qsa,
            "_load_boltops_qsa_kernels",
            lambda: (selected_mqa, selected_sparse),
        )

    backend = _backend(monkeypatch)

    assert backend.name == expected_name
    if loader is not None:
        assert backend.mqa_paged_score is selected_mqa
        assert backend.sparse_gqa_paged_attn is selected_sparse
    else:
        assert backend.mqa_paged_score is _triton_mqa
        assert backend.sparse_gqa_paged_attn is _triton_sparse


def test_invalid_qsa_backend_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_QSA_BACKEND", "cutlasss")

    with pytest.raises(ValueError, match="VLLM_HCU_QSA_BACKEND"):
        _backend(monkeypatch)


@pytest.mark.parametrize(
    ("configured", "loader_name"),
    [("cutlass", "_load_flash_qsa_kernels"), ("boltops", "_load_boltops_qsa_kernels")],
)
def test_optional_backend_load_failure_falls_back_to_triton(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    configured: str,
    loader_name: str,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_QSA_BACKEND", configured)

    def fail_loader():
        raise RuntimeError(f"{configured} unavailable")

    monkeypatch.setattr(qsa, loader_name, fail_loader)
    with caplog.at_level(logging.WARNING, logger=qsa.__name__):
        backend = _backend(monkeypatch)

    assert backend.name == qsa.QSA_BACKEND_TRITON
    assert "falling back to Triton" in caplog.text


def test_legacy_cutlass_variable_warns_but_does_not_control_selection(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_QSA_BACKEND", "triton")
    monkeypatch.setenv("VLLM_HCU_USE_QSA_CUTLASS", "0")

    with caplog.at_level(logging.WARNING, logger=qsa.__name__):
        backend = _backend(monkeypatch)

    assert backend.name == qsa.QSA_BACKEND_TRITON
    assert "VLLM_HCU_USE_QSA_CUTLASS is deprecated and ignored" in caplog.text


def test_qsa_backend_environment_definition_replaces_legacy_definition(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_HCU_QSA_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_HCU_USE_QSA_CUTLASS", raising=False)
    for name in hcu_envs.hcu_vllm_environment_variables:
        hcu_envs.__dict__.pop(name, None)

    assert hcu_envs.VLLM_HCU_QSA_BACKEND == "cutlass"
    assert "VLLM_HCU_QSA_BACKEND" in hcu_envs.hcu_vllm_environment_variables
    assert "VLLM_HCU_USE_QSA_CUTLASS" not in hcu_envs.hcu_vllm_environment_variables


def test_qsa_runtime_patch_targets_backend_dispatcher_entry_points():
    assert qsa_patch.TARGET_MODULE == "vllm.models.qwen4_exp.amd.ops.qsa"
    assert qsa_patch.TARGETS == (
        "vllm.models.qwen4_exp.amd.ops.qsa.qsa_mqa_paged",
        "vllm.models.qwen4_exp.amd.ops.qsa.qsa_sparse_paged_attention",
    )


def test_qsa_runtime_patch_delegates_both_compute_entry_points(
    monkeypatch: pytest.MonkeyPatch,
):
    module = ModuleType(qsa_patch.TARGET_MODULE)

    def original_mqa(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns=None,
        score_scale=None,
    ):
        del (
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            compress_ratio,
            num_columns,
            score_scale,
        )
        return "original-mqa"

    def original_sparse(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out=None,
    ):
        del q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
        return "original-sparse"

    selected_mqa = lambda *args, **kwargs: "selected-mqa"
    selected_sparse = lambda *args, **kwargs: "selected-sparse"
    module.qsa_mqa_paged = original_mqa
    module.qsa_sparse_paged_attention = original_sparse

    def fake_backend(**kwargs):
        assert kwargs["triton_mqa_paged"] is original_mqa
        assert kwargs["triton_sparse_gqa_paged_attn"] is original_sparse
        return qsa.QSAKernelBackend(
            name=qsa.QSA_BACKEND_TRITON,
            mqa_paged_score=selected_mqa,
            sparse_gqa_paged_attn=selected_sparse,
        )

    monkeypatch.setattr(qsa_patch, "get_qsa_kernel_backend", fake_backend)

    assert qsa_patch.apply_to_module(module) is True
    assert module.qsa_mqa_paged(1, 2, 3, 4, 5, 6, 7) == "selected-mqa"
    assert module.qsa_sparse_paged_attention(1, 2, 3, 4, 5, 6) == (
        "selected-sparse"
    )
    assert qsa_patch.apply_to_module(module) is False


@pytest.mark.parametrize(
    ("backend_name", "expected_dtype"),
    [
        (qsa.QSA_BACKEND_TRITON, torch.int64),
        (qsa.QSA_BACKEND_CUTLASS, torch.int64),
        (qsa.QSA_BACKEND_BOLTOPS, torch.int32),
    ],
)
def test_query_positions_dtype_depends_on_qsa_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    expected_dtype: torch.dtype,
):
    module = ModuleType(qsa_patch.TARGET_MODULE)
    seen: dict[str, torch.dtype] = {}

    def original_mqa(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns=None,
        score_scale=None,
    ):
        del (
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            compress_ratio,
            num_columns,
            score_scale,
        )
        return "original-mqa"

    def original_sparse(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out=None,
    ):
        del q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
        return "original-sparse"

    def selected_mqa(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns=None,
        score_scale=None,
    ):
        del (
            q,
            k_cache,
            page_table,
            token_to_req,
            sequence_lengths,
            compress_ratio,
            num_columns,
            score_scale,
        )
        seen["query_positions"] = query_positions.dtype
        return "selected-mqa"

    module.qsa_mqa_paged = original_mqa
    module.qsa_sparse_paged_attention = original_sparse
    monkeypatch.setattr(
        qsa_patch,
        "get_qsa_kernel_backend",
        lambda **kwargs: qsa.QSAKernelBackend(
            name=backend_name,
            mqa_paged_score=selected_mqa,
            sparse_gqa_paged_attn=lambda *args, **kwargs: "selected-sparse",
        ),
    )

    assert qsa_patch.apply_to_module(module) is True
    i32 = torch.zeros(2, dtype=torch.int32)
    i64 = torch.zeros(2, dtype=torch.int64)
    assert module.qsa_mqa_paged(None, None, i32, i32, i64, i32, 1) == (
        "selected-mqa"
    )
    # BoltOPs pins its index metadata to int32, so the int64 logical-position
    # buffer must be narrowed; the other backends consume it unchanged.
    assert seen["query_positions"] is expected_dtype


def test_optional_backend_load_failure_is_cached_after_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_QSA_BACKEND", "cutlass")
    attempts = []
    constructed = []

    class TrackedError(RuntimeError):
        def __init__(self, *args):
            constructed.append(1)
            super().__init__(*args)

    def fail_loader():
        attempts.append(1)
        raise TrackedError("cutlass unavailable")

    monkeypatch.setattr(qsa, "_load_flash_qsa_kernels", fail_loader)
    with caplog.at_level(logging.WARNING, logger=qsa.__name__):
        first = _backend(monkeypatch)
        second = _backend(monkeypatch)
        third = _backend(monkeypatch)

    assert first.name == qsa.QSA_BACKEND_TRITON
    assert second.name == qsa.QSA_BACKEND_TRITON
    assert third.name == qsa.QSA_BACKEND_TRITON
    assert first.mqa_paged_score is _triton_mqa
    assert third.sparse_gqa_paged_attn is _triton_sparse
    # A decode loop calls this per forward. Once the backend is known to be
    # unavailable the import, the warning, and the exception must not repeat,
    # and the fallback must not travel through exception control flow at all.
    assert len(attempts) == 1
    assert len(constructed) == 1
    assert caplog.text.count("falling back to Triton") == 1


def test_qsa_runtime_patch_narrows_metadata_for_boltops(
    monkeypatch: pytest.MonkeyPatch,
):
    module = ModuleType(qsa_patch.TARGET_MODULE)
    seen: dict[str, tuple[torch.dtype, ...]] = {}

    def original_mqa(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns=None,
        score_scale=None,
    ):
        del (
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            compress_ratio,
            num_columns,
            score_scale,
        )
        return "original-mqa"

    def original_sparse(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out=None,
    ):
        del q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
        return "original-sparse"

    def selected_mqa(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns=None,
        score_scale=None,
    ):
        del q, k_cache, compress_ratio, num_columns, score_scale
        seen["dtypes"] = (
            page_table.dtype,
            token_to_req.dtype,
            query_positions.dtype,
            sequence_lengths.dtype,
        )
        return "selected-mqa"

    module.qsa_mqa_paged = original_mqa
    module.qsa_sparse_paged_attention = original_sparse

    monkeypatch.setattr(
        qsa_patch,
        "get_qsa_kernel_backend",
        lambda **kwargs: qsa.QSAKernelBackend(
            name=qsa.QSA_BACKEND_BOLTOPS,
            mqa_paged_score=selected_mqa,
            sparse_gqa_paged_attn=lambda *args, **kwargs: "selected-sparse",
        ),
    )

    assert qsa_patch.apply_to_module(module) is True
    i32 = torch.zeros(2, dtype=torch.int32)
    i64 = torch.zeros(2, dtype=torch.int64)
    assert module.qsa_mqa_paged(None, None, i32, i32, i64, i32, 1) == (
        "selected-mqa"
    )
    assert seen["dtypes"] == (torch.int32,) * 4
