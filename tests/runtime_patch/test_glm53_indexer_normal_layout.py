# SPDX-License-Identifier: Apache-2.0

from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_glm5next_channel_fp8 as glm_patch
from vllm_hcu.platforms import envs as henvs


class _FakeKernel:
    arg_names = ["x", "y", "LAYOUT", "PRESHUFFLE"]

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return run


def _fake_kpool_module():
    class SparseAttnIndexerKpool:
        def forward_hip(
            self,
            hidden_states,
            q_quant,
            k,
            weights,
            *,
            gate_score=None,
            compress_ape=None,
            index_kpool=1,
            positions=None,
        ):
            return None

    kpool_ops = ModuleType("fake_kpool_ops")
    for attr, _, _ in glm_patch._NORMAL_LAYOUT_KPOOL_KERNELS:
        setattr(kpool_ops, attr, _FakeKernel())
    kpool = ModuleType(glm_patch.KPOOL_MODULE)
    kpool.SparseAttnIndexerKpool = SparseAttnIndexerKpool
    kpool.kpool_ops = kpool_ops
    return kpool


class _RecordingLogger:
    def __init__(self):
        self.records = []

    def info(self, msg, *args):
        self.records.append(("info", msg % args))

    def warning(self, msg, *args):
        self.records.append(("warning", msg % args))


def _record_layout_logs(monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(glm_patch, "_LAYOUT_LOGGER", recorder)
    return recorder.records


def _restore_upstream_sparse(monkeypatch, upstream_sparse):
    attrs = ["rocm_fp8_mqa_logits", "rocm_fp8_paged_mqa_logits"]
    attrs += [attr for attr, _, _ in glm_patch._NORMAL_LAYOUT_SPARSE_KERNELS]
    for attr in attrs:
        monkeypatch.setattr(upstream_sparse, attr, getattr(upstream_sparse, attr))


def test_pinned_constexpr_replaces_positional_and_keyword_arguments() -> None:
    kernel = _FakeKernel()
    pinned = glm_patch._PinnedConstexprKernel(kernel, "LAYOUT", "NORMAL")

    pinned[(1,)](1, 2, "SHUFFLE", True)
    pinned[(2,)](1, 2)
    pinned[(3,)](1, 2, LAYOUT="SHUFFLE", PRESHUFFLE=True)

    assert kernel.launches == [
        ((1,), (1, 2, "NORMAL", True), {}),
        ((2,), (1, 2), {"LAYOUT": "NORMAL"}),
        ((3,), (1, 2), {"LAYOUT": "NORMAL", "PRESHUFFLE": True}),
    ]
    assert pinned.arg_names is kernel.arg_names


def test_pinned_constexpr_rejects_unknown_argument() -> None:
    with pytest.raises(glm_patch.PatchCompatibilityError, match="MISSING"):
        glm_patch._PinnedConstexprKernel(_FakeKernel(), "MISSING", 0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "preshuffle"), ("NORMAL", "normal"), (" preshuffle ", "preshuffle")],
)
def test_indexer_kcache_layout_env(monkeypatch, raw, expected) -> None:
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    if raw is None:
        monkeypatch.delenv(henvs.INDEXER_KCACHE_LAYOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(henvs.INDEXER_KCACHE_LAYOUT_ENV, raw)
    assert henvs.indexer_kcache_layout() == expected


@pytest.mark.parametrize("master", ["0", "false", "False"])
def test_indexer_kcache_normal_layout_respects_custom_ops_master(
    monkeypatch, master: str
) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv(henvs.INDEXER_KCACHE_LAYOUT_ENV, "normal")
    assert henvs.indexer_kcache_layout() == "preshuffle"


def test_indexer_kcache_layout_env_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv(henvs.INDEXER_KCACHE_LAYOUT_ENV, "shuffle")
    with pytest.raises(ValueError, match="shuffle"):
        henvs.indexer_kcache_layout()


def test_dispatcher_rejects_unknown_kcache_layout() -> None:
    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as hcu_sparse

    with pytest.raises(ValueError, match="tiled"):
        hcu_sparse.rocm_fp8_paged_mqa_logits(
            torch.empty((1, 1, 1, 128), dtype=torch.float8_e4m3fn),
            torch.empty((1, 16, 1, 132), dtype=torch.uint8),
            torch.ones((1, 1)),
            torch.tensor([1], dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
            None,
            16,
            kcache_layout="tiled",
        )


@pytest.mark.parametrize(
    ("layout", "custom_ops", "normal"),
    [
        ("preshuffle", "1", False),
        ("normal", "1", True),
        ("normal", "0", False),
    ],
)
def test_glm5next_layout_env_selects_writers_and_decode_reader(
    monkeypatch, layout: str, custom_ops: str, normal: bool
) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as upstream_sparse

    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as hcu_sparse

    logs = _record_layout_logs(monkeypatch)
    monkeypatch.setenv(henvs.INDEXER_KCACHE_LAYOUT_ENV, layout)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", custom_ops)
    monkeypatch.setattr(hcu_sparse, "on_gfx938", lambda: True)
    _restore_upstream_sparse(monkeypatch, upstream_sparse)
    calls = []

    def dispatcher(*args, **kwargs):
        calls.append(kwargs)
        return torch.ones((1, 16))

    monkeypatch.setattr(hcu_sparse, "rocm_fp8_paged_mqa_logits", dispatcher)
    kpool = _fake_kpool_module()
    glm_patch._patch_sparse_indexer_kpool(kpool)

    upstream_sparse.rocm_fp8_paged_mqa_logits(
        torch.empty((1, 1, 1, 128), dtype=torch.float8_e4m3fn),
        torch.empty((1, 16, 1, 132), dtype=torch.uint8),
        torch.ones((1, 1)),
        torch.tensor([1], dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int32),
        torch.empty(0),
        16,
    )
    assert ("kcache_layout" in calls[0]) is normal
    if normal:
        assert calls[0]["kcache_layout"] == "normal"
    for module, kernels in (
        (kpool.kpool_ops, glm_patch._NORMAL_LAYOUT_KPOOL_KERNELS),
        (upstream_sparse, glm_patch._NORMAL_LAYOUT_SPARSE_KERNELS),
    ):
        for attr, _, _ in kernels:
            pinned = isinstance(
                getattr(module, attr), glm_patch._PinnedConstexprKernel
            )
            assert pinned is normal
    assert len(logs) == 1
    assert logs[0][0] == "info"
    expected_layout = "normal" if normal else "preshuffle"
    assert f"layout: {expected_layout} " in logs[0][1]


def test_glm5next_normal_layout_is_ignored_off_gfx938(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as upstream_sparse

    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as hcu_sparse

    logs = _record_layout_logs(monkeypatch)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv(henvs.INDEXER_KCACHE_LAYOUT_ENV, "normal")
    monkeypatch.setattr(hcu_sparse, "on_gfx938", lambda: False)
    _restore_upstream_sparse(monkeypatch, upstream_sparse)
    kpool = _fake_kpool_module()
    glm_patch._patch_sparse_indexer_kpool(kpool)

    for attr, _, _ in glm_patch._NORMAL_LAYOUT_KPOOL_KERNELS:
        assert isinstance(getattr(kpool.kpool_ops, attr), _FakeKernel)
    assert [level for level, _ in logs] == ["warning", "info"]
    assert "only applies to gfx938" in logs[0][1]
    assert "layout: preshuffle " in logs[1][1]


def test_dispatcher_rejects_normal_layout_off_gfx938(monkeypatch) -> None:
    from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as hcu_sparse

    monkeypatch.setattr(hcu_sparse, "on_gfx938", lambda: False)
    with pytest.raises(ValueError, match="only supported on gfx938"):
        hcu_sparse.rocm_fp8_paged_mqa_logits(
            torch.empty((1, 1, 1, 128), dtype=torch.float8_e4m3fn),
            torch.empty((1, 16, 1, 132), dtype=torch.uint8),
            torch.ones((1, 1)),
            torch.tensor([1], dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
            None,
            16,
            kcache_layout="normal",
        )


def test_glm5next_normal_layout_requires_kpool_ops(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as upstream_sparse

    kpool = ModuleType(glm_patch.KPOOL_MODULE)
    with pytest.raises(glm_patch.PatchCompatibilityError, match="kpool_ops"):
        glm_patch._install_normal_indexer_kcache_layout(kpool, upstream_sparse)
