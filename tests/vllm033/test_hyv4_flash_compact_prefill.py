import ast
from importlib import util
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/compact_prefill.py"
)
_SPEC = util.spec_from_file_location("hyv4_flash_compact_prefill", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_LEAF = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LEAF)


@pytest.mark.parametrize(
    ("seq_lens", "query_tokens", "expected"),
    [
        ([462, 901, 2738], 4096, [0, 462, 1363, 4101]),
        ([1024, 1024], 1024, [0, 1024, 2048]),
        ([8192, 8192], 16384, [0, 8192, 16384]),
        ([4096], 256, None),
        ([4097, 4096], 4096, None),
        ([8192, 8193], 16384, None),
        ([], 4096, None),
        ([0, 4096], 4096, None),
    ],
)
def test_compact_prefill_keeps_context_rows_within_workspace(
    seq_lens, query_tokens, expected
):
    assert _LEAF.compact_prefill_offsets(seq_lens, query_tokens) == expected


def test_sparse_and_window_backends_use_the_compact_prefill_builder():
    source = _SOURCE.with_name("swa.py")
    classes = {"FlashSparseMetadataBuilder", "FlashSparseBackend", "FlashSWABackend"}
    definitions = [node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name in classes]
    compact_builder = type("CompactPrefillMetadataBuilder", (), {})
    namespace = {
        "CompactPrefillMetadataBuilder": compact_builder,
        "FlashSparseKernelBackend": object,
        "AttentionCGSupport": SimpleNamespace(UNIFORM_BATCH=object()),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    for backend in ("FlashSparseBackend", "FlashSWABackend"):
        assert issubclass(namespace[backend].get_builder_cls(), compact_builder)


def test_compact_prefill_respects_the_explicit_batch_and_absolute_row_limit():
    lengths = [4096] * 6
    assert _LEAF.compact_prefill_offsets(lengths, 24576) is None
    assert _LEAF.compact_prefill_offsets(
        lengths, 24576, max_batched_tokens=24576
    ) == list(range(0, 24577, 4096))
    assert _LEAF.compact_prefill_offsets([24577], 24576, max_batched_tokens=32768) is None
