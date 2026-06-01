# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


PATCH_FILE = Path(
    "/models/zb/vllm_021/vllm-hcu/vllm_hcu/patches/"
    "vllm__v1__core__kv_cache_utils.patch.py"
)
SOURCE_FILE = Path("/models/zb/vllm_021/vllm/vllm/v1/core/kv_cache_utils.py")


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("kv_patch", PATCH_FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kv_cache_utils_patch_applies_and_compiles() -> None:
    original = SOURCE_FILE.read_text()
    mod = _load_patch_module()

    patched = original
    for old, new in getattr(mod, "PATCHES", []):
        assert old in patched, "Literal patch old-snippet not found in source"
        patched = patched.replace(old, new)

    for pattern, new in getattr(mod, "REGEX_PATCHES", []):
        patched_next = re.sub(pattern, new, patched, flags=re.MULTILINE | re.DOTALL)
        assert patched_next != patched, f"Regex patch did not match: {pattern!r}"
        patched = patched_next

    assert "def _rebuild_spec_with_target_page(" in patched
    assert "target_page_size = lcm(*sorted(page_sizes))" in patched
    assert "use_lcm_fallback = any(" in patched
    assert "kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)" in patched

    compile(patched, str(SOURCE_FILE), "exec")
