# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patches for vLLM compatibility with HCU platform.

This module contains patches that modify vLLM source files at runtime
to ensure compatibility with the HCU Triton version.

Each ``*.patch.py`` file targets one vLLM module (see filename mapping in
``_get_patch_files``). Use **one** of the styles below per patch entry.

----------------------------------------------------------------------
1) String match (literal) — default, use for most patches
----------------------------------------------------------------------

Put ``(old, new)`` in ``PATCHES``. ``old`` must be a normal ``str`` (not
``re.compile``). The engine finds the exact substring and replaces it.
Parentheses, dots, and brackets in ``old`` are ordinary characters.

Example::

    PATCHES = [
        (
            "ops.concat_and_cache_mla(",
            "torch.ops.hcu_ops.concat_and_cache_mla(",
        ),
        (
            '''
            from vllm import _custom_ops as ops
            ''',
            '''
            from vllm_hcu.v1.attention.backends.fa_utils import hcu_ops
            ''',
        ),
    ]

Note: writing ``r"ops\\.foo"`` in the patch file is only a Python raw string;
it still counts as **literal** match, not regex.

----------------------------------------------------------------------
2) Regex match — only when you explicitly opt in (three equivalent ways)
----------------------------------------------------------------------

Regex entries use ``re.MULTILINE | re.DOTALL``. Escape metacharacters in the
pattern (e.g. ``\\.`` for a dot, ``\\(`` for ``(``).

**Way A — separate list (recommended when mixing with literal patches)**::

    import re

    PATCHES = [
        ("exact literal old", "literal new"),
    ]

    REGEX_PATCHES = [
        (
            r"ops\.concat_and_cache_mla\(\s*kv_c_normed",
            r"torch.ops.hcu_ops.concat_and_cache_mla(kv_c_normed",
        ),
    ]

**Way B — tagged tuple inside ``PATCHES``**::

    PATCHES = [
        (
            "regex",  # or "re"
            r"from vllm import _custom_ops.*?ops\.concat",
            "replacement text",
        ),
    ]

**Way C — compiled pattern inside ``PATCHES``**::

    import re

    PATCHES = [
        (
            re.compile(r"raise\s+NotImplementedError", re.MULTILINE | re.DOTALL),
            "raise RuntimeError('HCU')",
        ),
    ]

Do **not** put regex-only patterns as plain ``(str, str)`` in ``PATCHES``;
they will be treated as literal strings and will not match as regex.
"""

import importlib.util
import re
import sys
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

_patches_applied = False

_REGEX_FLAGS = re.MULTILINE | re.DOTALL

# Normalized entry: (kind, old, new) where kind is "literal" or "regex"
_PatchEntry = tuple[str, str | re.Pattern[str], str]


def _compile_regex(pattern: str) -> re.Pattern[str] | None:
    try:
        return re.compile(pattern, _REGEX_FLAGS)
    except re.error as e:
        logger.warning(f"Invalid regex patch pattern {pattern!r}: {e}")
        return None


def _normalize_patch_entry(entry: object) -> _PatchEntry | None:
    if not isinstance(entry, tuple):
        logger.warning(f"Invalid patch entry (expected tuple): {entry!r}")
        return None

    if len(entry) == 3 and entry[0] in ("regex", "re"):
        pattern = entry[1]
        new = entry[2]
        if isinstance(pattern, re.Pattern):
            return ("regex", pattern, new)
        if isinstance(pattern, str):
            compiled = _compile_regex(pattern)
            if compiled is None:
                return None
            return ("regex", compiled, new)
        logger.warning(f"Invalid regex patch pattern type: {type(pattern)}")
        return None

    if len(entry) == 2:
        old, new = entry
        if isinstance(old, re.Pattern):
            return ("regex", old, new)
        return ("literal", old, new)

    logger.warning(f"Invalid patch entry length or format: {entry!r}")
    return None


def _patch_in_source(kind: str, old: str | re.Pattern[str], source: str) -> bool:
    if kind == "literal":
        return old in source
    return old.search(source) is not None


def _apply_patch_entry(
    kind: str, old: str | re.Pattern[str], new: str, source: str
) -> str:
    if kind == "literal":
        return source.replace(old, new)
    return old.sub(new, source)


def _get_patch_files():
    """Get all patch files in the patches directory."""
    patches_dir = Path(__file__).parent
    patch_files = []

    for patch_file in patches_dir.glob("*.patch.py"):
        # Extract module name from filename
        # Format: module.name.patch.py -> module.name
        module_name = patch_file.stem.rsplit(".patch", 1)[0]
        # Convert filename format to module format
        # vllm__attention__ops__triton_unified_attention -> vllm.attention.ops.triton_unified_attention
        # .../quantization/__init__.py -> stem ...quantization____init[__.]patch
        # Strip before __ -> . so "__init__" is not mangled into "..init." / "..init"
        if module_name.endswith("____init__"):
            module_name = module_name[: -len("____init__")]
        elif module_name.endswith("____init"):
            module_name = module_name[: -len("____init")]
        module_name = module_name.replace("__", ".")
        patch_files.append((module_name, patch_file))
    return patch_files

def _load_patch_config(patch_file: Path) -> list[_PatchEntry]:
    """Load and normalize literal + regex patches from a patch file."""
    spec = importlib.util.spec_from_file_location("patch_config", patch_file)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.warning(f"Failed to load patch config from {patch_file}: {e}")
        return []

    entries: list[_PatchEntry] = []
    for entry in getattr(module, "PATCHES", []):
        normalized = _normalize_patch_entry(entry)
        if normalized is not None:
            entries.append(normalized)

    for entry in getattr(module, "REGEX_PATCHES", []):
        if not isinstance(entry, tuple) or len(entry) != 2:
            logger.warning(
                f"REGEX_PATCHES entry must be (pattern, new), got {entry!r}"
            )
            continue
        pattern, new = entry
        if isinstance(pattern, re.Pattern):
            compiled = pattern
        elif isinstance(pattern, str):
            compiled = _compile_regex(pattern)
            if compiled is None:
                continue
        else:
            logger.warning(
                f"REGEX_PATCHES pattern must be str or re.Pattern, "
                f"got {type(pattern)}"
            )
            continue
        entries.append(("regex", compiled, new))

    return entries


def apply_patches():
    """Apply all patches for MUSA compatibility.

    This function should be called early during platform initialization.
    """
    global _patches_applied
    if _patches_applied:
        return

    patch_files = _get_patch_files()

    for module_name, patch_file in patch_files:
        try:
            # Find the module spec
            try:
                spec = importlib.util.find_spec(module_name)
            except (ModuleNotFoundError, ImportError) as e:
                # Module doesn't exist in this vLLM version (e.g., vllm.worker.worker
                # exists in vLLM 0.10.x but not in 0.13.0 where V0 engine was removed)
                # or has circular import issues during spec discovery
                logger.debug(
                    f"Module {module_name} not found or has import issues: {e}, "
                    "skipping patch (this is expected for version-specific patches "
                    "or when modules are not yet fully initialized)"
                )
                continue
            if spec is None or spec.origin is None:
                logger.debug(f"Module {module_name} not found, skipping patch")
                continue

            # Read the source file
            try:
                with open(spec.origin, "r") as f:
                    source = f.read()
            except (IOError, OSError) as e:
                logger.debug(f"Cannot read {spec.origin}: {e}, skipping patch")
                continue

            # Load patches from patch file
            patches = _load_patch_config(patch_file)
            if not patches:
                continue
            needs_patch = any(
                _patch_in_source(kind, old, source) for kind, old, _ in patches
            )
            if not needs_patch:
                logger.debug(f"No patches needed for {module_name}")
                continue

            patched_source = source
            applied_count = 0

            for i, (kind, old, new) in enumerate(patches):
                patch_id = f"# PATCHED_{module_name.replace('.', '_')}_patch_{i}"

                if patch_id in patched_source:
                    continue

                if not _patch_in_source(kind, old, patched_source):
                    logger.debug(
                        f"Patch segment {i} ({kind}) not found in {module_name}, "
                        "skipping"
                    )
                    continue

                before = patched_source
                patched_source = _apply_patch_entry(kind, old, new, patched_source)
                if patched_source == before:
                    logger.warning(
                        f"Patch {i} ({kind}) for {module_name} matched but made "
                        "no changes"
                    )
                    continue

                patched_source += f"\n{patch_id}\n"
                applied_count += 1
            
            if applied_count > 0:
                # Write back the patched source
                with open(spec.origin, "w") as f:
                    f.write(patched_source)

                # Remove from cache to force reload
                if module_name in sys.modules:
                    del sys.modules[module_name]

                logger.info(f"Applied {applied_count} patch(es) to {module_name}")

        except Exception as e:
            # More detailed error handling for circular imports
            if "circular import" in str(e) or "partially initialized" in str(e):
                logger.debug(
                    f"Skipping patch for {module_name} due to circular import "
                    f"during initialization: {e}"
                )
            else:
                logger.warning(f"Failed to apply patches to {module_name}: {e}")

    _patches_applied = True
