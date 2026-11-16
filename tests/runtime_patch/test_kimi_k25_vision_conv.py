# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only contract tests for the optional Kimi vision Conv2d adapter."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.worker.core_fix import patch_kimi_k25_vision_conv as adapter
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


CONV_MODULE = "aiter.ops.triton.conv.conv2d"
CLASS_MARKER = "_vllm_hcu_kimi_k25_vision_conv_applied"
WRAPPER_MARKER = "_vllm_hcu_kimi_k25_vision_conv_wrapper"
ORIGINAL_ATTR = "_vllm_hcu_original_kimi_k25_vision_proj"


@pytest.fixture
def target():
    """Model just the audited vLLM surface, without torch or accelerator imports."""

    module = ModuleType(adapter.TARGET_MODULE)
    module.torch = SimpleNamespace(
        float16=object(),
        bfloat16=object(),
        float32=object(),
        compiler=SimpleNamespace(assume_constant_result=lambda fn: fn),
    )
    platform = SimpleNamespace(rocm=True)
    platform.is_rocm = lambda: platform.rocm
    module.current_platform = platform
    warnings = []
    module.logger = SimpleNamespace(warning_once=lambda *args: warnings.append(args))

    class Projection:
        def __init__(self):
            self.calls = []
            self.result = object()
            self.error = None

        def __call__(self, x):
            self.calls.append(x)
            if self.error is not None:
                raise self.error
            return self.result

    class MoonVision3dPatchEmbed:
        def __init__(self):
            self.proj = Projection()
            self.native_calls = []
            self.native_result = object()
            self.native_error = None

        def _proj(self, x):
            """An upstream implementation whose return and failures must survive."""
            self.native_calls.append(x)
            if self.native_error is not None:
                raise self.native_error
            return self.native_result

    module.MoonVision3dPatchEmbed = MoonVision3dPatchEmbed
    return SimpleNamespace(
        module=module,
        cls=MoonVision3dPatchEmbed,
        original=MoonVision3dPatchEmbed._proj,
        platform=platform,
        warnings=warnings,
        tensor=lambda dtype: SimpleNamespace(dtype=dtype),
    )


def mock_conv_import(monkeypatch, *, error=None):
    """Intercept only the optional capability loader."""

    calls = []

    def load_conv2d():
        calls.append(CONV_MODULE)
        if error is not None:
            raise error
        return object()

    monkeypatch.setattr(adapter, "_load_aiter_conv2d", load_conv2d)
    return calls


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "missing_name", ["aiter.ops.triton.conv", "aiter.ops.triton.conv.conv2d"]
)
def test_missing_optional_conv_uses_projection(
    target, monkeypatch, dtype_name, missing_name
):
    assert adapter.apply_to_module(target.module) is True
    imports = mock_conv_import(
        monkeypatch, error=ModuleNotFoundError("optional Conv2d", name=missing_name)
    )
    embed = target.cls()
    x = target.tensor(getattr(target.module.torch, dtype_name))

    assert embed._proj(x) is embed.proj.result

    assert imports == [CONV_MODULE]
    assert embed.proj.calls == [x]
    assert embed.native_calls == []
    assert len(target.warnings) == 1


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_available_conv_preserves_original_behavior(target, monkeypatch, dtype_name):
    adapter.apply_to_module(target.module)
    imports = mock_conv_import(monkeypatch)
    embed = target.cls()
    x = target.tensor(getattr(target.module.torch, dtype_name))

    assert embed._proj(x) is embed.native_result

    assert imports == [CONV_MODULE]
    assert embed.native_calls == [x]
    assert embed.proj.calls == []
    assert target.warnings == []


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("native kernel failed"),
        ImportError("native kernel import failed"),
        ModuleNotFoundError("native execution failed", name=CONV_MODULE),
    ],
)
def test_original_errors_are_not_interpreted_as_missing_capability(
    target, monkeypatch, error
):
    adapter.apply_to_module(target.module)
    mock_conv_import(monkeypatch)
    embed = target.cls()
    embed.native_error = error
    x = target.tensor(target.module.torch.float16)

    with pytest.raises(type(error)) as raised:
        embed._proj(x)

    assert raised.value is error
    assert embed.native_calls == [x]
    assert embed.proj.calls == []
    assert target.warnings == []


@pytest.mark.parametrize(
    "missing_name",
    [
        "aiter",
        "aiter.ops",
        "aiter.ops.triton",
        "triton",
        "torch",
        "aiter.ops.triton.conv.helper",
        "aiter.ops.triton.conv.conv2d.helper",
        None,
    ],
)
def test_other_missing_imports_propagate(target, monkeypatch, missing_name):
    adapter.apply_to_module(target.module)
    error = ModuleNotFoundError("dependency is missing", name=missing_name)
    mock_conv_import(monkeypatch, error=error)
    embed = target.cls()

    with pytest.raises(ModuleNotFoundError) as raised:
        embed._proj(target.tensor(target.module.torch.float16))

    assert raised.value is error
    assert embed.proj.calls == []
    assert embed.native_calls == []
    assert target.warnings == []


@pytest.mark.parametrize(
    "error",
    [
        ImportError("broken optional module", name=CONV_MODULE),
        RuntimeError("optional module initialization failed"),
    ],
)
def test_non_missing_import_failures_propagate(target, monkeypatch, error):
    adapter.apply_to_module(target.module)
    mock_conv_import(monkeypatch, error=error)
    embed = target.cls()

    with pytest.raises(type(error)) as raised:
        embed._proj(target.tensor(target.module.torch.float16))

    assert raised.value is error
    assert embed.proj.calls == []
    assert embed.native_calls == []
    assert target.warnings == []


@pytest.mark.parametrize(
    ("rocm", "dtype_name"),
    [(False, "float16"), (False, "bfloat16"), (True, "float32")],
)
def test_unaffected_platform_or_dtype_skips_optional_import(
    target, monkeypatch, rocm, dtype_name
):
    target.platform.rocm = rocm
    adapter.apply_to_module(target.module)
    imports = mock_conv_import(monkeypatch, error=AssertionError("unexpected import"))
    embed = target.cls()
    x = target.tensor(getattr(target.module.torch, dtype_name))

    assert embed._proj(x) is embed.native_result

    assert imports == []
    assert embed.native_calls == [x]
    assert embed.proj.calls == []
    assert target.warnings == []


def test_capability_is_rechecked_after_environment_changes(target, monkeypatch):
    adapter.apply_to_module(target.module)
    available = False
    imports = []

    def load_conv2d():
        imports.append(CONV_MODULE)
        if not available:
            raise ModuleNotFoundError("optional Conv2d", name=CONV_MODULE)
        return object()

    monkeypatch.setattr(adapter, "_load_aiter_conv2d", load_conv2d)
    embed = target.cls()
    x = target.tensor(target.module.torch.bfloat16)

    assert embed._proj(x) is embed.proj.result
    available = True
    assert embed._proj(x) is embed.native_result
    available = False
    assert embed._proj(x) is embed.proj.result

    assert imports == [CONV_MODULE, CONV_MODULE, CONV_MODULE]
    assert embed.proj.calls == [x, x]
    assert embed.native_calls == [x]


def test_fallback_projection_failure_propagates(target, monkeypatch):
    adapter.apply_to_module(target.module)
    mock_conv_import(
        monkeypatch, error=ModuleNotFoundError("optional Conv2d", name=CONV_MODULE)
    )
    embed = target.cls()
    error = ModuleNotFoundError("projection execution failed", name=CONV_MODULE)
    embed.proj.error = error

    with pytest.raises(ModuleNotFoundError) as raised:
        embed._proj(target.tensor(target.module.torch.float16))

    assert raised.value is error
    assert embed.native_calls == []


def test_installation_preserves_metadata_and_is_idempotent(target, monkeypatch):
    assert adapter.apply_to_module(target.module) is True
    wrapper = target.cls._proj
    assert wrapper.__wrapped__ is target.original
    assert wrapper.__name__ == target.original.__name__
    assert wrapper.__doc__ == target.original.__doc__
    assert getattr(target.cls, ORIGINAL_ATTR) is target.original
    assert getattr(target.cls, CLASS_MARKER) is True
    assert getattr(wrapper, WRAPPER_MARKER) is True

    assert adapter.apply_to_module(target.module) is False
    assert adapter.apply(target.module) is False
    assert target.cls._proj is wrapper
    mock_conv_import(monkeypatch)
    embed = target.cls()
    x = target.tensor(target.module.torch.float16)
    assert embed._proj(x) is embed.native_result
    assert embed.native_calls == [x]


def test_apply_resolves_exact_target_module(target, monkeypatch):
    monkeypatch.setitem(sys.modules, adapter.TARGET_MODULE, target.module)
    assert adapter.apply() is True
    assert getattr(target.cls, ORIGINAL_ATTR) is target.original


def test_wrong_module_is_rejected_before_mutation(target):
    target.module.__name__ = "vllm.model_executor.models.other_vit"
    with pytest.raises(PatchCompatibilityError):
        adapter.apply_to_module(target.module)
    assert target.cls._proj is target.original
    assert not hasattr(target.cls, CLASS_MARKER)
    assert not hasattr(target.cls, ORIGINAL_ATTR)


@pytest.mark.parametrize("replacement", [None, SimpleNamespace(), object()])
def test_missing_or_non_class_target_is_rejected(target, replacement):
    if replacement is None:
        del target.module.MoonVision3dPatchEmbed
    else:
        target.module.MoonVision3dPatchEmbed = replacement
    with pytest.raises(PatchCompatibilityError):
        adapter.apply_to_module(target.module)


@pytest.mark.parametrize("replacement", [None, 42])
def test_missing_or_non_callable_projection_is_rejected(target, replacement):
    if replacement is None:
        del target.cls._proj
    else:
        target.cls._proj = replacement
    with pytest.raises(PatchCompatibilityError):
        adapter.apply_to_module(target.module)
    assert not hasattr(target.cls, CLASS_MARKER)
    assert not hasattr(target.cls, ORIGINAL_ATTR)


def renamed_argument(self, inputs):
    return inputs


def additional_argument(self, x, y):
    return x


def default_argument(self, x=None):
    return x


def keyword_only_argument(self, *, x):
    return x


def variadic_argument(self, x, *args):
    return x


@pytest.mark.parametrize(
    "replacement",
    [
        renamed_argument,
        additional_argument,
        default_argument,
        keyword_only_argument,
        variadic_argument,
    ],
)
def test_changed_projection_signature_is_rejected(target, replacement):
    target.cls._proj = replacement
    with pytest.raises(PatchCompatibilityError):
        adapter.apply_to_module(target.module)
    assert target.cls._proj is replacement
    assert not hasattr(target.cls, CLASS_MARKER)
    assert not hasattr(target.cls, ORIGINAL_ATTR)


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("current_platform", None),
        ("current_platform", SimpleNamespace(is_rocm=False)),
        ("torch", None),
        ("torch", SimpleNamespace(float16=object())),
        (
            "torch",
            SimpleNamespace(float16=object(), bfloat16=object()),
        ),
        (
            "torch",
            SimpleNamespace(
                float16=object(),
                bfloat16=object(),
                compiler=SimpleNamespace(assume_constant_result=False),
            ),
        ),
        ("logger", None),
        ("logger", SimpleNamespace(warning_once=False)),
    ],
)
def test_missing_runtime_contract_is_rejected_before_mutation(
    target, attribute, replacement
):
    setattr(target.module, attribute, replacement)
    with pytest.raises(PatchCompatibilityError):
        adapter.apply_to_module(target.module)
    assert target.cls._proj is target.original
    assert not hasattr(target.cls, CLASS_MARKER)
    assert not hasattr(target.cls, ORIGINAL_ATTR)


@pytest.mark.parametrize("damage", ["wrapper_marker", "original", "replacement"])
def test_stale_installation_markers_are_rejected(target, damage):
    adapter.apply_to_module(target.module)
    if damage == "wrapper_marker":
        delattr(target.cls._proj, WRAPPER_MARKER)
    elif damage == "original":
        delattr(target.cls, ORIGINAL_ATTR)
    else:
        target.cls._proj = target.original

    with pytest.raises(PatchCompatibilityError, match="stale"):
        adapter.apply_to_module(target.module)


@pytest.mark.parametrize(
    "damage",
    ["missing_class_marker", "false_class_marker", "wrapper_only", "original_only"],
)
def test_incomplete_installation_markers_are_rejected(target, damage):
    if damage in ("missing_class_marker", "false_class_marker"):
        adapter.apply_to_module(target.module)
        if damage == "missing_class_marker":
            delattr(target.cls, CLASS_MARKER)
        else:
            setattr(target.cls, CLASS_MARKER, False)
    elif damage == "wrapper_only":
        setattr(target.cls._proj, WRAPPER_MARKER, True)
    else:
        setattr(target.cls, ORIGINAL_ATTR, target.original)

    with pytest.raises(PatchCompatibilityError, match="incomplete"):
        adapter.apply_to_module(target.module)


def test_import_is_cpu_only_and_does_not_load_optional_or_target_modules():
    # Synthetic parents isolate the adapter's import behavior from package
    # initializers. The real sibling _common is still imported from disk.
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import sys
        from pathlib import Path
        from types import ModuleType

        adapter_path = Path(sys.argv[1])
        forbidden_roots = ('torch', 'aiter', 'vllm')

        class RejectRuntimeImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == root or fullname.startswith(root + '.')
                       for root in forbidden_roots):
                    raise AssertionError(
                        'runtime import while loading adapter: ' + fullname)

        sys.meta_path.insert(0, RejectRuntimeImports())
        package = 'vllm_hcu.patch.worker.core_fix'
        parts = package.split('.')
        for count in range(1, len(parts) + 1):
            name = '.'.join(parts[:count])
            parent = ModuleType(name)
            parent.__path__ = [str(adapter_path.parent)] if name == package else []
            sys.modules[name] = parent
        fullname = package + '.patch_kimi_k25_vision_conv'
        spec = importlib.util.spec_from_file_location(fullname, adapter_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        assert module.TARGET_MODULE == 'vllm.model_executor.models.kimi_k25_vit'
        assert module.PATCH_ID == 'worker.core_fix.kimi_k25.vision_conv'
        assert not any(name == root or name.startswith(root + '.')
                       for root in forbidden_roots for name in sys.modules)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(adapter.__file__).resolve())],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("available", [True, False], ids=["available", "missing"])
def test_fullgraph_compile_with_optional_conv_capability(monkeypatch, available):
    import torch
    from vllm.logger import init_logger

    # Exercise the real import helper and compiler decorator without executing
    # an AITER kernel or loading an accelerator device.
    conv_module = ModuleType(CONV_MODULE)
    conv_module.conv2d = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, CONV_MODULE, conv_module if available else None)

    vision = ModuleType(adapter.TARGET_MODULE)
    vision.torch = torch
    vision.current_platform = SimpleNamespace(is_rocm=lambda: True)
    vision.logger = init_logger(__name__)

    class MoonVision3dPatchEmbed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(
                3, 8, 2, stride=2, device="cpu", dtype=torch.bfloat16
            )

        def _proj(self, x):
            return self.proj(x)

    vision.MoonVision3dPatchEmbed = MoonVision3dPatchEmbed
    torch.compiler.reset()
    try:
        assert adapter.apply_to_module(vision) is True
        model = MoonVision3dPatchEmbed().eval()
        x = torch.randn(2, 3, 8, 8, device="cpu", dtype=torch.bfloat16)
        with torch.no_grad():
            # Do not warm the wrapper before compiling either capability path.
            expected = model.proj(x)
            compiled = torch.compile(model._proj, backend="eager", fullgraph=True)
            actual = compiled(x)
        torch.testing.assert_close(actual, expected)
    finally:
        torch.compiler.reset()
