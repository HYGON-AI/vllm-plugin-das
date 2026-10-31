"""Independent Flash lifecycle CPU contracts; no kernel/model claim."""

from importlib.metadata import EntryPoint
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hcu_adapter.extensions import (
    ExtensionContext, ExtensionSpec, ExtensionStatus, RequiredExtensionError,
    activate_extensions,
)
from hcu_vllm_das import flash_activation as activation


POINT = EntryPoint(name="das_hcu_flash", value="hcu_vllm_das.flash_activation:extension_spec", group="hcu_adapter.extensions")


@pytest.fixture
def provider(monkeypatch):
    calls = []
    failure = {}

    def varlen():
        calls.append("vendor-callable-preflight")
        if failure.get("vendor"):
            raise ImportError("flash_attn public provider absent")
        return lambda **kwargs: None

    def load(name):
        calls.append(name)
        if name == "hcu_adapter.bridges.vllm":
            return SimpleNamespace(detect_hcu=lambda: SimpleNamespace(detected=not failure.get("hcu"), reason="test-only device evidence"))
        if name == "hcu_vllm_das.attention.flash_attn":
            if failure.get("seam"):
                raise RuntimeError("missing official attention seam")
            return SimpleNamespace(_vendor_varlen_func=varlen)
        raise AssertionError("unexpected dependency " + name)

    monkeypatch.setattr(activation, "import_module", load)
    return calls, failure


def context(version="0.32.0", required=False):
    return ExtensionContext("0.1.0", version, requested_capabilities={activation.CAPABILITY} if required else ())


def test_real_entry_point_factory_and_public_host_register_independent_owner(provider):
    calls, _ = provider
    result, = activate_extensions(context(required=True), (POINT,))
    assert result.status is ExtensionStatus.ACTIVE
    assert result.capabilities == frozenset({activation.CAPABILITY})
    assert calls == ["hcu_adapter.bridges.vllm", "hcu_vllm_das.attention.flash_attn", "vendor-callable-preflight"]


@pytest.mark.parametrize("failure", ["hcu", "vendor", "seam"])
@pytest.mark.parametrize("required", [False, True])
def test_public_host_reports_prepare_failure_and_required_refuses(provider, failure, required):
    _, flags = provider
    flags[failure] = True
    if required:
        with pytest.raises(RequiredExtensionError) as caught:
            activate_extensions(context(required=True), (POINT,))
        reports = caught.value.reports
    else:
        reports = activate_extensions(context(), (POINT,))
    report, = reports
    assert report.status is ExtensionStatus.FAILED
    assert report.capabilities == frozenset({activation.CAPABILITY})


@pytest.mark.parametrize("version", [None, "invalid", "1!0.32.0", "0.31.99", "0.33.0"])
def test_invalid_release_fails_before_vendor_or_framework_import(provider, version):
    calls, _ = provider
    with pytest.raises(activation.FlashActivationError, match="release line"):
        activation.activate(context(version))
    assert calls == []


@pytest.mark.parametrize("version", ["0.32.0rc1", "0.32.0+vendor", "0.32.1", "0.32.99.post1"])
def test_release_line_accepts_compatible_builds(provider, version):
    activation.activate(context(version))


def test_clean_legacy_failure_does_not_change_independent_flash_ownership(provider):
    class Legacy:
        name = "das_hcu"

        def load(self):
            def fail(_):
                raise RuntimeError("clean legacy prepare failure; no mutations")
            return lambda: ExtensionSpec("legacy", 1, 100, {"aiter-int8-moe"}, fail)

    reports = activate_extensions(context(required=True), (Legacy(), POINT))
    by_name = {r.entry_point_name: r for r in reports}
    assert by_name["das_hcu"].status is ExtensionStatus.FAILED
    assert by_name["das_hcu_flash"].status is ExtensionStatus.ACTIVE


def test_factory_imports_no_optional_runtime():
    plugin_root = Path(activation.__file__).resolve().parents[1]
    adapter_root = Path(__import__("hcu_adapter").__file__).resolve().parent.parent
    script = f'''
import sys
sys.path[:0] = {[str(plugin_root), str(adapter_root)]!r}
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {{'torch','vllm','flash_attn','aiter','lightop','boltops'}}:
   raise AssertionError('discovery imported optional runtime '+fullname)
sys.meta_path.insert(0,Block())
from hcu_vllm_das.flash_activation import extension_spec
assert extension_spec().capabilities == frozenset({{'hcu-flash-attention'}})
'''
    # -S removes sitecustomize and vendor/framework initialization. Only the
    # packaging distribution is added explicitly for this source-stage test.
    import packaging
    script = f"import sys; sys.path.append({str(Path(packaging.__file__).resolve().parent.parent)!r})\n" + script
    result = subprocess.run([sys.executable, "-S", "-B", "-c", script], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=""))
    assert result.returncode == 0, result.stdout + result.stderr
