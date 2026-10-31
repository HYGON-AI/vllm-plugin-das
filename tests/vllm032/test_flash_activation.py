"""Independent Flash lifecycle CPU contracts; no kernel/model claim."""

from importlib.metadata import EntryPoint
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace
from threading import RLock

import pytest

from hcu_adapter.extensions import (
    ExtensionCommitError, ExtensionContext, ExtensionSpec, ExtensionStatus, RequiredExtensionError,
    activate_extensions,
)
from hcu_vllm_das import flash_activation as activation


POINT = EntryPoint(name="das_hcu_flash", value="hcu_vllm_das.flash_activation:extension_spec", group="hcu_adapter.extensions")


@pytest.fixture
def provider(monkeypatch):
    calls = []
    failure = {}

    vendor = lambda **kwargs: None
    class Backend:
        validate_configuration = classmethod(lambda cls, **kwargs: [])

    class Enum:
        path = None
        def is_overridden(self):return self.path is not None
        def get_path(self):return self.path or "official.FlashBackend"
        def get_class(self):return Backend if self.path == activation.BACKEND else object
    enum=Enum()
    def register(member,path):
        calls.append("publish")
        member.path=path
        if failure.get("commit"):raise RuntimeError("post-publication synthetic failure")
    registry=SimpleNamespace(AttentionBackendEnum=SimpleNamespace(FLASH_ATTN=enum),register_backend=register)
    monkeypatch.setattr(activation,"_published",None)
    monkeypatch.setattr(activation,"_failure",None)
    monkeypatch.setattr(activation,"_publishing",False)
    monkeypatch.setattr(activation,"_lock",RLock())
    failure.update(enum=enum,registry=registry,backend=Backend)

    def varlen():
        calls.append("vendor-callable-preflight")
        if failure.get("vendor"):
            raise ImportError("flash_attn public provider absent")
        return vendor

    def load(name):
        calls.append(name)
        if name == activation.REGISTRY:return registry
        if name == "hcu_vllm_das.attention.flash_attn":
            if failure.get("seam"):
                raise RuntimeError("missing official attention seam")
            return SimpleNamespace(_vendor_varlen_func=varlen,HcuFlashAttentionBackend=Backend)
        if name.endswith("e5m2_cache"):
            if failure.get("dependency"):raise ImportError("missing writer dependency")
            return SimpleNamespace(reshape_and_cache_e5m2=vendor)
        if name.endswith("e5m2_scales"):return SimpleNamespace(run_e5m2_attention=vendor)
        raise AssertionError("unexpected dependency " + name)

    monkeypatch.setattr(activation, "import_module", load)
    return calls, failure


def context(version="0.32.0", required=False):
    return ExtensionContext("0.1.0", version, requested_capabilities={activation.CAPABILITY} if required else ())


def test_real_entry_point_factory_and_public_host_register_independent_owner(provider):
    calls, _ = provider
    assert activation.extension_spec().check_health is activation.check_health
    result, = activate_extensions(context(required=True), (POINT,))
    assert result.status is ExtensionStatus.ACTIVE
    assert result.capabilities == frozenset({activation.CAPABILITY})
    assert calls[-1] == "publish"
    assert "hcu_adapter.bridges.vllm" not in calls


@pytest.mark.parametrize("failure", ["dependency", "vendor", "seam"])
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


def test_foreign_override_is_clean_optional_and_required_failure(provider):
    calls,flags=provider
    flags['enum'].path='foreign.Backend'
    report,=activate_extensions(context(),(POINT,))
    assert report.status is ExtensionStatus.FAILED
    assert flags['enum'].path=='foreign.Backend' and 'publish' not in calls
    assert activation._failure is None
    with pytest.raises(RequiredExtensionError):activate_extensions(context(required=True),(POINT,))


def test_idempotent_exact_publication_uses_public_getters_without_republishing(provider):
    calls,flags=provider
    activation.activate(context());activation.activate(context())
    activation.require_published(flags['backend'])
    assert calls.count('publish')==1


@pytest.mark.parametrize('drift',['path','class','vendor','version'])
def test_published_drift_is_fatal_sticky_and_never_repairs(provider,drift):
    calls,flags=provider;activation.activate(context())
    if drift=='path':flags['enum'].path='foreign.Backend'
    if drift=='class':flags['enum'].get_class=lambda:object
    if drift=='vendor':flags['vendor']=True
    if drift=='vendor':
        original=activation.import_module
        activation.import_module=lambda name:SimpleNamespace(_vendor_varlen_func=lambda:object()) if name.endswith('.flash_attn') else original(name)
    with pytest.raises(ExtensionCommitError):activation.activate(context('0.33.0' if drift=='version' else '0.32.0'))
    with pytest.raises(ExtensionCommitError):activation.activate(context())
    assert calls.count('publish')==1


def test_commit_error_never_becomes_optional_or_runs_later_owner(provider):
    calls,flags=provider;flags['commit']=True
    with pytest.raises(ExtensionCommitError) as caught:activate_extensions(context(),(POINT,))
    assert caught.value.reports[0].status is ExtensionStatus.FAILED
    flags['commit']=False
    with pytest.raises(ExtensionCommitError):activation.activate(context())
    assert calls.count('publish')==1


@pytest.mark.parametrize('phase', ['commit', 'health', 'registry', 'state'])
@pytest.mark.parametrize('format_error', [RuntimeError, KeyboardInterrupt])
def test_postpublication_bad_string_is_typed_fatal_before_cleanup_and_cannot_reactivate(provider, phase, format_error):
    calls, flags = provider
    observed = []
    class BadString(BaseException):
        def __str__(self):
            observed.append(activation._failure)
            raise format_error('formatting failed')
    if phase == 'commit':
        original = flags['registry'].register_backend
        def fail(*args):
            original(*args)
            raise BadString()
        flags['registry'].register_backend = fail
        action = lambda: activation.activate(context())
    else:
        activation.activate(context())
        original = activation.import_module
        def fail(name):
            if name == activation.REGISTRY:
                raise BadString()
            return original(name)
        activation.import_module = fail
        action = activation.check_health if phase == 'health' else lambda: activation.activate(context())
        if phase == 'state':
            activation.import_module = original
            flags['enum'].get_path = lambda: (_ for _ in ()).throw(BadString())
    with pytest.raises(ExtensionCommitError) as caught:
        action()
    assert isinstance(caught.value.__cause__, BadString)
    assert activation._failure and all(observed)
    assert not activation._publishing
    first_failure = activation._failure
    with pytest.raises(ExtensionCommitError):
        activation.activate(context())
    assert activation._failure == first_failure and calls.count('publish') == 1


def test_duplicate_capability_reports_no_publication(provider):
    calls,_=provider
    class Duplicate:
        name='other_flash'
        def load(self):return lambda:ExtensionSpec('other-flash',1,1,{activation.CAPABILITY},lambda _:None)
    reports=activate_extensions(context(),(POINT,Duplicate()))
    assert len(reports)==2 and all(r.status is ExtensionStatus.FAILED for r in reports)
    assert 'publish' not in calls


@pytest.mark.skipif(not hasattr(os,'fork'),reason='requires actual CPU fork')
def test_fork_requires_explicit_current_process_activation(provider):
    _,flags=provider;activation.activate(context())
    pid=os.fork()
    if pid==0:
        try:
            try:activation.require_published(flags['backend'])
            except activation.FlashActivationError:pass
            else:os._exit(3)
            activation.activate(context());activation.require_published(flags['backend'])
        except BaseException:os._exit(2)
        os._exit(0)
    _,status=os.waitpid(pid,0);assert os.waitstatus_to_exitcode(status)==0


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
