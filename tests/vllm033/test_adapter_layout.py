"""Filesystem imports exercise both layouts and preserve dependency failures."""
from pathlib import Path
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[2] / "adapter_extension/src/hcu_vllm_das/adapter_api.py"
if not HELPER.is_file():
    import hcu_vllm_das

    HELPER = Path(hcu_vllm_das.__file__).with_name("adapter_api.py")
NAMES = {
    "runtime_lifecycle": "infra.runtime_lifecycle",
    "hcu_env": "infra.hcu_env",
    "registry": "infra.registry",
    "hcu_config": "infra.hcu_config",
    "kill_switch": "infra.kill_switch",
    "dsv41_activation": "features.dsv41.dsv41_activation",
    "ct_w4a8_registration": "features.ct_w4a8.ct_w4a8_registration",
    "v41_fp8_indexer": "features.dsv41.v41_fp8_indexer",
    "dsv41_channel_fp8": "features.dsv41.dsv41_channel_fp8",
    "dsv41_shared_act_quant": "features.dsv41.shared_act_quant",
    "fp4_indexer": "features.dsv41.fp4_indexer",
    "v41_fp8_indexer_triton": "features.dsv41.v41_fp8_indexer_triton",
    "dsv41_dspark_registration": "features.dsv41.dsv41_dspark_registration",
    "dsv41_draft_weight_alias": "features.dsv41.dsv41_draft_weight_alias",
    "dsv41_int8_linear": "features.dsv41.dsv41_int8_linear",
    "ct_w4a8_masked_activation": "features.ct_w4a8.masked_activation",
    "channel_fp8_empty_linear": "features.ct_w4a8.channel_fp8_empty_linear",
    "ct_w4a8_packed_loader": "features.ct_w4a8.ct_w4a8_packed_loader",
}


def write_module(root, dotted, body=""):
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent / part
        parent.mkdir(exist_ok=True)
        (parent / "__init__.py").touch()
    (parent / (parts[-1] + ".py")).write_text(body)


def probe(root, body, *args):
    script = """
import importlib, importlib.util, pathlib, sys
root, helper, *arguments = sys.argv[1:]
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location('layout_helper', helper)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
assert 'hcu_adapter' not in sys.modules
""" + body + """
assert not any(name in sys.modules for name in ('torch', 'vllm', 'sglang'))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(root), str(HELPER), *args],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name,canonical", NAMES.items())
@pytest.mark.parametrize("layout", ("root", "canonical"))
def test_layout_import_preserves_real_module_and_mutable_state(tmp_path, name, canonical, layout):
    target = name if layout == "root" else canonical
    write_module(tmp_path, "hcu_adapter." + target, "state = []\n")
    if layout == "canonical":
        # Infrastructure presence selects the complete new layout, including
        # providers under features/. A legacy file must never be executed.
        (tmp_path / "hcu_adapter/infra").mkdir(exist_ok=True)
        (tmp_path / "hcu_adapter/infra/__init__.py").touch()
        write_module(tmp_path, "hcu_adapter." + name,
                     "raise AssertionError('legacy owner was imported')\n")
    probe(tmp_path, """
name, target, layout = arguments
owner = api.adapter_module(name)
assert owner is importlib.import_module('hcu_adapter.' + target)
owner.state.append(42)
assert api.adapter_module(name).state == [42]
assert owner.__name__ == 'hcu_adapter.' + target
assert pathlib.Path(owner.__file__).resolve().is_relative_to(pathlib.Path(root))
if layout == 'canonical':
    assert 'hcu_adapter.' + name not in sys.modules
""", name, target, layout)


@pytest.mark.parametrize("failure", ("missing-owner", "dependency", "infra-dependency"))
def test_new_layout_never_retries_a_legacy_owner_after_failure(tmp_path, failure):
    write_module(tmp_path, "hcu_adapter.runtime_lifecycle",
                 "raise AssertionError('legacy fallback hid new-layout failure')\n")
    infra = tmp_path / "hcu_adapter/infra"
    infra.mkdir()
    (infra / "__init__.py").write_text(
        "import layout_missing_dependency\n" if failure == "infra-dependency" else "")
    if failure == "dependency":
        (infra / "runtime_lifecycle.py").write_text("import layout_missing_dependency\n")
    expected = "hcu_adapter.infra.runtime_lifecycle" if failure == "missing-owner" else "layout_missing_dependency"
    probe(tmp_path, """
try:
    api.adapter_module('runtime_lifecycle')
except ModuleNotFoundError as error:
    assert error.name == arguments[0], (error.name, arguments[0])
else:
    raise AssertionError('broken new layout was silently accepted')
assert 'hcu_adapter.runtime_lifecycle' not in sys.modules
""", expected)


def test_unmoved_public_extensions_owner_keeps_its_module_identity(tmp_path):
    write_module(tmp_path, "hcu_adapter.extensions", "state = []\n")
    probe(tmp_path, """
owner = api.adapter_module('extensions')
assert owner is importlib.import_module('hcu_adapter.extensions')
assert 'hcu_adapter.infra' not in sys.modules
""")


@pytest.mark.parametrize("layout", ("root", "canonical", "broken-canonical"))
def test_rms_extension_discovery_uses_the_shared_lifecycle_owner(tmp_path, layout):
    """Import the real extension, including its lifecycle construction."""
    import packaging

    write_module(tmp_path, "hcu_adapter.extensions", """
class ExtensionContext:
    pass
class ExtensionSpec:
    def __init__(self, **values):
        self.__dict__.update(values)
""")
    lifecycle = """
class ProcessLocalLifecycle:
    def __init__(self, **values):
        self.options = values
"""
    if layout == "root":
        write_module(tmp_path, "hcu_adapter.runtime_lifecycle", lifecycle)
    else:
        (tmp_path / "hcu_adapter/infra").mkdir()
        (tmp_path / "hcu_adapter/infra/__init__.py").touch()
        write_module(tmp_path, "hcu_adapter.runtime_lifecycle",
                     "raise AssertionError('legacy lifecycle was imported')\n")
        if layout == "canonical":
            write_module(tmp_path, "hcu_adapter.infra.runtime_lifecycle", lifecycle)
    script = """
import importlib, importlib.abc, pathlib, sys
root, plugin_source, packaging_source, layout = sys.argv[1:]
sys.path[:0] = [root, plugin_source, packaging_source]
class NoFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','vllm','sglang','aiter','lightop','triton'}:
            raise AssertionError('framework imported during extension discovery: ' + fullname)
sys.meta_path.insert(0, NoFramework())
try:
    from hcu_vllm_das import extension
except ModuleNotFoundError as error:
    assert layout == 'broken-canonical', error
    assert error.name == 'hcu_adapter.infra.runtime_lifecycle', error.name
else:
    assert layout != 'broken-canonical'
    owner = importlib.import_module('hcu_adapter.' + (
        'runtime_lifecycle' if layout == 'root' else 'infra.runtime_lifecycle'))
    assert extension.ProcessLocalLifecycle is owner.ProcessLocalLifecycle
    assert type(extension._activation_lifecycle) is owner.ProcessLocalLifecycle
    spec = extension.extension_spec()
    assert spec.activate is extension.activate_extension
    assert spec.check_health is extension.check_health
    if layout == 'canonical':
        assert 'hcu_adapter.runtime_lifecycle' not in sys.modules
assert not any(n in sys.modules for n in ('torch','vllm','sglang','aiter','lightop','triton'))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(tmp_path),
         str(HELPER.parent.parent), str(Path(packaging.__file__).parent.parent), layout],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
