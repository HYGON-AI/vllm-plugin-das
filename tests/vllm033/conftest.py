"""Opt-in RECORD ownership checks for installed CI; no provider computation."""
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import sys



_OWNERS = {}

# These installed facades deliberately expose the Adapter module itself so
# historical attribute writes still reach its state. Keep this list exact:
# another Plugin name or another Adapter module is not a compatibility alias.
_ADAPTER_ALIASES = {
    'hcu_vllm_das.dsv41_activation': 'hcu_adapter.dsv41_activation',
    'hcu_vllm_das.ct_w4a8_activation': 'hcu_adapter.ct_w4a8_registration',
    'hcu_vllm_das.dsv41_dspark_registration': 'hcu_adapter.dsv41_dspark_registration',
    'hcu_vllm_das.pcp_dspark': 'hcu_adapter.kernels.pcp_dspark',
    'hcu_vllm_das.quantization.dsv41_channel_fp8': 'hcu_adapter.dsv41_channel_fp8',
    'hcu_vllm_das.quantization.dsv41_int8_linear': 'hcu_adapter.dsv41_int8_linear',
}

# The same public owners have these exact canonical locations after Adapter's
# directory migration. No other Plugin facade or Adapter target is admitted.
_RELOCATED_ADAPTER_ALIASES = {
    'hcu_adapter.dsv41_activation':
        'hcu_adapter.features.dsv41.dsv41_activation',
    'hcu_adapter.ct_w4a8_registration':
        'hcu_adapter.features.ct_w4a8.ct_w4a8_registration',
    'hcu_adapter.dsv41_dspark_registration':
        'hcu_adapter.features.dsv41.dsv41_dspark_registration',
    'hcu_adapter.dsv41_channel_fp8':
        'hcu_adapter.features.dsv41.dsv41_channel_fp8',
    'hcu_adapter.dsv41_int8_linear':
        'hcu_adapter.features.dsv41.dsv41_int8_linear',
}


def _owned_adapter_alias(name, module, modules):
    target = _ADAPTER_ALIASES.get(name)
    relocated = _RELOCATED_ADAPTER_ALIASES.get(target)
    if relocated is not None and getattr(module, '__name__', None) == relocated:
        target = relocated
    if target is None or module is not modules.get(target):
        return False
    if getattr(module, '__name__', None) != target:
        return False
    plugin = modules.get('hcu_vllm_das')
    adapter = modules.get('hcu_adapter')
    if not getattr(plugin, '__file__', None) or not getattr(adapter, '__file__', None):
        return False
    facade = Path(plugin.__file__).resolve().parent.joinpath(*name.split('.')[1:]).with_suffix('.py')
    destination = Path(adapter.__file__).resolve().parent.joinpath(*target.split('.')[1:]).with_suffix('.py')
    return (facade in _OWNERS.get('hcu_vllm_das', ())
            and destination in _OWNERS.get('hcu_adapter', ())
            and bool(getattr(module, '__file__', None))
            and Path(module.__file__).resolve() == destination)


def _verify_installed_modules(modules=None):
    modules = sys.modules if modules is None else modules
    for prefix, owned in _OWNERS.items():
        if prefix not in modules:
            raise RuntimeError(prefix + ' is not owned by the installed distribution')
        for name, module in tuple(modules.items()):
            if name != prefix and not name.startswith(prefix + '.'):
                continue
            origin = getattr(module, '__file__', None)
            if name in _ADAPTER_ALIASES:
                if _owned_adapter_alias(name, module, modules):
                    continue
                raise RuntimeError(name + ' is not owned by the installed distribution: ' + str(origin))
            if origin is None and name != prefix:
                continue  # Some Core dispatcher/namespace modules have no file.
            if origin and Path(origin).resolve() in owned:
                continue
            raise RuntimeError(name + ' is not owned by the installed distribution: ' + str(origin))


def pytest_sessionstart(session):
    _OWNERS.clear()
    if os.environ.get('HCU_V033_VERIFY_INSTALLED') != '1':
        return
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('_hcu_plugin_installed_guard_owner',
        Path(__file__).resolve().parents[2]/'adapter_extension/tools/installed_adapter_owner.py')
    helper=module_from_spec(spec)
    spec.loader.exec_module(helper)
    adapter_name=helper.adapter_distribution_name()
    for distribution, prefix in [('vllm', 'vllm'), (adapter_name, 'hcu_adapter'),
                                 ('hcu-vllm-das-plugin', 'hcu_vllm_das')]:
        import_module(prefix)  # Pin package __path__ before test-module imports.
        installed = metadata.distribution(distribution)
        if not installed.files or not installed.read_text('RECORD'):
            raise RuntimeError(distribution + ' has no installed RECORD')
        direct_url = installed.read_text('direct_url.json')
        if direct_url and json.loads(direct_url).get('dir_info', {}).get('editable'):
            raise RuntimeError(distribution + ' must not be editable in installed CI')
        # Snapshot the installed manifest before a unit fixture changes sys.path.
        _OWNERS[prefix] = frozenset(Path(installed.locate_file(item)).resolve()
                                   for item in installed.files)
    _verify_installed_modules()


def pytest_collection_finish(session):
    _verify_installed_modules()


def pytest_sessionfinish(session, exitstatus):
    _verify_installed_modules()


def shared_fp8_selected_ops(monkeypatch):
    from hcu_adapter.bridges.vllm_input_quant_lightop import _wrap
    from vllm.model_executor.layers.quantization import input_quant_fp8 as module

    original = module.QuantFP8.forward_cuda
    original = getattr(original, "__wrapped__", original)
    monkeypatch.setattr(module.QuantFP8, "forward_cuda", _wrap(module, original))

    def construct(custom_ops):
        from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.activation import SiluAndMulWithClamp
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

        with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(
                mode=0, custom_ops=custom_ops))):
            quant = module.QuantFP8(False, GroupShape.PER_TOKEN, compile_native=False)
            act = SiluAndMulWithClamp(10.0, compile_native=False)
        return quant, act

    return construct



# The installed-origin probes import this file with python -S. Register the
# device-test fixture only when pytest is already running.
if "pytest" in sys.modules:
    shared_fp8_selected_ops = sys.modules["pytest"].fixture(shared_fp8_selected_ops)
