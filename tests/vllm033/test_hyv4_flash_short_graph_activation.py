"""Actual Adapter registry with explicit model/config/Core doubles; no GPU math."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


@pytest.mark.parametrize("drift", [False, True, "parent-health", "parent-match", "parent-create", "unselected-health"])
def test_selected_activation_owns_runner_publication(drift):
    root = Path(__file__).resolve().parents[2]
    adapter = Path(importlib.util.find_spec("hcu_adapter").origin).parent.parent
    bootstrap = "import sys;sys.path[:0]=" + repr([
        str(root / "adapter_extension/src"), str(adapter),
        sysconfig.get_paths()["purelib"],
    ]) + ";"
    code = r'''
import importlib, os, sys, types
from types import SimpleNamespace as NS
os.environ['HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES']='hyv4-flash-model'
def module(name):
    result=types.ModuleType(name);sys.modules[name]=result;return result
torch=module('torch')
class ModuleDouble: pass
torch.nn=NS(Module=ModuleDouble)
class ModelDouble(ModuleDouble): pass
config=module('hcu_vllm_das.hyv4_flash_config_activation')
config.activate=lambda context: None
config.check_health=lambda: None
class RegistryDouble:
    models={}
    @classmethod
    def register_model(cls, arch, target):
        mod,name=target.split(':')
        cls.models[arch]=NS(module_name=mod,class_name=name)
module('vllm.model_executor.models').ModelRegistry=RegistryDouble
base='hcu_vllm_das.models.hy_v4_flash.'
for leaf in ['model','attention','sparse_backend','swa','compressed_indexer',
             'indexpool_cache','engram','dfly','vl_model']:
    module(base+leaf)
sys.modules[base+'model'].HYV4FlashForCausalLM=ModelDouble
sys.modules[base+'vl_model'].HYV4FlashVLForConditionalGeneration=ModelDouble
compat=module(base+'dfly_compat')
compat.install_compat=lambda: None
compat.check_health=lambda: None
owner=importlib.import_module('hcu_vllm_das.hyv4_flash_model_activation')
context=NS(vllm_version='0.33.0',requested_extensions=set(),
           requested_capabilities={'hyv4-flash-model'})
owner.activate(context)
runner=importlib.import_module(base+'fullgraph_runner')
assert runner._registration is not None, 'selected model omitted runner publication'
token=runner._registration
owner.activate(context)
assert runner._registration is token
owner.check_health()
'''
    if isinstance(drift, str):
        code = code[:code.index("owner.activate(context)")]
        code += r'''
from hcu_adapter.extensions import ExtensionCommitError
import ast
def fail_register(cls, arch, target):
    raise RuntimeError('registrar failure after runner publication')
RegistryDouble.register_model=classmethod(fail_register)
try:owner.activate(context)
except ExtensionCommitError:pass
else:raise AssertionError('expected fatal parent publication failure')
runner=importlib.import_module(base+'fullgraph_runner')
token=runner._registration
assert token is not None, 'witness must cross runner-publication boundary'
source=__import__('pathlib').Path('tests/vllm033/test_hyv4_flash_short_graph_runner.py')
tree=ast.parse(source.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configuration')
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<configuration-double>','exec'))
cfg=configuration()
assert runner.is_qualified_config(cfg,breakable=False)
runtime=module(base+'runtime');runtime.is_hcu_platform=lambda:True
breakable=module('vllm.compilation.breakable_cudagraph')
breakable.is_breakable_cudagraph_enabled=lambda:False
operation={
    'parent-health':lambda:runner.check_health(),
    'parent-match':lambda:token.matches(cfg),
    'parent-create':lambda:token.create(cfg,object()),
}[CASE]
for _ in range(2):
    try:operation()
    except ExtensionCommitError:pass
    else:raise AssertionError('partially published runner remained healthy/eligible')
'''
        code = "CASE=" + repr(drift) + "\n" + code
        if drift == "unselected-health":
            original = "try:owner.activate(context)\nexcept ExtensionCommitError:pass\nelse:raise AssertionError('expected fatal parent publication failure')"
            replacement = "os.environ['HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES']=''\nowner.activate(NS(vllm_version='0.33.0',requested_extensions=set(),requested_capabilities=set()))\nimportlib.import_module(base+'fullgraph_runner').install()"
            assert code.count(original) == 1
            code = code.replace(original, replacement)
            code = code.replace("'parent-health':lambda:runner.check_health(),", "'parent-health':lambda:runner.check_health(),\n    'unselected-health':lambda:runner.check_health(),")
    elif drift:
        code += r'''
original=runner.create
runner.create=lambda *args: None
from hcu_adapter.extensions import ExtensionCommitError
for restore in [False,True]:
    if restore:runner.create=original
    try:owner.check_health()
    except ExtensionCommitError:pass
    else:raise AssertionError('runner drift was accepted or repaired')
'''
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("HCU_ADAPTER_VLLM_REQUIRED_")}
    env.update(HCU_ADAPTER_HOST="none", VLLM_PLUGINS="none",
               HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", bootstrap + code],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
