"""Public INT8 publication/selection/cache identity; CPU metadata contracts.

Hardware discovery, host health and native file acquisition are controlled here.
The Core registry, Plugin publication, expert predicates, conversion and cache
factor are real. Device/provider acceptance is a separate marked HCU gate.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

TAG = 'HCU_INT8_ARCH_RESULT='


def _scenario_supported(architecture, drift="value"):
    import torch
    from hcu_adapter.extensions import ExtensionContext, ExtensionCommitError
    from hcu_adapter.bridges import vllm as host
    from hcu_vllm_das import int8_activation as owner
    from hcu_vllm_das.moe.int8_experts import HcuAiterInt8Experts
    from hcu_vllm_das.moe.int8_identity import FrozenFiles
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.oracle import int8 as oracle
    from vllm.model_executor.layers.fused_moe.oracle.int8_backend_registry import get_int8_moe_backend

    assert get_int8_moe_backend('aiter') is None
    config = FusedMoEConfig(num_experts=8, experts_per_token=2, hidden_dim=256,
        intermediate_size=128, num_local_experts=8, num_logical_experts=8,
        activation=MoEActivation.SILU, device='cuda', in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.Default, moe_backend='aiter',
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel())
    with tempfile.TemporaryDirectory() as folder:
        provider_file = Path(folder) / 'cpu-native-identity'
        provider_file.write_bytes(b'controlled identical provider content for both architectures')
        identity = FrozenFiles.capture({'native-fixture': provider_file})
        provider_architecture = [architecture]
        chip = ModuleType('aiter.jit.utils.chip_info')
        chip.get_gfx = lambda: provider_architecture[0]
        # Native acquisition would load JIT modules/configs. Only that external
        # acquisition is replaced; publication and its official consumers run.
        asm_dir = folder + '/'
        with patch.dict(os.environ, {'AITER_ASM_DIR': asm_dir}), \
             patch.dict(sys.modules, {'aiter': ModuleType('aiter'), 'aiter.moe': ModuleType('aiter.moe'), chip.__name__: chip}), \
             patch.object(host, 'detect_hcu', return_value=host.HcuDetection(True, architecture, '')) as detect, \
             patch.object(host, 'require_extension_healthy', return_value=None), \
             patch.object(torch.cuda, 'current_device', return_value=1), \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(gcnArchName=architecture)) as properties, \
             patch.object(owner, '_prepare_identity', return_value=(identity, ((chip, 'get_gfx', chip.get_gfx),), chip, os.fsencode(asm_dir))):
            context = ExtensionContext(adapter_version='0.5.22', vllm_version='0.33.0',
                requested_capabilities={'hcu-aiter-int8-moe'})
            owner.activate(context)
            registered = get_int8_moe_backend('aiter')
            assert registered is not None
            selected, cls = oracle.select_int8_moe_backend(config)
            assert selected is registered and cls is HcuAiterInt8Experts
            w13 = torch.zeros((8, 256, 256), dtype=torch.int8)
            w2 = torch.zeros((8, 256, 128), dtype=torch.int8)
            converted = oracle.convert_to_int8_moe_kernel_format(selected, w13, w2)
            assert converted[0] is w13 and converted[1] is w2
            factor = owner.compile_factor()
            assert factor.startswith('int8-asm-v1:' + architecture + ':'), factor
            # Cache identity belongs to the published architecture, rather than
            # a later mutable hardware reader value.
            detect.return_value = host.HcuDetection(True, 'gfx942', '')
            assert owner.compile_factor() == factor
            assert oracle.map_int8_backend('triton') is oracle.Int8MoeBackend.TRITON
            owner.check_health()
            detect.return_value = host.HcuDetection(True, architecture, '')
            if drift == 'mixed-worker':
                properties.return_value = SimpleNamespace(
                    gcnArchName='gfx938' if architecture == 'gfx936' else 'gfx936')
                with pytest.raises(ValueError):
                    oracle.select_int8_moe_backend(config)
                from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config
                quant = int8_w8a8_moe_quant_config(
                    torch.ones((8,256,1)), torch.ones((8,256,1)), None, None,
                    per_act_token_quant=True)
                with pytest.raises(ValueError, match='worker device'):
                    HcuAiterInt8Experts(config, quant)
                return dict(architecture=architecture, mixed_worker_refused=True)
            if drift == 'ep-scope':
                from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config
                parallel = FusedMoEParallelConfig(tp_size=1, pcp_size=1, dp_size=1,
                    ep_size=2, tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
                    sp_size=1, use_ep=True, all2all_backend='allgather_reducescatter',
                    enable_eplb=False)
                ep = FusedMoEConfig(num_experts=256, experts_per_token=8,
                    hidden_dim=2048, intermediate_size=512, num_local_experts=128,
                    num_logical_experts=256, activation=MoEActivation.SILU,
                    device='cuda', in_dtype=torch.bfloat16,
                    routing_method=RoutingMethodType.Default, moe_backend='aiter',
                    moe_parallel_config=parallel)
                quant = int8_w8a8_moe_quant_config(
                    torch.empty((128,1024,1),device='meta'),
                    torch.empty((128,2048,1),device='meta'),None,None,
                    per_act_token_quant=True)
                assert oracle.select_int8_moe_backend(ep)[1] is HcuAiterInt8Experts
                assert isinstance(HcuAiterInt8Experts(ep,quant),HcuAiterInt8Experts)
                return dict(architecture=architecture,tp1=selected.value,
                    ep2='admitted')
            if drift == 'asm-dir':
                # This changes what C getenv sees without changing os.environ.
                os.putenv('AITER_ASM_DIR', folder)
            elif drift == 'value':
                provider_architecture[0] = 'gfx936' if architecture == 'gfx938' else 'gfx938'
            else:
                replacement = ModuleType(chip.__name__)
                replacement.get_gfx = chip.get_gfx
                sys.modules[chip.__name__] = replacement
            with pytest.raises(ExtensionCommitError, match='unhealthy'):
                owner.check_health()
            with pytest.raises(ExtensionCommitError, match='unhealthy'):
                oracle.select_int8_moe_backend(config)
    return dict(architecture=architecture, selected=selected.value, expert=cls.__name__, factor=factor,
        provider_drift_refused=drift)


def _scenario_unsupported():
    from hcu_adapter.extensions import ExtensionContext
    from hcu_adapter.bridges import vllm as host
    from hcu_vllm_das import int8_activation as owner
    from vllm.model_executor.layers.fused_moe.oracle.int8_backend_registry import get_int8_moe_backend
    context = ExtensionContext(adapter_version='0.5.22', vllm_version='0.33.0',
        requested_capabilities={'hcu-aiter-int8-moe'})
    architectures = ['gfx928', 'gfx92a', 'gfx942', 'gfx938:unknown', None]
    for architecture in architectures:
        with patch.object(host, 'detect_hcu', return_value=host.HcuDetection(True, architecture, '')):
            with pytest.raises(RuntimeError):
                owner.activate(context)
        assert get_int8_moe_backend('aiter') is None
        with pytest.raises(RuntimeError, match='not active'):
            owner.check_health()
    return dict(refused=architectures)


def _run_scenario(name, *arguments):
    import hcu_vllm_das
    package_root = str(Path(hcu_vllm_das.__file__).resolve().parent.parent)
    env = dict(os.environ, HCU_ADAPTER_HOST='none', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
        VLLM_ROCM_USE_AITER='0', VLLM_ROCM_USE_AITER_MOE='0', VLLM_ROCM_USE_AITER_CUSTOM_AR='0')
    env.pop('HCU_AITER_AR', None)
    code = ('import sys,runpy,json;sys.path.insert(0,sys.argv[1]);'
            "f=runpy.run_path(sys.argv[2]);print(" + repr(TAG) + "+json.dumps(f[sys.argv[3]](*sys.argv[4:])))")
    result = subprocess.run([sys.executable, '-P', '-B', '-c', code, package_root,
        __file__, name, *arguments], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line[len(TAG):]) for line in result.stdout.splitlines() if line.startswith(TAG)]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize('architecture', ['gfx936', 'gfx938'])
def test_public_registration_selects_exact_hcu_architecture_and_binds_cache(architecture):
    row = _run_scenario('_scenario_supported', architecture)
    assert row['architecture'] == architecture and row['selected'] == 'aiter'
    assert row['expert'] == 'HcuAiterInt8Experts'


def test_unsupported_architecture_refusal_keeps_public_registry_empty():
    assert _run_scenario('_scenario_unsupported')['refused'] == [
        'gfx928', 'gfx92a', 'gfx942', 'gfx938:unknown', None]


def _scenario_mismatched_provider_architecture():
    from hcu_adapter.extensions import ExtensionContext, ExtensionCommitError
    from hcu_adapter.bridges import vllm as host
    from hcu_vllm_das import int8_activation as owner
    from vllm.model_executor.layers.fused_moe.oracle.int8_backend_registry import get_int8_moe_backend
    import importlib
    chip = ModuleType('aiter.jit.utils.chip_info')
    chip.get_gfx = lambda: 'gfx936'
    native, moe = ModuleType('aiter'), ModuleType('aiter.moe')
    def acquire(name):
        if name == 'aiter': return native
        if name == 'aiter.moe': return moe
        if name == 'aiter.jit.utils.chip_info': return chip
        if name == 'aiter.fused_moe_asm_wna16':
            raise AssertionError('native acquisition reached before architecture check')
        return importlib.import_module(name)
    context = ExtensionContext(adapter_version='0.5.22', vllm_version='0.33.0',
        requested_capabilities={'hcu-aiter-int8-moe'})
    with patch.object(host, 'detect_hcu', return_value=host.HcuDetection(True, 'gfx938', '')), \
         patch.object(owner, 'import_module', side_effect=acquire):
        with pytest.raises(ExtensionCommitError) as failure:
            owner.activate(context)
        cause = failure.value.__cause__
        assert isinstance(cause, RuntimeError) and 'plan architecture' in str(cause), repr(cause)
        assert 'gfx936' in str(cause) and 'gfx938' in str(cause)
        assert get_int8_moe_backend('aiter') is None
    return dict(physical='gfx938', provider='gfx936', refused_before_native=True)


def test_provider_plan_architecture_must_match_real_hcu_before_native_acquisition():
    row = _run_scenario('_scenario_mismatched_provider_architecture')
    assert row == dict(physical='gfx938', provider='gfx936', refused_before_native=True)


def test_replaced_provider_architecture_module_refuses_later_public_selection():
    assert _run_scenario('_scenario_supported', 'gfx938', 'module')['provider_drift_refused'] == 'module'


def test_native_loader_directory_drift_refuses_later_public_selection():
    assert _run_scenario('_scenario_supported', 'gfx936', 'asm-dir')['provider_drift_refused'] == 'asm-dir'


@pytest.mark.parametrize('architecture', ['gfx936', 'gfx938'])
def test_worker_architecture_must_match_the_published_native_provider(architecture):
    assert _run_scenario('_scenario_supported', architecture, 'mixed-worker')['mixed_worker_refused']


def test_asm_walk_keeps_loader_paths_and_rejects_directory_cycles(tmp_path):
    from hcu_vllm_das.int8_activation import _asm_objects

    root = tmp_path / 'objects'
    root.mkdir()
    target = tmp_path / 'objects-outside'
    target.mkdir()
    (target / 'kernel.co').write_bytes(b'fixture')
    alias = root / 'w8a8'
    alias.symlink_to(target, target_is_directory=True)
    assert list(_asm_objects(root)) == [alias / 'kernel.co']
    (target / 'cycle').symlink_to(root, target_is_directory=True)
    with pytest.raises(RuntimeError, match='directory cycle'):
        list(_asm_objects(root))


def _scenario_route_source_drift():
    """A changed integer route program must invalidate the real content factor."""
    import importlib
    from hcu_vllm_das import int8_activation as owner
    assert 'aiter.fused_moe_asm_wna16' not in sys.modules
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        def fixture(name):
            module=ModuleType(name);path=root/(name.replace('.','_')+'.fixture')
            path.write_text('CPU identity fixture; never loaded as native code')
            module.__file__=str(path);return module
        native=fixture('aiter');moe=fixture('aiter.moe')
        chip=fixture('aiter.jit.utils.chip_info');chip.get_gfx=lambda:'gfx936'
        asm=fixture('aiter.fused_moe_asm_wna16');jit=fixture('aiter.jit.core')
        route=fixture('hcu_vllm_das.moe.int8_routes')
        native.moe_sorting_fwd=lambda:None;native.asm_fmoe_a8=lambda:None
        moe.get_aiter_moe_config=lambda:None;asm.get_moe_asm_solution=lambda:None
        config=root/'channel.csv';config.write_text('CPU fixture,not a vendor config')
        asm.get_csv_path=lambda *_:str(config)
        asm.MoeQuantType=SimpleNamespace(INT8_W8A8_C=object())
        asm.load_and_cache_csv_for_quant=lambda *_:True
        codeobjects=root/'objects';codeobjects.mkdir()
        (codeobjects/'cpu.fixture.co').write_text('CPU identity fixture,not a GPU binary')
        jit.get_asm_dir=lambda:str(codeobjects)+'/'
        modules={name:fixture(name)for name in('module_cpp_api','module_moe_asm')}
        jit.get_module=lambda name:modules[name]
        external={m.__name__:m for m in(native,moe,chip,asm,jit,route)}
        def acquire(name):
            return external[name] if name in external else importlib.import_module(name)
        with patch.object(owner,'import_module',side_effect=acquire):
            with patch.dict(os.environ, {'AITER_ASM_DIR': str(codeobjects)}):
                with pytest.raises(RuntimeError, match='native loader directory'):
                    owner._prepare_identity(native,moe,'gfx936')
            with patch.dict(os.environ, {'AITER_ASM_DIR': jit.get_asm_dir()}):
                identity,_,_,_=owner._prepare_identity(native,moe,'gfx936')
            identity.validate()
            Path(route.__file__).write_text('Changed CPU route source fixture')
            with pytest.raises(RuntimeError,match='files changed'):
                identity.validate()
    return dict(route_source_drift_refused=True)


def test_route_program_source_changes_invalidate_prepared_content_identity():
    assert _run_scenario('_scenario_route_source_drift')['route_source_drift_refused']


@pytest.mark.parametrize('architecture,ep2', [('gfx936','admitted'),('gfx938','admitted')])
def test_public_selection_and_constructor_preserve_qualified_ep2_scope(architecture,ep2):
    assert _run_scenario('_scenario_supported',architecture,'ep-scope') == dict(
        architecture=architecture,tp1='aiter',ep2=ep2)
