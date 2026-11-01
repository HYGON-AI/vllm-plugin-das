"""Actual a198 public registry/ROCm selector with explicit CPU-only fixtures.

Not collected as HCU/numerical support. Run this file as a script with one case
in a clean process, with the independently installed Adapter/Plugin/target.
No selector function, registry table or configuration validation is replaced.
"""
import hashlib
import inspect
import json
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch


def run_case(case):
    import torch
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('FORBIDDEN_GPU_OR_PROVIDER')
        raise AssertionError('CPU-only public selector test')
    with patch.object(torch.cuda, 'is_available', lambda: False), \
         patch.object(torch.cuda, 'device_count', lambda: 0), \
         patch.object(torch.cuda, 'current_device', forbidden), \
         patch.object(torch.cuda, 'get_device_properties', forbidden), \
         patch.object(torch.cuda, '_lazy_init', forbidden):
        from importlib.metadata import distribution
        from hcu_adapter.bridges import vllm as bridge
        from hcu_adapter.extensions import (activate_extensions, ExtensionContext, ExtensionSpec,
            ExtensionStatus, ExtensionCommitError, RequiredExtensionError)
        from hcu_vllm_das import flash_activation as owner
        from hcu_vllm_das.attention import flash_attn as backend
        from vllm.platforms.rocm import RocmPlatform
        from vllm.platforms.interface import DeviceCapability
        from vllm.v1.attention.backends import registry
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        from vllm.v1.attention.selector import AttentionSelectorConfig
        import vllm.config
        enum = registry.AttentionBackendEnum.FLASH_ATTN
        point = next(p for p in distribution('hcu-vllm-das-plugin').entry_points
                     if p.name == 'das_hcu_flash')
        platform = bridge.HcuPlatform
        cfg = AttentionSelectorConfig(128, torch.bfloat16, 'auto', 64)
        assert not enum.is_overridden(), 'fresh process registry required'
        # Early backend import and config unpickle must not publish an owner.
        cfg = pickle.loads(pickle.dumps(cfg))
        assert not enum.is_overridden() and not bridge.extension_reports()
        foreign_calls = []
        class ForeignBackend(FlashAttentionBackend):
            @classmethod
            def validate_configuration(cls, *args, **kwargs):
                foreign_calls.append('foreign-validation')
                return super().validate_configuration(*args, **kwargs)
        ForeignBackend.__module__ = __name__
        ForeignBackend.__qualname__ = 'ForeignBackend'
        globals()['ForeignBackend'] = ForeignBackend
        foreign_path = __name__ + '.ForeignBackend'
        def missing_vendor():
            raise ImportError('CPU fault injection: vendor unavailable before publication')
        vendor_resolver = missing_vendor if case in ('optional', 'required') else lambda: forbidden
        context = ExtensionContext('0.1.0', '0.32.0',
            requested_capabilities={owner.CAPABILITY} if case == 'required' else ())
        points = [point]
        actions = bridge._RUNTIME_ACTIONS
        detail = {}
        # Synthetic capability is only input to real official config validation.
        # This is not an architecture/provider support assertion or detection.
        with patch.object(platform, 'get_device_capability', lambda: DeviceCapability(9, 0)), \
             patch.object(vllm.config, 'get_current_vllm_config_or_none', lambda: None), \
             patch.object(backend, '_vendor_varlen_func', vendor_resolver):
            if case == 'foreign-before':
                registry.register_backend(enum, foreign_path)
            if case == 'duplicate':
                from types import SimpleNamespace
                points.append(SimpleNamespace(name='another-owner', load=lambda: lambda:
                    ExtensionSpec('another-owner', 1, 1, {owner.CAPABILITY}, lambda _: None)))
            def activate():
                reports = activate_extensions(context, points,
                    health_check_sink=actions.record_extension_health)
                actions.record_extension_reports(reports)
                return reports
            if case == 'required':
                try:
                    activate()
                except RequiredExtensionError as error:
                    assert len(error.reports) == 1 and error.reports[0].status is ExtensionStatus.FAILED
                else:
                    raise AssertionError('required prepare failure was swallowed')
                assert not enum.is_overridden() and not owner._published
            elif case == 'commit':
                original = registry.register_backend
                def fail_after_register(*args, **kwargs):
                    original(*args, **kwargs)
                    raise ValueError('CPU fault injection after actual public registration')
                with patch.object(registry, 'register_backend', fail_after_register):
                    try:
                        activate()
                    except ExtensionCommitError as error:
                        assert len(error.reports) == 1 and error.reports[0].status is ExtensionStatus.FAILED
                    else:
                        raise AssertionError('post-publication failure was optional')
                    assert enum.is_overridden()
                    try:
                        owner.activate(context)
                    except ExtensionCommitError:
                        pass
                    else:
                        raise AssertionError('published failure was cleared')
            else:
                reports = activate()
                if case in ('optional', 'foreign-before', 'duplicate'):
                    assert all(r.status is ExtensionStatus.FAILED for r in reports)
                    assert enum.get_path() == foreign_path if case == 'foreign-before' else not enum.is_overridden()
                    try:
                        platform.get_attn_backend_cls(enum, cfg, 8)
                    except bridge.LatchedVllmRuntimeError:
                        pass
                    else:
                        raise AssertionError('unowned Flash was selected')
                else:
                    assert len(reports) == 1 and reports[0].status is ExtensionStatus.ACTIVE
                    assert platform.get_attn_backend_cls(enum, cfg, 8) == owner.BACKEND
                    if case in ('clear', 'foreign'):
                        if case == 'clear':
                            enum.clear_override()
                        else:
                            registry.register_backend(enum, foreign_path)
                        try:
                            platform.get_attn_backend_cls(enum, cfg, 8)
                        except bridge.LatchedVllmRuntimeError as error:
                            detail['first_rejection'] = str(error)
                        else:
                            raise AssertionError('actual official selector bypassed health checker')
                        assert not foreign_calls
                        # A caught error/report replacement cannot authorize any
                        # later default/Triton/MLA/Sparse selection or activation.
                        actions.record_extension_reports(reports)
                        blocked = []
                        for selected, config in ((enum, cfg), (None, cfg),
                            (registry.AttentionBackendEnum.TRITON_ATTN, cfg),
                            (None, cfg._replace(use_mla=True)),
                            (None, cfg._replace(use_mla=True, use_sparse=True))):
                            try:
                                platform.get_attn_backend_cls(selected, config, 8)
                            except bridge.LatchedVllmRuntimeError:
                                blocked.append(str(selected))
                            else:
                                raise AssertionError('caught health failure bypassed host latch')
                        try:
                            bridge.hcu_runtime_plugin()
                        except bridge.LatchedVllmRuntimeError:
                            pass
                        else:
                            raise AssertionError('runtime ignored caught health failure')
                        detail['later_routes_blocked'] = len(blocked)
                    elif case == 'config':
                        rejected = []
                        for field, value in (('head_size', 120), ('block_size', 17), ('use_dcp', True),
                                             ('use_mla', True), ('use_sparse', True), ('has_sink', True)):
                            try:
                                platform.get_attn_backend_cls(enum, cfg._replace(**{field: value}), 8)
                            except ValueError:
                                rejected.append(field)
                            else:
                                raise AssertionError('official config validation bypassed: ' + field)
                        detail['official_configuration_rejections'] = rejected
                        bridge.require_runtime_healthy()  # config rejection is not owner drift
                    elif case == 'fork':
                        pid = os.fork()
                        if pid == 0:
                            try:
                                assert not bridge.extension_reports() and not actions._health_bindings
                                try:
                                    owner.check_health()
                                except owner.FlashActivationError:
                                    pass
                                else:
                                    os._exit(4)
                                activate()
                                assert platform.get_attn_backend_cls(enum, cfg, 8) == owner.BACKEND
                            except BaseException:
                                os._exit(2)
                            os._exit(0)
                        _, status = os.waitpid(pid, 0)
                        assert os.waitstatus_to_exitcode(status) == 0
                        assert bridge.extension_reports()[0] is reports[0]
                        detail['actual_fork_child_explicit_activation'] = 'PASS'
                    elif case != 'normal':
                        raise ValueError(case)
        def identity(obj):
            path = Path(inspect.getsourcefile(obj))
            return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        forbidden_imports = [name for name in sys.modules if
            name.startswith(('hcu_vllm_das.w4a8', 'hcu_vllm_das.activation', 'hcu_adapter.model_load'))]
        assert not forbidden_imports, forbidden_imports
        assert not calls and not torch.cuda.is_initialized()
        result = dict(case=case, status='PASS', detail=detail, compute_calls=0, GPU_initialized=False,
            target_selector=identity(RocmPlatform.get_attn_backend_cls), adapter=identity(bridge),
            plugin=identity(owner), forbidden_independent_flash_imports=forbidden_imports,
            scope='actual installed/source public selector CPU ABI only; no HCU/numerical claim')
        print('PUBLIC_SELECTOR=' + json.dumps(result), flush=True)
        return result


def spawn_child():
    run_case('normal')


def main():
    if len(sys.argv) != 2:
        raise SystemExit('one isolated case required')
    if sys.argv[1] == 'spawn':
        import multiprocessing
        child = multiprocessing.get_context('spawn').Process(target=spawn_child)
        child.start()
        child.join(120)
        assert not child.is_alive() and child.exitcode == 0
        print('SPAWN=PASS', flush=True)
    else:
        run_case(sys.argv[1])


if __name__ == '__main__':
    main()
