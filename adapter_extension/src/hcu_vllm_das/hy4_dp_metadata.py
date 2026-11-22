# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional same-node transport for native HY4 MRV2 DP consensus.

Only the owned runner instance gets private function bindings. All seven native
fields, Graph dispatch and draft consensus reuse remain Core-owned.
"""
from types import CellType, FunctionType, MethodType
import weakref

from .adapter_api import adapter_module
from .environment import _flag, custom_ops_enabled
from .hy4_ihc_launch import _device_arch


def _supported_config(cfg):
    p = cfg.parallel_config
    return (cfg.model_config.hf_config.model_type == 'hy_v4'
        and not cfg.model_config.is_multimodal_model and cfg.use_v2_model_runner
        and 2 <= p.data_parallel_size <= 32
        and p.data_parallel_size_local == p.data_parallel_size
        and p.data_parallel_backend == 'mp' and p.nnodes == 1
        and not p.enable_elastic_ep
        and all(getattr(p, name) == 1 for name in ('tensor_parallel_size',
            'pipeline_parallel_size', 'prefill_context_parallel_size',
            'decode_context_parallel_size')) and _device_arch() == 'gfx938')


def enabled(cfg):
    master = custom_ops_enabled()
    child = _flag('HCU_VLLM_HY4_DP_METADATA_SHM', default=False)
    return master and child and _supported_config(cfg)


def _copy(function, namespace, closure):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, closure)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__dict__.update(function.__dict__)
    result.__module__, result.__qualname__ = function.__module__, function.__qualname__
    result.__doc__ = function.__doc__
    return result


def _bind_native(function, substitutions):
    """Preserve native bytecode and exact wrappers; only bind reviewed globals."""
    if not isinstance(function, FunctionType) or not substitutions:
        raise RuntimeError('HY4 DP native function anchor changed')
    if all(name in function.__code__.co_names and name in function.__globals__
           for name in substitutions):
        return _copy(function, {**function.__globals__, **substitutions}, function.__closure__)
    wrapped = getattr(function, '__wrapped__', None)
    cells = function.__closure__ or ()
    positions = [i for i, cell in enumerate(cells) if cell.cell_contents is wrapped]
    if isinstance(wrapped, FunctionType) and len(positions) == 1:
        bound = _bind_native(wrapped, substitutions)
        cells = list(cells)
        cells[positions[0]] = CellType(bound)
        result = _copy(function, function.__globals__, tuple(cells))
        result.__wrapped__ = bound
        return result
    raise RuntimeError('HY4 DP native function anchor changed')


def prepare(cfg):
    """Register before native weight loading; binding happens before profiling."""
    if enabled(cfg):
        from hcu_adapter.bridges.vllm_model_load import (
            register_post_model_load, require_post_model_load,
        )
        require_post_model_load(cfg)
        register_post_model_load('hy4-dp-metadata', on_model_loaded)


class _GroupProxy:
    def __init__(self, group, getter):
        self.group, self.getter, self.transport = group, getter, None

    def cpu_all_reduce_(self, tensor):
        if self.getter() is not self.group:
            raise RuntimeError('HY4 DP group changed; restart workers')
        return self.transport.all_reduce_(tensor)

    def __getattr__(self, name):
        return getattr(self.group, name)


def on_model_loaded(context):
    cfg, runner = context.vllm_config, context.worker.model_runner
    previous = getattr(runner, '_hcu_dp_metadata_binding', None)
    if previous is not None:
        if runner.execute_model.__func__ is not previous:
            raise RuntimeError('HY4 DP runner binding changed; restart workers')
        return
    if not enabled(cfg):
        return
    from vllm.v1.worker.gpu import model_runner, dp_utils
    if type(runner) is not model_runner.GPUModelRunner or 'execute_model' in runner.__dict__:
        raise RuntimeError('HY4 DP metadata requires the exact native MRV2 runner')
    group = dp_utils.get_dp_group()
    if (group.world_size != cfg.parallel_config.data_parallel_size
            or group.world_size != runner.dp_size or group.rank_in_group != runner.dp_rank):
        raise RuntimeError('HY4 DP metadata topology disagrees with the runner')
    proxy = _GroupProxy(group, dp_utils.get_dp_group)
    sync = _bind_native(dp_utils.sync_cudagraph_and_dp_padding,
                        {'get_dp_group': lambda: proxy})
    dispatch = _bind_native(dp_utils.dispatch_cg_and_sync_dp,
                            {'sync_cudagraph_and_dp_padding': sync})
    execute = _bind_native(type(runner).execute_model, {'dispatch_cg_and_sync_dp': dispatch})
    proxy.transport = adapter_module('cpu_metadata').create_transport(group, 7 * group.world_size)
    if proxy.transport is None:
        return  # Every rank acknowledged the same native fallback at bootstrap.
    runner.execute_model = MethodType(execute, runner)
    runner._hcu_dp_metadata_binding = execute
    weakref.finalize(runner, proxy.transport.close)
