# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / 'adapter_extension/src/hcu_vllm_das/hy4_dp_metadata.py'


def owned():
    assert SOURCE.is_file(), 'HY4 metadata policy is not implemented'
    spec = importlib.util.spec_from_file_location('hcu_vllm_das.hy4_dp_metadata', SOURCE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def config():
    return NS(model_config=NS(hf_config=NS(model_type='hy_v4'), is_multimodal_model=False),
        use_v2_model_runner=True, parallel_config=NS(tensor_parallel_size=1,
        pipeline_parallel_size=1, prefill_context_parallel_size=1,
        decode_context_parallel_size=1, data_parallel_size=16,
        data_parallel_size_local=16, data_parallel_backend='mp', nnodes=1,
        enable_elastic_ep=False))


def test_default_off_and_master_rollback(monkeypatch):
    m = owned()
    monkeypatch.delenv('HCU_VLLM_HY4_DP_METADATA_SHM', raising=False)
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    assert not m.enabled(config())
    monkeypatch.setenv('HCU_VLLM_HY4_DP_METADATA_SHM', '1')
    monkeypatch.setattr(m, '_device_arch', lambda: 'gfx938')
    assert m.enabled(config())
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '0')
    assert not m.enabled(config())


@pytest.mark.parametrize('field,value', [('tensor_parallel_size',2),
    ('pipeline_parallel_size',2),('prefill_context_parallel_size',16),
    ('decode_context_parallel_size',2),('data_parallel_size',1),
    ('data_parallel_size',33),('data_parallel_size_local',8),
    ('data_parallel_backend','ray'),('nnodes',2),('enable_elastic_ep',True)])
def test_unsupported_topology_keeps_native(monkeypatch, field, value):
    m = owned()
    cfg = config()
    setattr(cfg.parallel_config, field, value)
    monkeypatch.setattr(m, '_device_arch', lambda: 'gfx938')
    assert not m._supported_config(cfg)


def native_dispatch(value):
    return value + 1


@torch.inference_mode()
def decorated_execute(self, value):
    assert torch.is_inference_mode_enabled()
    return native_dispatch(value)


def test_private_binding_preserves_actual_torch_decorator_and_globals():
    m = owned()
    replacement = lambda value: value + 7
    bound = m._bind_native(decorated_execute, {'native_dispatch': replacement})
    assert bound(None, 4) == 11 and decorated_execute(None, 4) == 5
    assert bound.__code__ is decorated_execute.__code__
    assert bound.__wrapped__.__code__ is decorated_execute.__wrapped__.__code__
    assert decorated_execute.__wrapped__.__globals__['native_dispatch'] is native_dispatch
    assert bound.__wrapped__.__globals__['native_dispatch'] is replacement
    assert not torch.is_inference_mode_enabled()


def test_missing_native_anchor_refuses_instead_of_silent_patch():
    m = owned()
    with pytest.raises(RuntimeError, match='anchor'):
        m._bind_native(decorated_execute, {'changed_upstream_name': lambda x: x})


def test_native_runner_and_dp_helper_globals_remain_unchanged():
    m = owned()
    from vllm.v1.worker.gpu import model_runner, dp_utils
    from vllm.distributed.parallel_state import get_dp_group
    original_execute = model_runner.GPUModelRunner.execute_model
    original_dispatch = dp_utils.dispatch_cg_and_sync_dp
    original_sync = dp_utils.sync_cudagraph_and_dp_padding
    sync = m._bind_native(original_sync, {'get_dp_group': lambda: object()})
    dispatch = m._bind_native(original_dispatch, {'sync_cudagraph_and_dp_padding': sync})
    execute = m._bind_native(original_execute, {'dispatch_cg_and_sync_dp': dispatch})
    assert execute.__code__ is original_execute.__code__
    assert execute.__wrapped__.__code__ is original_execute.__wrapped__.__code__
    assert dp_utils.dispatch_cg_and_sync_dp is original_dispatch
    assert dp_utils.sync_cudagraph_and_dp_padding is original_sync
    assert original_sync.__globals__['get_dp_group'] is get_dp_group
    assert original_execute.__wrapped__.__globals__['dispatch_cg_and_sync_dp'] is original_dispatch
