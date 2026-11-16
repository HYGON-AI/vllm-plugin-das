# SPDX-License-Identifier: Apache-2.0
"""Transport correctness, fallback, and graph replay for Step5 dispatch packing."""

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.distributed.device_communicators.packed_all_gather import (
    packed_step5_all_gatherv,
)


def test_other_shapes_and_variable_sizes_do_not_enter_a_collective():
    def forbidden(*args, **kwargs):
        raise AssertionError("fallback must not enter the packed collective")

    comm = SimpleNamespace(world_size=4)
    inputs = [torch.zeros(32, 4096, dtype=torch.uint8),
              torch.zeros(32, 8), torch.zeros(32, 8, dtype=torch.int32),
              torch.ones(32, 1)]
    assert packed_step5_all_gatherv(comm, inputs, forbidden) is None
    inputs = [torch.zeros(64, 4096, dtype=torch.uint8),
              torch.zeros(64, 8), torch.zeros(64, 8, dtype=torch.int32),
              torch.ones(64, 1)]
    assert packed_step5_all_gatherv(comm, inputs, forbidden, sizes=[64, 32, 64, 64]) is None
    assert packed_step5_all_gatherv(comm, inputs, forbidden, dim=1) is None


def test_native_adapter_enables_packing_only_for_opted_in_dp_groups(monkeypatch):
    import vllm.config
    from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator as adapter

    class Communicator:
        def __init__(self, cpu_group, device=None, device_group=None, unique_name="",
                     global_ranks=None, global_world_size=None, tcp_store_group=None,
                     use_all2all=False):
            self.world_size = cpu_group
            self.use_all2all = use_all2all

        def all_gatherv(self, input_, dim=0, sizes=None):
            return input_

        def destroy(self):
            pass

    module = ModuleType(adapter.TARGET_MODULE)
    module.CudaCommunicator = Communicator
    monkeypatch.setattr(adapter, '_maybe_init_ep_custom_allreduce', lambda *args: None)
    monkeypatch.setattr(adapter, '_maybe_init_dp_ipc_all_gather', lambda obj, name: setattr(obj, '_hcu_ipc_ag', None))
    monkeypatch.setattr(vllm.config, 'get_current_vllm_config_or_none', lambda: None)
    monkeypatch.setattr(adapter.henvs, 'optional_custom_op_enabled', lambda requested: requested)
    assert adapter.apply_to_module(module)
    monkeypatch.delenv('VLLM_HCU_DP_PACKED_AG', raising=False)
    assert not Communicator(4, unique_name='dp:0')._hcu_dp_packed_ag
    monkeypatch.setenv('VLLM_HCU_DP_PACKED_AG', '1')
    assert not Communicator(4, unique_name='dp:0')._hcu_dp_packed_ag
    owner=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='step5')))
    monkeypatch.setattr(vllm.config, 'get_current_vllm_config_or_none', lambda: owner)
    monkeypatch.setattr(adapter, 'get_hcu_config', lambda _:SimpleNamespace(deepep_auto=False))
    assert Communicator(4, unique_name='dp:0')._hcu_dp_packed_ag
    assert not Communicator(4, unique_name='tp:0')._hcu_dp_packed_ag
    assert not Communicator(1, unique_name='dp:0')._hcu_dp_packed_ag
    monkeypatch.setattr(adapter.henvs, 'optional_custom_op_enabled', lambda requested: False)
    assert not Communicator(4, unique_name='dp:0')._hcu_dp_packed_ag


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_bytes_and_changed_input_graph_replay(id_dtype, strided):
    torch.manual_seed(6479)
    raw = torch.randint(0, 256, (64, 4096), device="cuda", dtype=torch.uint8)
    inputs = [raw.view(torch.float8_e4m3fn),
              torch.randn(64, 8, device="cuda"),
              torch.randint(0, 352, (64, 8), device="cuda", dtype=id_dtype),
              torch.rand(64, 1, device="cuda")]
    if strided:
        inputs = [torch.stack((t, t), dim=-1)[..., 0] for t in inputs]
        assert not inputs[0].is_contiguous()
    calls = []

    def gather(comm, tensor, dim=0, sizes=None):
        calls.append((tensor.dtype, dim, sizes))
        return tensor.repeat(comm.world_size)

    comm = SimpleNamespace(world_size=4)

    def check(outputs):
        for src, out in zip(inputs, outputs):
            expected = torch.cat([src.contiguous()] * 4, dim=0)
            assert src.dtype == out.dtype
            assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))

    check(packed_step5_all_gatherv(comm, inputs, gather))
    assert calls == [(torch.uint8, 0, None)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = packed_step5_all_gatherv(comm, inputs, gather)
    for value in [0, 1]:
        inputs[0].fill_(value)
        for output in outputs:
            output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        check(outputs)
