"""Real vLLM EPLB transfer on CPU/Gloo, with Kimi packed weights and scales.

This proves checkpoint-layout transport, NOT AITER/HIPC execution, router
statistics, online map commit, or CUDA-graph safety after a rebalance.
"""
import contextlib
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.models.kimi_k3.test_kimi_moe_metadata import WEIGHTS, make_runner
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator, TorchDistNcclEplbCommunicator,
)
from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
from vllm_hcu.model_executor.layers.quantization.kimi_k3_w4a8 import (
    KimiK3W4A8MoEMethod, pack_int4_twos_complement,
)
from vllm_hcu.models.kimi_k3.amd.linear import KimiLinearForCausalLM, KimiMoE


def _decode(packed, shape):
    # Independent high/low signed-nibble reference, not production unpack().
    byte = packed.view(torch.uint8).reshape(shape[:-1] + (shape[-1] // 2,)).int()
    values = torch.stack((byte // 16, byte % 16), -1).reshape(shape)
    return (values - (values >= 8).int() * 16).float()


def _forward(w13, w2, s13, s2, x):
    projected = x @ (w13 * s13 * 16).T
    gate, up = projected.chunk(2, dim=-1)
    activated = 1.5 * torch.tanh(gate / 1.5) * torch.sigmoid(gate)
    activated *= 2. * torch.tanh(up / 2.)
    return activated @ (w2 * s2 * 16).T


def _worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method=rendezvous, timeout=timedelta(seconds=60))
    try:
        old = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1], [5, 4, 3, 2, 1, 0, 5, 4]])
        new = torch.tensor([[5, 5, 2, 1, 3, 0, 4, 4], [0, 0, 3, 4, 2, 5, 1, 1]])
        local = 8 // world
        model = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=6, num_expert_group=1, num_shared_experts=1)
        model.model = torch.nn.Module()
        model.model.start_layer, model.model.end_layer = 0, 2
        model.model.layers = torch.nn.ModuleList()
        references = []
        for layer_index in range(2):
            torch.manual_seed(41 + layer_index)
            runner = make_runner(logical=6, physical=8, local=local)
            owner = runner.routed_experts
            for name in WEIGHTS:
                delattr(owner, name)
            method = KimiK3W4A8MoEMethod(SimpleNamespace(
                activation=SimpleNamespace(value="situ"), swiglu_beta=None,
                moe_parallel_config=SimpleNamespace(use_deepep_ht_kernels=False,
                                                   use_deepep_ll_kernels=False)))
            method.create_weights(owner, local, 32, 32, torch.bfloat16)
            raw13 = torch.randint(-8, 8, (6, 64, 32), dtype=torch.int8)
            raw2 = torch.randint(-8, 8, (6, 32, 32), dtype=torch.int8)
            s13, s2 = torch.rand(6, 64, 1) * .04 + .01, torch.rand(6, 32, 1) * .04 + .01
            refs = (raw13, raw2, s13, s2)
            references.append(refs)
            ids = old[layer_index, rank * local:(rank + 1) * local]
            for name, value in zip(WEIGHTS, (pack_int4_twos_complement(raw13).view(torch.int8),
                                            pack_int4_twos_complement(raw2).view(torch.int8), s13, s2)):
                getattr(owner, name).data.copy_(value[ids])
            layer = torch.nn.Module()
            layer.mlp = KimiMoE.__new__(KimiMoE)
            torch.nn.Module.__init__(layer.mlp)
            layer.mlp.experts = runner
            model.model.layers.append(layer)
        model._init_moe_metadata()
        model.set_eplb_state(torch.zeros(2, 8), torch.zeros(2, 6, 2, dtype=torch.int64),
                             torch.ones(2, 6, dtype=torch.int32))
        buffers = [torch.empty_like(w) for w in model.expert_weights[0]]
        addresses = [[w.data_ptr() for w in row] for row in model.expert_weights]
        biases = [runner.routed_experts.e_score_correction_bias.clone() for runner in model.moe_layers]
        # Only stream context and the PP-dependent log hook are replaced. The
        # upstream send/receive scheduling, torch P2P calls and all copies run.
        with patch.object(torch.cuda, "stream", lambda stream: contextlib.nullcontext()), \
                patch.object(EplbCommunicator, "_log_initialized", lambda self: None):
            communicator = TorchDistNcclEplbCommunicator(dist.group.WORLD)
            for before, after in ((old, new), (new, old), (old, old)):
                rearrange_expert_weights_inplace(before, after, model.expert_weights,
                    buffers, dist.group.WORLD, communicator)
                for layer_index, row in enumerate(model.expert_weights):
                    assert [w.data_ptr() for w in row] == addresses[layer_index]
                    expected_ids = after[layer_index, rank * local:(rank + 1) * local]
                    w13, w2 = _decode(row[0], (local, 64, 32)), _decode(row[1], (local, 32, 32))
                    s13, s2 = row[2].reshape(local, 64, 1), row[3].reshape(local, 32, 1)
                    ref13, ref2, refs13, refs2 = references[layer_index]
                    for actual, expected in zip((w13, w2, s13, s2), references[layer_index]):
                        torch.testing.assert_close(actual, expected[expected_ids].to(actual.dtype), atol=0, rtol=0)
                    x = torch.arange(96, dtype=torch.float32).reshape(3, 32) / 100
                    for slot, logical in enumerate(expected_ids.tolist()):
                        actual = _forward(w13[slot], w2[slot], s13[slot], s2[slot], x)
                        expected = _forward(ref13[logical].float(), ref2[logical].float(),
                                            refs13[logical], refs2[logical], x)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(model.moe_layers[layer_index].routed_experts.e_score_correction_bias,
                                               biases[layer_index], atol=0, rtol=0)
                dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2])
def test_packed_expert_transfer_preserves_values_scales_and_addresses(tmp_path, world):
    mp.spawn(_worker, args=(world, (tmp_path / "gloo_init").as_uri()), nprocs=world, join=True)
