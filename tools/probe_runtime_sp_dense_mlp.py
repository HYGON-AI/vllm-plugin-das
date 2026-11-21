# SPDX-License-Identifier: Apache-2.0
"""Native TP4 dense-MLP/reload/Graph probe; not checkpoint qualification.

Run with a private installed Plugin/Adapter prefix and four explicitly selected
devices: python probe_runtime_sp_dense_mlp.py /path/to/new/evidence-directory.
The controller must check device identities/headroom and own process cleanup.
"""
from datetime import timedelta
from pathlib import Path
import json
import sys
import traceback

OUT = Path(sys.argv[1])


def worker(rank):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    # This tiny numerical probe is allowed alongside an existing allocation;
    # bound its own Torch allocator and make no serving/performance claim.
    torch.cuda.set_per_process_memory_fraction(.05)
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig
    from vllm.distributed import (
        init_distributed_environment, initialize_model_parallel,
        cleanup_dist_env_and_memory, get_tp_group, set_custom_all_reduce,
    )
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.models import glm4_moe, hy_v3
    from vllm.v1.worker.workspace import init_workspace_manager
    from hcu_vllm_das.parallel import runtime_sp_attention as sp
    from hcu_vllm_das.parallel.runtime_sp_batch import RuntimeSPBatch, _BATCH_KEY

    result = dict(rank=rank, cases=[], failures=[], allocator_fraction=.05,
                  architecture=torch.cuda.get_device_properties(rank).gcnArchName)

    def compare(label, actual, expected):
        error = float((actual.float() - expected.float()).abs().max())
        try:
            torch.testing.assert_close(actual, expected, atol=.016, rtol=.016)
        except AssertionError as exc:
            result['failures'].append(dict(gate=label, max_abs_error=error, error=str(exc)))
        return error

    def values(shape, epoch):
        count = 1
        for size in shape:
            count *= size
        return ((((torch.arange(count).reshape(shape) + epoch * 7) % 31) - 15).float() / 128)

    parallel = ParallelConfig(tensor_parallel_size=4, disable_custom_all_reduce=True,
                              enable_expert_parallel=True,
                              all2all_backend='allgather_reducescatter')
    config = VllmConfig(parallel_config=parallel, device_config=DeviceConfig('cuda'),
                       additional_config={'hcu': {'enable_custom_sp': True}})
    try:
        with set_current_vllm_config(config):
            set_custom_all_reduce(False)
            init_distributed_environment(world_size=4, rank=rank, local_rank=rank,
                distributed_init_method='file://' + str(OUT / 'rendezvous'),
                backend='nccl', timeout=timedelta(seconds=120))
            initialize_model_parallel(4, 1)
            init_workspace_manager(torch.device('cuda', rank))
            group = get_tp_group()
            assert group.device_communicator.ca_comm is None
            assert result['architecture'].split(':')[0] == 'gfx938'
            sp.install_attention_layers()
            for family, cls in [('glm4', glm4_moe.Glm4MoeMLP), ('hy3', hy_v3.HYV3FeedForward)]:
                for dtype in (torch.bfloat16, torch.float16):
                    torch.set_default_dtype(dtype)
                    with torch.device('cuda'):
                        reference = cls(128, 256, 'silu', prefix='reference.layers.0.mlp')
                        with sp.runtime_sp_model_construction():
                            candidate = cls(128, 256, 'silu', prefix='candidate.layers.0.mlp')
                    assert candidate.gate_up_proj.tp_size == candidate.down_proj.tp_size == 4
                    assert not candidate.down_proj.reduce_results and reference.down_proj.reduce_results
                    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in reference.parameters())
                    case = dict(family=family, dtype=str(dtype), weight_epochs=[],
                                local_parameters=sum(p.numel() for p in candidate.parameters()))
                    result['cases'].append(case)
                    padding = torch.zeros(7, dtype=torch.bool, device='cuda')
                    batch = RuntimeSPBatch.create(4, rank, 7, padding, padding.device)
                    context = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={},
                        is_padding=batch.local_padding, additional_kwargs={_BATCH_KEY: batch})
                    full = torch.zeros(8, 128, dtype=dtype, device='cuda')
                    local = torch.zeros(2, 128, dtype=dtype, device='cuda')
                    pointers = (full.data_ptr(), local.data_ptr(), batch.global_padding.data_ptr())

                    def update(epoch, active):
                        batch.global_padding.copy_(torch.arange(8, device='cuda') >= active)
                        full.copy_(values((8, 128), epoch).to(device='cuda', dtype=dtype))
                        full.masked_fill_(batch.global_padding[:, None], 0)
                        local.copy_(full[rank * 2:(rank + 1) * 2])
                        local.masked_fill_(batch.local_padding[:, None], float('nan'))

                    for weight_epoch in (0, 2):
                        for model in (reference, candidate):
                            for index in (0, 1):
                                model.gate_up_proj.weight.weight_loader(model.gate_up_proj.weight,
                                    values((256, 128), weight_epoch + index).to(dtype), index)
                            model.down_proj.weight.weight_loader(model.down_proj.weight,
                                values((128, 256), weight_epoch + 3).to(dtype))
                            for layer in (model.gate_up_proj, model.down_proj):
                                layer.quant_method.process_weights_after_loading(layer)
                                layer.update_param_tp_status()
                        for name in ('gate_up_proj', 'down_proj'):
                            torch.testing.assert_close(getattr(candidate, name).weight,
                                                       getattr(reference, name).weight, atol=0, rtol=0)
                        update(weight_epoch, 5)
                        with torch.inference_mode(), override_forward_context(context):
                            expected = reference(full)[rank * 2:(rank + 1) * 2]
                            eager = candidate(local)
                            compare('initial-eager-vs-tp', eager, expected)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                captured = candidate(local)
                            observations = []
                            for epoch, active in ((weight_epoch, 5), (weight_epoch + 1, 7), (weight_epoch + 3, 3)):
                                update(epoch, active)
                                expected = reference(full)[rank * 2:(rank + 1) * 2]
                                eager = candidate(local)
                                graph.replay()
                                torch.cuda.synchronize()
                                assert pointers == (full.data_ptr(), local.data_ptr(), batch.global_padding.data_ptr())
                                assert torch.isfinite(captured).all()
                                assert torch.count_nonzero(captured.masked_select(batch.local_padding[:, None])) == 0
                                # Identical policy and inputs must replay identically.
                                torch.testing.assert_close(captured, eager, atol=0, rtol=0)
                                observations.append(dict(epoch=epoch, active=active,
                                    eager_vs_tp=compare('changed-eager-vs-tp', eager, expected),
                                    graph_vs_tp=compare('graph-vs-tp', captured, expected),
                                    graph_vs_eager=float((captured.float() - eager.float()).abs().max())))
                        case['weight_epochs'].append(dict(weight_epoch=weight_epoch, observations=observations))
                        del graph, captured
                    del reference, candidate
                    torch.cuda.synchronize()
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        dist.barrier()
        cleanup_dist_env_and_memory()
        (OUT / f'rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
        if result['failures']:
            raise AssertionError('Original numerical gates failed; see rank receipt')
    except BaseException:
        result['error'] = traceback.format_exc()
        (OUT / f'rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
        raise


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.spawn(worker, nprocs=4, join=True)
