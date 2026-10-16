# SPDX-License-Identifier: Apache-2.0
"""Compare the v0.28.1-dev and BF16-output gate formulas on a live HCU.

Synthetic BF16 inputs at checkpoint dimensions; this is not model accuracy.
Run in a fresh process with VLLM_PLUGINS=__disabled__.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def validate_topk_routing(logits, ids, weights, topk, routed_scaling_factor):
    """Allow tied expert IDs only when their scores form a complete top-k."""
    scores = F.softplus(logits).sqrt()
    selected = scores.gather(1, ids)
    reference = scores.topk(topk, dim=-1).values
    torch.testing.assert_close(
        selected.sort(dim=-1).values, reference.sort(dim=-1).values,
        rtol=0, atol=0,
    )
    assert (ids.sort(-1).values.diff(dim=-1) > 0).all()
    expected_weights = selected / selected.sum(-1, keepdim=True) * routed_scaling_factor
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden-size', type=int, default=4096)
    parser.add_argument('--experts', type=int, default=256)
    parser.add_argument('--topk', type=int, default=6)
    parser.add_argument('--routed-scaling-factor', type=float, default=1.5)
    parser.add_argument('--tokens', type=int, nargs='+', default=[1, 17, 128, 512])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--construction-only', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('A live HCU/ROCm device is required')
    if not 0 < args.topk < args.experts:
        parser.error('topk must be between zero and experts')

    from vllm.config import VllmConfig, get_current_vllm_config_or_none, set_current_vllm_config
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                 destroy_model_parallel, destroy_distributed_environment)
    from vllm_hcu.patch.worker.op_opt import patch_gate_linear
    from vllm_hcu.platforms import envs as henvs

    assert Path(patch_gate_linear.__file__).resolve().is_relative_to(
        Path(__file__).resolve().parents[1]
    ), patch_gate_linear.__file__
    config = VllmConfig()
    module = importlib.import_module(patch_gate_linear.TARGET_MODULE)
    assert patch_gate_linear.apply_to_module(module)
    assert not patch_gate_linear.apply_to_module(module)
    report = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'hidden_size': args.hidden_size, 'experts': args.experts,
              'topk': args.topk, 'input': 'synthetic BF16 normal, weight std=1/sqrt(hidden_size)',
              'routing': 'sqrt(softplus), zero correction bias, normalized weights',
              'routed_scaling_factor': args.routed_scaling_factor,
              'construction': [], 'comparisons': []}
    with tempfile.TemporaryDirectory() as tmp:
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                    distributed_init_method=f'file://{tmp}/dist', backend='nccl')
        try:
            with set_current_vllm_config(config):
                initialize_model_parallel(1, 1)
            for model_type in ('deepseek_v4', 'deepseek_v3', None):
                # No checkpoint download is needed: the context itself and the
                # GateLinear constructor are real; only HF metadata is minimal.
                config.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
                with set_current_vllm_config(config), torch.device('cuda'):
                    assert patch_gate_linear._get_model_type() == model_type
                    gate = module.GateLinear(args.hidden_size, args.experts, bias=False,
                                             params_dtype=torch.bfloat16, out_dtype=torch.float32,
                                             prefix='model.layers.0.mlp.gate')
                assert get_current_vllm_config_or_none() is None
                assert gate._vllm_hcu_deepseek_v4_gate == (model_type == 'deepseek_v4')
                assert gate.weight.shape == (args.experts, args.hidden_size)
                assert gate.bias is None and gate.out_dtype == torch.float32
                assert gate.allow_cublas_router_gemm
                henvs.VLLM_USE_NN = False
                generator = torch.Generator(device='cuda').manual_seed(20261012)
                gate.weight.data.normal_(generator=generator)
                x = torch.randn(1, args.hidden_size, device='cuda',
                                dtype=torch.bfloat16, generator=generator)
                y, bias = gate.forward(x)
                assert bias is None and y.dtype == torch.float32
                assert y.shape == (1, args.experts)
                expected = (torch.mm(x, gate.weight.T).float() if model_type == 'deepseek_v4'
                            else torch.mm(x, gate.weight.T, out_dtype=torch.float32))
                torch.testing.assert_close(y, expected, rtol=0, atol=0)
                report['construction'].append(model_type)
                if model_type == 'deepseek_v4':
                    v4_gate = gate
            if not args.construction_only:
                config.model_config = None
                for nn_layout in (False, True):
                    henvs.VLLM_USE_NN = nn_layout
                    for seed in args.seeds:
                        generator = torch.Generator(device='cuda').manual_seed(seed)
                        weight = torch.randn(args.experts, args.hidden_size, device='cuda',
                                             dtype=torch.bfloat16, generator=generator) / math.sqrt(args.hidden_size)
                        v4_gate.weight.data = weight.T.contiguous() if nn_layout else weight
                        matrix = v4_gate.weight if nn_layout else v4_gate.weight.T
                        for tokens in args.tokens:
                            x = torch.randn(tokens, args.hidden_size, device='cuda',
                                            dtype=torch.bfloat16, generator=generator)
                            def route(logits):
                                scores = F.softplus(logits).sqrt()
                                ids = scores.topk(args.topk, dim=-1).indices
                                weights = scores.gather(1, ids)
                                return ids, weights / weights.sum(-1, keepdim=True) * args.routed_scaling_factor
                            def base():
                                logits = torch.mm(x, matrix, out_dtype=torch.float32)
                                return logits, *route(logits)
                            def head():
                                logits, _ = v4_gate.forward(x)
                                return logits, *route(logits)
                            eager_base, eager_head = base(), head()
                            expected = torch.mm(x, matrix).float()
                            torch.testing.assert_close(eager_head[0], expected, rtol=0, atol=0)
                            graph_results = []
                            graph_exact = True
                            graph_logits_exact = True
                            for name, fn, eager in [('base', base, eager_base), ('head', head, eager_head)]:
                                stream = torch.cuda.Stream()
                                stream.wait_stream(torch.cuda.current_stream())
                                with torch.cuda.stream(stream):
                                    for _ in range(3):
                                        fn()
                                torch.cuda.current_stream().wait_stream(stream)
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    captured = fn()
                                graph.replay()
                                torch.cuda.synchronize()
                                graph_logits_exact &= torch.equal(captured[0], eager[0])
                                torch.testing.assert_close(captured[0], eager[0], rtol=0, atol=0)
                                # BF16 rounding creates ties. CUDA graph topk may
                                # choose different tied IDs while remaining valid.
                                validate_topk_routing(
                                    captured[0], captured[1], captured[2],
                                    args.topk, args.routed_scaling_factor,
                                )
                                graph_exact &= all(torch.equal(a, b) for a, b in zip(captured, eager))
                                graph_results.append(tuple(t.clone() for t in captured))
                            for mode, b, h in [('eager', eager_base, eager_head),
                                               ('graph', *graph_results)]:
                                b_ids = b[1].sort(-1).values
                                h_ids = h[1].sort(-1).values
                                # Compare weights aligned over every expert, not ranks.
                                b_weights = torch.zeros_like(b[0]).scatter(1, b[1], b[2])
                                h_weights = torch.zeros_like(h[0]).scatter(1, h[1], h[2])
                                report['comparisons'].append({
                                    'mode': mode, 'nn_layout': nn_layout, 'seed': seed, 'tokens': tokens,
                                    'logits_max_abs': (h[0]-b[0]).abs().max().item(),
                                    'logits_mean_abs': (h[0]-b[0]).abs().mean().item(),
                                    'topk_set_changed_tokens': (b_ids != h_ids).any(-1).sum().item(),
                                    'weights_max_abs_aligned': (h_weights-b_weights).abs().max().item(),
                                    'weights_mean_l1_per_token': (h_weights-b_weights).abs().sum(-1).mean().item(),
                                    'eager_graph_logits_exact': graph_logits_exact,
                                    'eager_graph_routing_exact': graph_exact})
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
