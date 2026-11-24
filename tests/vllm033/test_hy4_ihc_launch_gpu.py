# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native decoder loop, real checkpoint gates; attention/MoE leaves doubled."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors import safe_open

pytestmark = pytest.mark.skipif(os.getenv('RUN_HCU_HY4_IHC_GPU') != '1',
                               reason='explicit HCU component run required')


@pytest.fixture(scope='module', params=[0, 37])
def weights(request):
    from hcu_vllm_das import hy4_ihc_launch as owned
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx938'
    root = Path(os.getenv('HY4_TEST_MODEL_DIR', '/models/hy4'))
    cfg = NS(**json.loads(root.joinpath('config.json').read_text()))
    mapping = json.loads(root.joinpath('model.safetensors.index.json').read_text())['weight_map']
    prefix = f'model.layers.{request.param}.'

    def read(name, preserve=False):
        key = prefix + name
        with safe_open(str(root / mapping[key]), framework='pt', device='cpu') as f:
            value = f.get_tensor(key)
        return (value if preserve else value.float()).contiguous().cuda()

    gates = {kind: [read(f'hc_{kind}_layer.hc_pre.{name}')
                   for name in ['hc_fn', 'hc_scale', 'hc_base']]
             for kind in ['attn', 'mlp']}
    norm = read('post_attention_layernorm.weight', preserve=True)
    return owned, cfg, gates, norm


class Attention(torch.nn.Module):
    def forward(self, *, positions, hidden_states):
        return hidden_states * 0.5


class MLP(torch.nn.Module):
    def forward(self, x):
        return x * 0.25


def decoder(cls, owned, cfg, gates, norm, tuned):
    result = cls.__new__(cls)
    torch.nn.Module.__init__(result)
    result.enable_ihc = True
    for kind in ['attn', 'mlp']:
        boundary = owned.hc.HYV4HCLayer.__new__(owned.hc.HYV4HCLayer)
        torch.nn.Module.__init__(boundary)
        boundary.enable_ihc = True
        boundary.config = cfg
        pre_cls = owned.TunedHCPre if tuned else owned.hc.HYV4HCPreLayer
        pre = pre_cls.__new__(pre_cls)
        torch.nn.Module.__init__(pre)
        pre.hpc_op = None
        pre.hc_fn = torch.nn.Module()
        pre.hc_fn.weight = torch.nn.Parameter(gates[kind][0], requires_grad=False)
        pre.hc_scale = torch.nn.Parameter(gates[kind][1], requires_grad=False)
        pre.hc_base = torch.nn.Parameter(gates[kind][2], requires_grad=False)
        pre.layernorm_epsilon = cfg.rms_norm_eps
        pre.hc_eps, pre.magnitude = cfg.hc_eps, cfg.hc_magnitude
        boundary.hc_pre = pre
        post = owned.hc.HYV4HCPostLayer.__new__(owned.hc.HYV4HCPostLayer)
        torch.nn.Module.__init__(post)
        post.hpc_op = None
        boundary.hc_post = post
        setattr(result, f'hc_{kind}_layer', boundary)
    result.input_layernorm = torch.nn.Identity()
    result.post_attention_layernorm = torch.nn.RMSNorm(
        cfg.hidden_size, eps=cfg.rms_norm_eps, device='cuda', dtype=torch.bfloat16)
    result.post_attention_layernorm.weight = torch.nn.Parameter(norm, requires_grad=False)
    result.self_attn, result.mlp = Attention(), MLP()
    return result.eval()


def close_bf16(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert bool(torch.isfinite(actual).all())
    if not actual.numel():
        return
    difference = actual.float() - expected.float()
    relative = difference.norm() / expected.float().norm().clamp_min(1e-12)
    assert float(relative) < 3e-4
    bound = float(expected.abs().max()) / 128 + 1e-5
    assert float(difference.abs().max()) <= bound


@pytest.mark.parametrize('rows', [0, 1, 12, 32, 378, 1134])
@torch.inference_mode()
def test_real_native_decoder_eager_and_changed_input_graph(weights, rows):
    owned, cfg, gates, norm = weights
    original = decoder(owned.native.HYV4DecoderLayer, owned, cfg, gates, norm, False)
    candidate = decoder(owned.TunedDecoder, owned, cfg, gates, norm, True)
    params = list(original.parameters())
    snapshots = [p.clone() for p in params]
    torch.manual_seed(20261122 + rows)
    x = torch.randn(rows, cfg.hc_mult, cfg.hidden_size, dtype=torch.bfloat16, device='cuda')
    positions = torch.arange(rows, dtype=torch.long, device='cuda')
    expected, unused = original(positions, x, None)
    actual, other = candidate(positions, x, None)
    assert unused is None and other is None
    close_bf16(actual, expected)
    if rows:
        # Core's complete post/pre/RMS chain, including the intermediate store.
        hidden = x[:, 0].contiguous()
        post = torch.sigmoid(torch.randn(rows, cfg.hc_mult, device='cuda'))
        arguments = (hidden, x, post, *gates['mlp'], norm,
                     cfg.hc_magnitude, cfg.hc_eps, cfg.rms_norm_eps)
        reference = owned.kernels.triton_ihc_post_pre_rms_norm(*arguments)
        result = owned.tuned_chain(*arguments)
        assert torch.equal(result[2], reference[2])
        torch.testing.assert_close(result[1], reference[1], rtol=1e-5, atol=2e-6)
        close_bf16(result[0], reference[0])
        graphs = []
        for model in [original, candidate]:
            model(positions, x, None)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, _ = model(positions, x, None)
            graphs.append((graph, output))
        for scale in [0.03125, 1.0, 32.0]:
            x.normal_().mul_(scale)
            for graph, _ in graphs:
                graph.replay()
            torch.cuda.synchronize()
            close_bf16(graphs[1][1], graphs[0][1])
            eager, _ = candidate(positions, x, None)
            assert torch.equal(eager, graphs[1][1])
        # Both native and candidate parameter storage/value remain unchanged.
    assert all(torch.equal(p, copy) for p, copy in zip(params, snapshots))
