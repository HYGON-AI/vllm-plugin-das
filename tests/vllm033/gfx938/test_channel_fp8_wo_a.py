"""Native BF16 producer/layout parity before Channel-FP8 WO-B, eager and graph."""
import json
import pytest
import os
from types import SimpleNamespace as NS
import torch
from hcu_vllm_das import wo_a_layout as wo
from hcu_vllm_das.quantization.dsv41_channel_fp8 import HcuDSV41ChannelFP8Config


def main():
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    cfg = NS(quant_config=object.__new__(HcuDSV41ChannelFP8Config),
             model_config=NS(hf_config=NS(model_type='deepseek_v41', hidden_size=5120)),
             parallel_config=NS(pipeline_parallel_size=1, tensor_parallel_size=1,
                                prefill_context_parallel_size=1,
                                data_parallel_size=8, enable_expert_parallel=True))
    assert wo.configured(cfg, device), 'native producer preflight did not admit FP8'
    os.environ['HCU_VLLM_WO_A_TOKEN_MAJOR'] = '0'
    assert not wo.configured(cfg, device), 'explicit rollback must win'
    del os.environ['HCU_VLLM_WO_A_TOKEN_MAJOR']
    assert not wo.configured(NS(**{**cfg.__dict__, 'quant_config': object()}), device)
    torch.manual_seed(7004)
    weight = torch.randn((8192, 4096), dtype=torch.bfloat16, device=device)
    owner = NS(n_local_heads=64, n_local_groups=8, o_lora_rank=1024,
               rotary_emb=None, rope_head_dim=64, wo_a=NS(weight=weight),
               _wo_b_after_wo_a=lambda value: value)
    cases = 0
    for rows in (1, 8, 32, 48, 64, 65, 128, 192):
        x = torch.randn((rows, 64, 512), dtype=torch.bfloat16, device=device)
        with torch.inference_mode():
            expected = wo._O_PROJ(owner, x, None)
            actual = wo._project(wo.native._get_cached_wo_a_bf16, owner.wo_a, x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.is_contiguous()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay = wo._project(wo.native._get_cached_wo_a_bf16, owner.wo_a, x)
            for _ in range(3):
                x.normal_()
                graph.replay()
                expected = wo._O_PROJ(owner, x, None)
                torch.testing.assert_close(replay, expected, rtol=0, atol=0)
                assert replay.is_contiguous()
                cases += 1
    torch.cuda.synchronize()
    print(json.dumps({'graph_cases': cases, 'eager_shapes': 8,
                      'worker_preflight': True, 'channel_fp8_admission': True,
                      'rollback_and_foreign_quant': True, 'byte_exact_bf16': True,
                      'consumer_contract': 'unchanged native BF16 input to Channel-FP8 quantization',
                      'model_accuracy': 'not run'}))


@pytest.mark.hcu
def test_channel_fp8_wo_a_replay():
    main()


if __name__ == '__main__':
    main()
