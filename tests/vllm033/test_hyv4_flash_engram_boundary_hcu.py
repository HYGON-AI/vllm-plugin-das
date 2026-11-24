# SPDX-License-Identifier: Apache-2.0
"""Actual HIP hash preparation through FX and changed-input graph replay."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.hcu


def test_real_engram_hashes_survive_profile_and_changed_input_graph():
    assert torch.version.hip and torch.cuda.is_available(), "assigned HIP device required"
    from vllm.config import CompilationConfig, CompilationMode
    import hcu_vllm_das.models.hy_v4_flash.engram as production

    class LayerDouble:
        pass

    config = NS(
        engram_layer_ids=[1, 2], engram_max_ngram_size=3,
        engram_n_head_per_ngram=2, engram_n_embed_per_ngram=8,
        engram_pad_id=0, vocab_size=128, engram_seed=13,
        engram_layer_head_vocab_sizes={
            "1": [[17, 19], [23, 29]], "2": [[31, 37], [41, 43]],
        },
    )
    state = production.FlashNgramHashState(config, dummy=True).to("cuda")
    layer = LayerDouble()
    compilation = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    compilation.splitting_ops = ["vllm::hcu_hyv4_flash_indexer_and_attn"]
    state.bind_attention_owner(layer, NS(compilation_config=compilation))
    assert "vllm::hcu_hyv4_flash_engram_hashes" in compilation.splitting_ops

    ids = torch.tensor([1, 2, 3, 4, 8, 9, 10, 11, -1, -1, -1, -1],
                       dtype=torch.int32, device="cuda")
    lookback = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([8, 11], dtype=torch.int32, device="cuda")
    metadata = NS(num_actual_tokens=12, seq_lens=lengths, query_start_loc=starts)
    context = NS(attn_metadata=None, no_compile_layers={"hash.attn": layer})
    output = torch.empty((2, 12, 4), dtype=torch.int32, device="cuda")
    graphs, frames = [], []

    def backend(graph, example_inputs):
        graphs.append(graph)
        def execute(*args):
            frames.append(args)
            return graph.forward(*args)
        return execute

    def invoke(ids, lookback, output):
        state.prepare_hashes_with_output(ids, lookback, "hash.attn", output)
        return output

    with patch.object(production, "get_forward_context", lambda: context):
        compiled = torch.compile(invoke, backend=backend, fullgraph=True)
        compiled(ids, lookback, output)
        assert len(graphs) == 1
        assert torch.equal(output, torch.full_like(output, -1))
        saved = frames[-1]
        context.attn_metadata = {"hash.attn": metadata}
        for step in range(3):
            ids[:8].add_(1)
            lookback.add_(1)
            expected = state.prepare_hashes(ids, lookback, metadata)
            graphs[0](*saved)
            assert torch.equal(output, expected)
            assert bool((output[:, :8] >= 0).any())
            assert torch.equal(output[:, 8:], torch.full_like(output[:, 8:], -1))

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                invoke(ids, lookback, output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke(ids, lookback, output)
        pointer = output.data_ptr()
        for step, split in enumerate([3, 5, 4]):
            ids[:8].add_(1)
            lookback.add_(1)
            starts.copy_(torch.tensor([0, split, 8], dtype=torch.int32, device="cuda"))
            lengths.copy_(torch.tensor([10 + step, 13 + step], dtype=torch.int32, device="cuda"))
            graph.replay()
            expected = state.prepare_hashes(ids, lookback, metadata)
            assert output.data_ptr() == pointer
            assert torch.equal(output, expected)
            assert torch.equal(output[:, 8:], torch.full_like(output[:, 8:], -1))
    print("ENGRAM_REAL_HASH_FX_REPLAYS=3 HIP_GRAPH_CHANGED_INPUT_REPLAYS=3")
