# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native builder/packed writer/index conversion/sink forward and graph parity."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace as NS


@pytest.mark.parametrize('lengths,queries,request_ids,capacity,padded,decode_queries', [
    ([115,257], [2,3], [7,7], 512, False, 0),
    ([115,257,133,389], [2,3,1,2], [7,7,9,9], 700, True, 0),
    ([115,257,133,389], [2,3,1,2], [7,7,9,9], 400, False, 0),
    ([79354,84835], [189,189], [7,7], 440000, False, 0),
    ([128,79354,84835], [4,189,189], [3,7,7], 440000, False, 4),
    ([128,79354,84835], [1,189,189], [3,7,7], 440000, True, 1),
])
def test_native_prefill_coalescing_and_changed_input_graph(lengths, queries, request_ids, capacity, padded, decode_queries):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires HCU and VLLM_PLUGINS=hcu_adapter')
    torch.cuda.set_device(0)
    from hcu_adapter.bridges.vllm import hcu_runtime_plugin
    hcu_runtime_plugin()
    from hcu_adapter.kernels import vllm_sparse_flashmla_dtk as adapter
    from hcu_adapter.bridges.vllm_sparse_flashmla import require_available
    from vllm import _custom_ops as ops
    from vllm.v1.worker.workspace import init_workspace_manager
    from hcu_vllm_das import hy4_prefill_backend as backend
    require_available()
    assert backend.HcuHY4PCPWorkspaceBackend.get_impl_cls() is adapter.HcuSparseFlashMLABackend.get_impl_cls()
    assert backend.HcuHY4PCPWorkspaceBackend.get_name() == adapter.HcuSparseFlashMLABackend.get_name()
    torch.manual_seed(397)
    init_workspace_manager(torch.device('cuda:0'))
    rows = sum(queries); page = 64; heads = 64
    unique = list(dict.fromkeys(request_ids)); max_len = max(lengths)
    blocks = (max_len + page - 1) // page
    pages = blocks * len(unique)
    backing = torch.zeros((pages, 2 if padded else 1, page, 656), dtype=torch.uint8, device='cuda')
    cache = backing[:, 0]
    latent = (torch.randn(pages * page, 512, device='cuda') * .1).bfloat16()
    rope = (torch.randn(pages * page, 64, device='cuda') * .1).bfloat16()
    scale = torch.ones(1, device='cuda')
    def write():
        ops.concat_and_cache_mla(latent, rope, cache, torch.arange(pages * page, device='cuda'), 'fp8_ds_mla', scale)
    write()
    table = torch.tensor([[unique.index(req) * blocks + i for i in range(blocks)]
                          for req in request_ids], dtype=torch.int32, device='cuda')
    cu = torch.tensor([0, *np.cumsum(queries).tolist()], dtype=torch.int32)
    seq_cpu = torch.tensor(lengths, dtype=torch.int32)
    seq_gpu = seq_cpu.cuda()
    common = NS(req_idx=np.array(request_ids), seq_lens_cpu_upper_bound=seq_cpu,
                seq_lens=seq_gpu, query_start_loc_cpu=cu, block_table_tensor=table,
                num_actual_tokens=rows)
    cfg = NS(model_config=NS(max_model_len=capacity), attention_config=NS(sparse_mla_prefill_workspace_factor=1))
    builder = NS(use_pcp=True, pcp_dcp_kv_gather=False, use_hisparse=False,
                 device=torch.device('cuda:0'), vllm_config=cfg,
                 _hcu_workspace_l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size,
                 dummy_block_table=torch.zeros((1, 1), dtype=torch.int32, device='cuda'),
                 max_model_len_tensor=torch.full((1,), max_len, dtype=torch.int32, device='cuda'))
    num_decodes = int(decode_queries > 0)
    classification = NS(num_decodes=num_decodes, num_prefills=len(lengths) - num_decodes,
                        num_decode_tokens=decode_queries)
    baseline_builder = object.__new__(adapter.HcuSparseFlashMLABackend.get_builder_cls())
    baseline_builder.__dict__.update(builder.__dict__)
    original = baseline_builder._build_fp8_separate_prefill_decode(common, classification)
    owned_builder = object.__new__(backend.HcuHY4PCPWorkspaceBackend.get_builder_cls())
    owned_builder.__dict__.update(builder.__dict__)
    changed = owned_builder._build_fp8_separate_prefill_decode(common, classification)
    if len(original.prefill.chunks) == 1 and sum(int(c.chunk_tot_seqlen) for c in original.prefill.chunks) * 576 * 2 <= builder._hcu_workspace_l2_bytes:
        # Separate builder calls allocate separate native metadata objects.
        assert [int(c.chunk_tot_seqlen) for c in changed.prefill.chunks] == [int(c.chunk_tot_seqlen) for c in original.prefill.chunks]
        assert torch.equal(changed.prefill.request_ids, original.prefill.request_ids)
        assert torch.equal(changed.prefill.workspace_starts, original.prefill.workspace_starts)
    else:
        assert changed is not original
    virtual = torch.repeat_interleave(torch.arange(len(lengths), device='cuda', dtype=torch.int32),
                                     torch.tensor(queries, device='cuda'))
    ids = torch.full((rows, 2048), -1, dtype=torch.int32, device='cuda')
    for i, length in enumerate(lengths):
        selection = slice(int(cu[i]), int(cu[i + 1])); valid = min(2048, length)
        ids[selection, :valid] = torch.arange(valid, device='cuda') * (length - 1) // max(valid - 1, 1)
    metadata = lambda extra: NS(fp8_extra_metadata=extra, req_id_per_token=virtual,
                                block_table=table, block_size=page)
    impl = object.__new__(adapter.HcuSparseFlashMLABackend.get_impl_cls())
    impl.num_heads = heads; impl.prefill_padding = 64; impl.softmax_scale = .25
    impl.fp8_decode_padded_heads = heads
    impl.sinks = torch.linspace(-1, 1, heads, device='cuda')
    impl.index_group = None; impl.pcp_dcp_kv_gather = False; impl.kv_cache_dtype = 'fp8_ds_mla'
    impl.workspace_specs = [((rows, heads, 576), torch.bfloat16),
                            ((capacity, 576), torch.bfloat16),
                            ((rows, heads, 576), torch.bfloat16),
                            ((rows, heads, 512), torch.bfloat16),
                            ((rows, heads, 512), torch.bfloat16)]
    query = (torch.randn(rows, heads, 576, device='cuda') * .1).bfloat16()
    def run(extra):
        return impl._forward_fp8_kv_separate_prefill_decode(query, cache, ids, metadata(extra))[0]
    for _ in range(3): run(original); run(changed)
    reference = run(original).clone(); actual = run(changed).clone()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): captured = run(changed)
    for factor in (.5, -1.):
        latent.mul_(factor); rope.add_(.01); write(); query.mul_(factor); impl.sinks.add_(.1)
        reference = run(original).clone()
        graph.replay()
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)
