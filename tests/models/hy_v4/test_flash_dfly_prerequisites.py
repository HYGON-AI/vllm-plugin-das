# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DFly prerequisites only; these do not qualify speculative decoding."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
FLASH = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4_flash"


def _method(path, name):
    tree = ast.parse((FLASH / path).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(FLASH / path),
                 "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("skip_topk", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_snapshot_keeps_logical_indices_before_attention_remap(skip_topk, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    run = _method("nvidia/attention.py", "_indexer_and_attn")
    rows, width = 3, 2048
    shared = torch.full((8, width), -11, device=device, dtype=torch.int32)
    snapshot = torch.full_like(shared, -7)
    logical = torch.arange(rows * width, device=device, dtype=torch.int32).view(rows, width)
    hidden = torch.zeros(rows, 3072, device=device, dtype=torch.bfloat16)
    out = torch.empty(rows, 2, device=device)

    def indexer(*args):
        shared[:rows].copy_(logical)

    def attention(*args, **kwargs):
        # A downstream layer/backend may overwrite the shared logical buffer.
        # A DFly snapshot must not alias it or be taken after attention.
        shared.add_(10000)
        return torch.ones_like(out)

    attention.impl = SimpleNamespace(record_logical_topk_ready=lambda: None)
    model = SimpleNamespace(indexer=indexer, is_sparse=True, skip_topk=skip_topk,
                            indexer_rope_emb=None, mla_attn=attention,
                            topk_snapshot=snapshot, topk_indices_buffer=shared)

    def invoke():
        run(model, hidden, None, None, None, None, None, out)

    invoke()
    if device == "cuda":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        logical.add_(19)
        graph.replay()
    if skip_topk:
        assert (snapshot == -7).all()
    else:
        torch.testing.assert_close(snapshot[:rows], logical)
        assert (snapshot[rows:] == -7).all()
        assert snapshot.data_ptr() != shared.data_ptr()
    assert (out == 1).all()


def test_indexpool_capture_reserves_full_context_width():
    tree = ast.parse((FLASH / "nvidia/indexpool_cache.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
             and n.name in ("IndexPoolMetadata", "IndexPoolMetadataBuilder")]

    class Base:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, *args):
            pass

    namespace = {"torch": torch, "dataclass": dataclass, "replace": replace,
                 "AttentionMetadataBuilder": Base,
                 "AttentionCGSupport": SimpleNamespace(ALWAYS=1)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<indexpool-metadata>",
                 "exec"), namespace)
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))
    builder = namespace["IndexPoolMetadataBuilder"](None, [], config, "cpu")
    cm = SimpleNamespace(query_start_loc=torch.tensor([0, 4, 6]),
                         seq_lens=torch.tensor([4, 2]),
                         block_table_tensor=torch.tensor([[1, 2], [3, 4]]),
                         num_reqs=2, num_actual_tokens=6, max_seq_len=4)
    ordinary = builder.build(0, cm)
    capture = builder.build_for_cudagraph_capture(cm)
    assert ordinary.max_seq_len == 4
    assert capture.max_seq_len == 8192
    assert capture.num_actual_tokens == ordinary.num_actual_tokens == 6
    for name in ("query_start_loc", "seq_lens", "block_table"):
        assert getattr(capture, name).data_ptr() == getattr(ordinary, name).data_ptr()


@pytest.mark.parametrize("empty_metadata", ["no_tokens", "no_requests"])
def test_engram_dummy_padding_does_not_read_checkpoint_tables(empty_metadata):
    prepare = _method("nvidia/engram.py", "prepare_hashes")
    state = SimpleNamespace(layout=SimpleNamespace(layer_ids=[2, 12, 22, 32],
                                                   lookup_table=None),
                            offsets=torch.empty((4, 3), dtype=torch.int64))
    metadata = SimpleNamespace(num_actual_tokens=0 if empty_metadata == "no_tokens" else 4,
                               seq_lens=torch.empty(0) if empty_metadata == "no_requests" else torch.ones(1))
    ids = torch.tensor([120009, 120009, 120009, 120009], dtype=torch.int64)
    hashes = prepare(state, ids, None, metadata)
    assert hashes.shape == (4, 4, 3)
    assert hashes.dtype == torch.int32
    assert (hashes == -1).all()


@pytest.fixture
def reuse_case():
    # Real v0.33 spec and HCU sink backend; only heavyweight attention
    # constructors/checkpoint loading are replaced by small module shells.
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, get_kv_quant_mode
    from vllm_hcu.models.hy_v4.attention import HYV4MLAAttention
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseBackend

    spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, max_tp_shards=1,
                            head_size=576, dtype=torch.uint8,
                            cache_dtype_str="fp8_ds_mla", state_content_bytes=656,
                            kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"))

    def attention(name, is_target):
        attn = HYV4MLAAttention.__new__(HYV4MLAAttention)
        torch.nn.Module.__init__(attn)
        attn.rotary_emb = torch.nn.Identity()
        attn.num_local_heads = 16
        attn.learnable_sink = True
        attn.indexer = torch.nn.Identity() if is_target else None
        attn.skip_topk = False
        attn.topk_indices_buffer = torch.zeros(4, 2048, dtype=torch.int32)
        attn.topk_snapshot = None
        attn.mla_attn = SimpleNamespace(
            layer_name=name, kv_sharing_target_layer_name=None,
            get_kv_cache_spec=lambda _: replace(spec,
                topk_buffer_width=2048 if is_target else 2176,
                is_index_group_leader=is_target),
            get_attn_backend=lambda: HYV4FlashMLASparseBackend,
            kv_cache=torch.empty(0, dtype=torch.uint8))
        return attn

    selected = [25, 29, 33, 37, 41]
    layers = [SimpleNamespace(self_attn=None) for _ in range(42)]
    for i in selected:
        layers[i].self_attn = attention(f"model.layers.{i}.self_attn", True)
    draft_layers = [SimpleNamespace(self_attn=attention(f"dfly.layers.{i}.self_attn", False))
                    for i in range(5)]
    config = SimpleNamespace(target_index_layer=selected, target_kv_layers=selected,
                             target_index_topk=2048, context_local_window=128)
    target = SimpleNamespace(model=SimpleNamespace(layers=layers))
    draft = SimpleNamespace(config=config, model=SimpleNamespace(
        layers=draft_layers, query_width=6,
        topk_indices=torch.full((5, 24, 2304), -99, dtype=torch.int32)))
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8_ds_mla"))
    return target, draft, vllm_config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dfly_kv_alias_survives_storage_validation(reuse_case, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_target_reuse import HYV4DFlyTargetReuse
    target, draft, config = reuse_case
    reuse = HYV4DFlyTargetReuse(target, draft, config)
    groups = SimpleNamespace(kv_cache_groups=[SimpleNamespace(
        layer_names=list(reuse.aliases) + list(reuse.aliases.values()))])
    reuse.validate_groups(groups)
    for source, dest in reuse.layer_pairs:
        assert dest.mla_attn.kv_sharing_target_layer_name == source.mla_attn.layer_name
        source.mla_attn.kv_cache = torch.zeros((2, 64, 656), dtype=torch.uint8, device=device)
        dest.mla_attn.kv_cache = source.mla_attn.kv_cache.view_as(source.mla_attn.kv_cache)
    reuse.validate_storage()
    for source, dest in reuse.layer_pairs:
        assert dest.mla_attn.kv_sharing_target_layer_name == source.mla_attn.layer_name
        # Keep allocation metadata for later initialization; storage remains
        # writable through the draft-owned Core033 MLA slot mapping.
        dest.mla_attn.kv_cache[:, 1:7].fill_(23)
        assert torch.all(source.mla_attn.kv_cache[:, 1:7] == 23)
        assert torch.all(source.mla_attn.kv_cache[:, 7:] == 0)
    assert reuse.snapshots.shape == (5, 4, 2048)
    assert torch.all(reuse.snapshots == -1)
    assert reuse.layer_snapshot_slots.tolist() == list(range(5))


@pytest.mark.parametrize("bad_layout", ["empty", "copy", "shape", "stride", "dtype"])
def test_dfly_rejects_nonidentical_kv_storage_before_enabling_writes(reuse_case, bad_layout):
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_target_reuse import HYV4DFlyTargetReuse
    reuse = HYV4DFlyTargetReuse(*reuse_case)
    for source, dest in reuse.layer_pairs:
        source.mla_attn.kv_cache = torch.zeros((2, 64, 656), dtype=torch.uint8)
        dest.mla_attn.kv_cache = source.mla_attn.kv_cache
    source, dest = reuse.layer_pairs[-1]
    a = source.mla_attn.kv_cache
    dest.mla_attn.kv_cache = {"empty": torch.empty(0, dtype=torch.uint8),
        "copy": a.clone(), "shape": a.view(1, 128, 656),
        "stride": a.as_strided(a.shape, (1, 2, 128)), "dtype": a.view(torch.int8)}[bad_layout]
    with pytest.raises(ValueError, match="alias the same local layout"):
        reuse.validate_storage()
    assert all(d.mla_attn.kv_sharing_target_layer_name is not None
               for _, d in reuse.layer_pairs)


@pytest.mark.parametrize("bad_contract", ["kv_group", "cache_spec", "sink", "duplicate",
                                          "layer_count", "layer_bounds"])
def test_dfly_rejects_incompatible_target_reuse_contract(reuse_case, bad_contract):
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_target_reuse import HYV4DFlyTargetReuse
    target, draft, config = reuse_case
    if bad_contract == "cache_spec":
        attn = draft.model.layers[0].self_attn.mla_attn
        original = attn.get_kv_cache_spec
        attn.get_kv_cache_spec = lambda c: replace(original(c), state_content_bytes=672)
    elif bad_contract == "sink":
        draft_backend = target.model.layers[25].self_attn.mla_attn
        class NoSink(draft_backend.get_attn_backend()):
            @classmethod
            def supports_sink(cls):
                return False
        draft_backend.get_attn_backend = lambda: NoSink
    elif bad_contract == "duplicate":
        draft.config.target_kv_layers = [25] * 5
    elif bad_contract == "layer_count":
        draft.model.layers.pop()
    elif bad_contract == "layer_bounds":
        draft.config.target_index_layer = [25, 29, 33, 37, 999]
    if bad_contract == "kv_group":
        reuse = HYV4DFlyTargetReuse(*reuse_case)
        with pytest.raises(ValueError, match="share a KV group"):
            reuse.validate_groups(SimpleNamespace(kv_cache_groups=[
                SimpleNamespace(layer_names=list(reuse.aliases)),
                SimpleNamespace(layer_names=list(reuse.aliases.values()))]))
    else:
        with pytest.raises(ValueError):
            HYV4DFlyTargetReuse(*reuse_case)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU/CUDA")
@pytest.mark.parametrize("shared_index_layer", [False, True])
def test_dfly_plugin_reuse_indices_graph_tracks_changed_inputs(reuse_case, shared_index_layer):
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_target_reuse import HYV4DFlyTargetReuse
    target, draft, config = reuse_case
    if shared_index_layer:
        draft.config.target_index_layer = [25] * 5
    reuse = HYV4DFlyTargetReuse(target, draft, config)
    reuse.snapshots = reuse.snapshots.cuda()
    reuse.layer_snapshot_slots = reuse.layer_snapshot_slots.cuda()
    draft.model.topk_indices = torch.full((5, 30, 2304), -99, dtype=torch.int32, device="cuda")
    for slot in range(reuse.snapshots.shape[0]):
        reuse.snapshots[slot].copy_(
            (torch.arange(4 * 2048, device="cuda").view(4, 2048) + slot * 7) % 8192)
    active = torch.tensor([True, True, False, False], device="cuda")
    rows = torch.tensor([1, 2, 0, 0], device="cuda", dtype=torch.int64)
    anchors = torch.tensor([8186, 0, 33, 8187], device="cuda", dtype=torch.int64)

    def check():
        expected = torch.full_like(draft.model.topk_indices, -99)
        for layer, slot in enumerate(reuse.layer_snapshot_slots.tolist()):
            for req in range(4):
                for offset in range(6):
                    row = expected[layer, req * 6 + offset]
                    row.fill_(-1)
                    if not active[req]:
                        continue
                    anchor = int(anchors[req])
                    start = anchor - 128
                    history = reuse.snapshots[slot, int(rows[req])]
                    row[:2048] = torch.where(
                        (history >= 0) & (history < start), history, -1)
                    row[2048:2176] = torch.arange(start, anchor, device="cuda").clamp_min(-1)
                    row[2176:2176 + offset + 1] = torch.arange(
                        anchor, anchor + offset + 1, device="cuda")
        torch.testing.assert_close(draft.model.topk_indices, expected, rtol=0, atol=0)

    reuse.update_indices(active, rows, anchors, 4)
    check()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reuse.update_indices(active, rows, anchors, 4)
    reuse.snapshots.add_(19)
    anchors[1] = 144
    rows[0] = 3
    graph.replay()
    check()


def test_dfly_reuse_accepts_actual_flash_type_and_scoped_rotary_adapter(reuse_case):
    from vllm_hcu.models.hy_v4_flash.nvidia.attention import HYV4FlashAttention
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_rotary import HYV4DFlyRotaryEmbedding
    from vllm_hcu.models.hy_v4_flash.nvidia.dfly_target_reuse import HYV4DFlyTargetReuse
    target, draft, config = reuse_case
    for index in draft.config.target_kv_layers:
        original = target.model.layers[index].self_attn
        actual_type_shell = HYV4FlashAttention.__new__(HYV4FlashAttention)
        torch.nn.Module.__init__(actual_type_shell)
        actual_type_shell.__dict__.update(original.__dict__)
        actual_type_shell.mla_attn.impl = SimpleNamespace(
            topk_indices_buffer=original.topk_indices_buffer)
        target.model.layers[index].self_attn = actual_type_shell
    for layer in draft.model.layers:
        layer.self_attn.rotary_emb = HYV4DFlyRotaryEmbedding(layer.self_attn.rotary_emb)
    reuse = HYV4DFlyTargetReuse(target, draft, config)
    assert len(reuse.aliases) == 5
    # Do not accept an arbitrary wrapper: only the reviewed shape adapter can
    # be unwrapped when comparing target/draft rotary implementations.
    draft.model.layers[0].self_attn.rotary_emb = torch.nn.Sequential(torch.nn.Identity())
    with pytest.raises(ValueError, match="incompatible RoPE"):
        HYV4DFlyTargetReuse(target, draft, config)
