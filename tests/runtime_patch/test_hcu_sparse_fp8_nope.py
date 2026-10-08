import pytest
import torch
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
from vllm.v1.worker import workspace

from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
    HcuFlashMLASparseImpl,
)


def test_nope_query_is_zero_padded_for_ds_fp8_mla(monkeypatch):
    observed = {}

    def capture_forward(self, q, kv_cache, metadata, layer):
        observed["q"] = q.clone()
        return q, None

    monkeypatch.setattr(FlashMLASparseImpl, "forward_mqa", capture_forward)
    impl = object.__new__(HcuFlashMLASparseImpl)
    impl._fp8_nope = False
    impl.q_concat_buffer = torch.zeros((1, 64, 576), dtype=torch.bfloat16)
    impl.dcp_world_size = 1
    q_nope = torch.ones((1, 64, 512), dtype=torch.bfloat16)
    q_pe = torch.empty((1, 64, 0), dtype=torch.bfloat16)

    impl.forward_mqa((q_nope, q_pe), torch.empty(0), None, None)

    assert observed["q"].shape == (1, 64, 576)
    assert torch.all(observed["q"][..., :512] == 1)
    assert torch.all(observed["q"][..., 512:] == 0)


def test_nope_fp8_padding_only_applies_to_512_wide_query(monkeypatch):
    def init_with_other_head_size(self, *args, **kwargs):
        self.kv_cache_dtype = "fp8_ds_mla"
        self.qk_rope_head_dim = 0
        self.q_concat_buffer = torch.zeros((1, 64, 384), dtype=torch.bfloat16)

    monkeypatch.setattr(FlashMLASparseImpl, "__init__", init_with_other_head_size)
    impl = HcuFlashMLASparseImpl()
    assert not impl._fp8_nope
    assert impl.q_concat_buffer.shape[-1] == 384


@pytest.fixture
def nope_workspace(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU workspace and graph replay")
    manager = workspace.WorkspaceManager(torch.device("cuda"), num_ubatches=2, num_lanes=2)
    monkeypatch.setattr(workspace, "_manager", manager)
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    observed = []

    def init_shared(self, *args, **kwargs):
        self.kv_cache_dtype = "fp8_ds_mla"
        self.qk_rope_head_dim = 0
        self.dcp_world_size = 1
        (self.q_concat_buffer,) = manager.get_simultaneous(((8, 2, 512), torch.bfloat16))

    def consume_query(self, q, kv_cache, metadata, layer):
        observed.append(q)
        return q.clone(), None

    # Only metadata-heavy upstream initialization and the attention kernel
    # are replaced. Allocation and slot ownership use the real manager.
    monkeypatch.setattr(FlashMLASparseImpl, "__init__", init_shared)
    monkeypatch.setattr(FlashMLASparseImpl, "forward_mqa", consume_query)
    for ubatch in (0, 1):
        active_ubatch[0] = ubatch
        for lane in (0, 1):
            with workspace.use_workspace_lane(lane):
                manager.get_simultaneous(((8, 2, 576), torch.bfloat16))
                HcuFlashMLASparseImpl()
    active_ubatch[0] = 0
    manager.lock()
    return manager, active_ubatch, observed


def test_nope_query_storage_is_shared_across_layers(nope_workspace):
    layers = [HcuFlashMLASparseImpl() for _ in range(12)]
    pointers = {layer.q_concat_buffer.untyped_storage().data_ptr() for layer in layers}
    assert len(pointers) == 1


@pytest.mark.parametrize("ubatch,lane", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_nope_query_uses_current_workspace_slot_and_clears_tail(
    nope_workspace, ubatch, lane
):
    manager, active_ubatch, observed = nope_workspace
    layer = HcuFlashMLASparseImpl()
    layer.q_concat_buffer.fill_(7)
    q = torch.ones((3, 2, 512), dtype=torch.bfloat16, device="cuda")
    pe = q.new_empty((3, 2, 0))
    active_ubatch[0] = ubatch
    with workspace.use_workspace_lane(lane):
        (slot,) = manager.get_simultaneous(((8, 2, 576), torch.bfloat16))
        slot.fill_(9)
        actual, _ = layer.forward_mqa((q, pe), q.new_empty(0), None, None)
        assert observed[-1].untyped_storage().data_ptr() == slot.untyped_storage().data_ptr()
    assert torch.all(actual[..., :512] == 1)
    assert torch.all(actual[..., 512:] == 0)


def test_nope_query_graph_replay_clears_reused_workspace(nope_workspace):
    manager, _, _ = nope_workspace
    layer = HcuFlashMLASparseImpl()
    q = torch.ones((3, 2, 512), dtype=torch.bfloat16, device="cuda")
    pe = q.new_empty((3, 2, 0))
    cache = q.new_empty(0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer.forward_mqa((q, pe), cache, None, None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = layer.forward_mqa((q, pe), cache, None, None)
    (slot,) = manager.get_simultaneous(((8, 2, 576), torch.bfloat16))
    for value in (2, 3):
        slot.fill_(11)
        layer.q_concat_buffer.fill_(11)
        q.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(captured[..., :512] == value)
        assert torch.all(captured[..., 512:] == 0)
