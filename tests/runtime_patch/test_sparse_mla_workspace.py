from types import SimpleNamespace
import pickle

import pytest

from vllm_hcu.patch.config import HcuFeatureConfig


@pytest.mark.parametrize('value', [1, 0, -1, True, 2.5, '2'])
def test_reject_invalid_factor(value):
    with pytest.raises((ValueError, TypeError)):
        HcuFeatureConfig(sparse_mla_prefill_workspace_factor=value)


@pytest.mark.parametrize('length,factor', [(163840, 5), (262144, 5), (262145, 2), (1048576, 2)])
def test_resolve_once_and_serialize(length, factor):
    from vllm_hcu.v1.attention.ops.sparse_mla_workspace import resolve_workspace_factor, workspace_rows, indexer_rows
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=length), additional_config={})
    resolve_workspace_factor(cfg)
    assert workspace_rows(cfg) == length * factor
    assert indexer_rows(cfg) == length * 8 * factor
    cfg = pickle.loads(pickle.dumps(cfg))
    cfg.model_config.max_model_len = 88000
    resolve_workspace_factor(cfg)
    assert workspace_rows(cfg) == 88000 * factor


def test_explicit_override():
    from vllm_hcu.v1.attention.ops.sparse_mla_workspace import resolve_workspace_factor, workspace_rows
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=1048576), additional_config={'other': 7, 'hcu': {'sparse_mla_prefill_workspace_factor': 3}})
    resolve_workspace_factor(cfg)
    assert workspace_rows(cfg) == 3 * 1048576
    assert cfg.additional_config['other'] == 7


def test_bound_helper_does_not_change_shared_globals():
    from vllm_hcu.v1.attention.ops.sparse_mla_workspace import bind_workspace_helper
    namespace = {'get_prefill_workspace_size': lambda length: 5 * length}
    exec('def build(self, metadata):\n return get_prefill_workspace_size(self.vllm_config.model_config.max_model_len)\n', namespace)
    build = namespace['build']
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=1048576), additional_config={'hcu': {'sparse_mla_prefill_workspace_factor': 2}})
    bound = bind_workspace_helper(build, cfg)
    assert bound(SimpleNamespace(vllm_config=cfg), None) == 2097152
    assert namespace['get_prefill_workspace_size'](1048576) == 5242880


def test_real_v0251_flashmla_allocation_and_helper(monkeypatch):
    import vllm.v1.attention.backends.mla.flashmla_sparse as flash
    from vllm_hcu.v1.attention.ops.sparse_mla_workspace import install_workspace_adapters, indexer_rows
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=1048576), scheduler_config=SimpleNamespace(max_num_batched_tokens=32), additional_config={'hcu': {'sparse_mla_prefill_workspace_factor': 2}})
    allocations = []
    def allocate(*specs):
        allocations.extend(shape for shape, dtype in specs)
        return [None] * len(specs)
    # A private module/class retains the actual installed method's bytecode.
    from types import ModuleType
    module = ModuleType('private_flashmla')
    module.__dict__.update(flash.__dict__)
    class Impl(flash.FlashMLASparseImpl):
        pass
    class Builder(flash.FlashMLASparseMetadataBuilder):
        pass
    module.FlashMLASparseImpl = Impl
    module.FlashMLASparseMetadataBuilder = Builder
    module.get_current_vllm_config = lambda: cfg
    monkeypatch.setitem(flash.FlashMLASparseImpl.__init__.__globals__, 'get_current_vllm_config', lambda: cfg)
    monkeypatch.setitem(flash.FlashMLASparseImpl.__init__.__globals__, 'current_workspace_manager', lambda: SimpleNamespace(get_simultaneous=allocate))
    install_workspace_adapters(module)
    obj = Impl(64, 576, 0.1, 1, None, None, 'fp8_ds_mla', None, 'decoder', None, kv_lora_rank=512)
    assert obj.prefill_workspace_shape == (2097152, 576)
    assert allocations[-1] == (2097152, 576)
    assert indexer_rows(cfg) == 16777216
    assert flash.get_prefill_workspace_size(1048576) == 5242880


@pytest.mark.parametrize('factor,expected_chunks', [(2, 2), (5, 1)])
def test_real_prefill_chunks_fit_reserved_rows(monkeypatch, factor, expected_chunks):
    import torch
    import vllm.v1.attention.backends.mla.flashmla_sparse as flash
    from vllm_hcu.v1.attention.ops.sparse_mla_workspace import bind_workspace_helper
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100), additional_config={'hcu': {'sparse_mla_prefill_workspace_factor': factor}})
    original = flash.FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode
    bound = bind_workspace_helper(original, cfg)
    bound.__globals__['split_decodes_and_prefills'] = lambda *args, **kwargs: (0, 3, 0, 6)
    zeros = torch.zeros
    monkeypatch.setattr(torch, 'zeros', lambda *args, **kwargs: zeros(*args, **dict(kwargs, pin_memory=False)))
    cm = SimpleNamespace(num_actual_tokens=6, seq_lens_cpu_upper_bound=torch.tensor([100, 100, 100]), seq_lens=torch.tensor([100, 100, 100]), query_start_loc_cpu=torch.tensor([0, 2, 4, 6]), block_table_tensor=torch.zeros(3, 2, dtype=torch.int32))
    builder = SimpleNamespace(vllm_config=cfg, reorder_batch_threshold=1, device='cpu')
    metadata = bound(builder, cm)
    assert len(metadata.prefill.chunks) == expected_chunks
    assert sum(int(c.chunk_tot_seqlen) for c in metadata.prefill.chunks) == 300
    assert all(int(c.chunk_tot_seqlen) <= factor * 100 for c in metadata.prefill.chunks)
