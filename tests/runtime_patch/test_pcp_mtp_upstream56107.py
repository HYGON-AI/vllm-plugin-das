"""Behavioral regressions for the PCP/MTP backport from vLLM #56107."""

from types import SimpleNamespace

import torch

from tests.runtime_patch.test_glm52_pcp_manager import (
    _InMemoryBlockTables,
    _make_batch,
    _make_managers,
)
from tests.runtime_patch.test_glm52_pcp_runner import (
    _config,
    _MTPExecuteModelState,
)

pytest_plugins = ("tests.runtime_patch.test_glm52_pcp_runner",)


def test_draft_prefill_uses_shifted_ids_in_target_local_token_order():
    batch = _make_batch([("p", list(range(10, 18)), 8, True)])
    managers, _ = _make_managers()
    locals_ = [m.partition_batch(batch) for m in managers]
    shifted = torch.arange(101, 109, dtype=torch.int32)
    fallback = SimpleNamespace()
    for manager in managers:
        manager.prepare_draft_prefill(batch, shifted)
    assert managers[0].get_draft_input_buffers(fallback) is locals_[0]
    assert locals_[0].input_ids.tolist() == [107, 108, 101, 102]
    assert locals_[1].input_ids.tolist() == [103, 104, 105, 106]
    assert batch.input_ids.tolist() == list(range(10, 18))


def test_draft_restores_three_dimensional_feedback_without_alias_copy():
    batch = _make_batch([("p", list(range(10, 18)), 8, True)])
    managers, groups = _make_managers()
    locals_ = [m.partition_batch(batch) for m in managers]
    hidden = [
        (x.input_ids - 10).reshape(-1, 1, 1).expand(-1, 2, 3).clone() for x in locals_
    ]
    for group in groups:
        group.gathered = hidden
    for manager, group, local_hidden in zip(managers, groups, hidden):
        manager.prepare_draft_prefill(batch, batch.input_ids)
        last, feedback = manager.restore_draft_prefill(local_hidden, local_hidden)
        assert last is feedback
        assert last[:, 0, 0].tolist() == list(range(8))
        assert tuple(last.shape) == (8, 2, 3)
        assert group.collective_calls == 1
        fallback = object()
        assert manager.get_draft_input_buffers(fallback) is fallback


def test_replicated_draft_refreshes_uninitialized_global_request_block_rows():
    class ScratchTables(_InMemoryBlockTables):
        def gather_block_tables(self, idx_mapping, num_reqs_padded):
            rows = super().gather_block_tables(idx_mapping, num_reqs_padded)
            for source, scratch in zip(rows, self.input_block_tables):
                scratch[:num_reqs_padded].copy_(source)
            return tuple(t[:num_reqs_padded] for t in self.input_block_tables)

    tables = ScratchTables()
    batch = _make_batch([("a", [1, 2, 3, 4], 4, True), ("b", [5], 11, False)])
    managers, _ = _make_managers(block_tables=tables)
    manager = managers[0]
    local = manager.partition_batch(batch)
    manager.prepare_attn(local)
    assert tables.input_block_tables[0][0].tolist() == [-77] * 4
    tables.allow_global_gather = True
    manager.refresh_draft_block_tables(batch)
    assert tables.input_block_tables[0][:2].tolist() == [
        [40, 41, 42, 43],
        [44, 45, 46, 47],
    ]


def test_new_target_batch_drops_a_stale_local_draft_view():
    first = _make_batch([("p", list(range(8)), 8, True)])
    second = _make_batch([("d", [42], 9, False)])
    managers, _ = _make_managers()
    manager = managers[0]
    manager.partition_batch(first)
    manager.prepare_draft_prefill(first, first.input_ids)
    manager.partition_batch(second)
    fallback = object()
    assert manager.get_draft_input_buffers(fallback) is fallback


def test_draft_prefill_preserves_corrected_positions_and_virtual_padding():
    batch = _make_batch(
        [
            ("p", list(range(10, 15)), 5, True),
            ("continued", [20, 21, 22], 13, True),
            ("decode", [30, 31], 18, False),
        ]
    )
    # GPU rejection correction can lag the CPU upper bound by two tokens.
    batch.positions[-2:] -= 2
    global_ids = batch.input_ids.clone()
    managers, _ = _make_managers()
    locals_ = [manager.partition_batch(batch) for manager in managers]
    originals = [
        (local.input_ids.clone(), local.positions.clone()) for local in locals_
    ]
    assert any(local.is_padding.any() for local in locals_)
    for manager, local, (ids, positions) in zip(managers, locals_, originals):
        manager.prepare_draft_prefill(batch, global_ids + 100)
        assert torch.equal(
            local.input_ids[~local.is_padding], ids[~local.is_padding] + 100
        )
        assert torch.equal(
            local.input_ids[local.is_padding], torch.zeros_like(ids[local.is_padding])
        )
        assert torch.equal(local.positions, positions)
    assert torch.equal(batch.input_ids, global_ids)


def test_mtp_first_forward_is_local_then_feedback_is_global(monkeypatch):
    from contextlib import contextmanager

    from vllm_hcu.model_executor.layers.attention.pcp import effective_pcp_world_size
    from vllm_hcu.v1.spec_decode import pcp_mtp as module

    batch = _make_batch([("p", list(range(10, 18)), 8, True)])
    managers, groups = _make_managers()
    local = [m.partition_batch(batch) for m in managers]
    manager = managers[0]
    shifted = torch.arange(101, 109, dtype=torch.int32)
    manager.prepare_draft_prefill(batch, shifted)
    draft = object.__new__(module.HcuPCPMTPSpeculator)
    draft.pcp_manager = manager
    draft.input_buffers = SimpleNamespace(
        input_ids=shifted.clone(), positions=torch.arange(8)
    )
    draft.hidden_states = torch.zeros(128, 1)
    draft.supports_mm_inputs = False
    draft.vllm_config = object()
    draft.last_token_indices = torch.tensor([7])
    draft.idx_mapping = torch.tensor([10])
    draft.draft_tokens = torch.zeros(1, 3, dtype=torch.int64)
    draft.temperature = draft.seeds = draft.current_draft_step = draft.draft_logits = (
        None
    )
    phases = []

    @contextmanager
    def forward_context(*args, **kwargs):
        phases.append((effective_pcp_world_size(2), kwargs.get("is_padding")))
        yield

    def model(**kwargs):
        ids = kwargs["input_ids"].float().reshape(-1, 1)
        return ids, ids * 10

    monkeypatch.setattr(module.ar, "set_forward_context", forward_context)
    draft.model = model
    draft.sample_draft = lambda states, *args: states[:, 0].to(torch.int64)
    gathered = [
        torch.tensor([[107.0], [108.0], [101.0], [102.0]]),
        torch.tensor([[103.0], [104.0], [105.0], [106.0]]),
    ]
    # The test collective supplies independent rank outputs; logits and feedback
    # restoration use the actual production index mapping, not mocked managers.
    group = groups[0]

    def all_gather(tensor, dim=0):
        assert dim == 0
        factor = 10 if tensor[0, 0] > 1000 else 1
        return torch.cat([x * factor for x in gathered])

    group.all_gather = all_gather
    draft._prefill(1, local[0].num_tokens_after_padding, {}, {}, None)
    assert draft.draft_tokens[0, 0].item() == 108
    assert draft.hidden_states[0, 0].item() == 1080
    assert draft.input_buffers.positions[0].item() == 7
    assert phases[0][0] == 2
    assert torch.equal(phases[0][1], local[0].is_padding)
    assert manager.draft_prefill_batch is None
    draft._run_model(1, {}, {}, None)
    assert phases[1][0] == 1


def test_factory_preserves_non_pcp_speculator(monkeypatch):
    from types import ModuleType

    from vllm_hcu.patch.worker.framework_opt import patch_pcp_mtp_speculator as patch

    module = ModuleType(patch.TARGET_MODULE)
    expected = object()

    def init_speculator(vllm_config, device):
        return expected

    module.init_speculator = init_speculator
    assert patch.apply_to_module(module)
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        speculative_config=SimpleNamespace(method="mtp"),
    )
    assert module.init_speculator(cfg, "cpu") is expected
    assert not patch.apply_to_module(module)


def test_factory_selects_only_single_module_pcp_mtp(monkeypatch):
    from types import ModuleType

    from vllm_hcu.patch.worker.framework_opt import patch_pcp_mtp_speculator as patch
    from vllm_hcu.v1.spec_decode import pcp_mtp

    module = ModuleType(patch.TARGET_MODULE)
    original = object()
    selected = object()

    def init_speculator(vllm_config, device):
        return original

    module.init_speculator = init_speculator
    monkeypatch.setattr(pcp_mtp, "HcuPCPMTPSpeculator", lambda cfg, device: selected)
    patch.apply_to_module(module)
    for pcp, method, gemma, expected in [
        (16, "mtp", False, selected),
        (1, "mtp", False, original),
        (16, "mtp", True, original),
        (16, "eagle3", False, original),
    ]:
        cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
            speculative_config=SimpleNamespace(
                method=method, use_gemma4_mtp=lambda: gemma
            ),
        )
        assert module.init_speculator(cfg, "cpu") is expected


def test_replicated_draft_metadata_marks_all_rows_as_decode(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode import speculator as base

    from vllm_hcu.v1.spec_decode import pcp_mtp as module

    draft = object.__new__(module.HcuPCPMTPSpeculator)
    draft.arange = torch.arange(5, dtype=torch.int32)
    draft.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    draft.input_buffers = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2, 2, 2]),
        seq_lens=torch.tensor([7, 11, 0, 0]),
    )
    draft.block_tables = SimpleNamespace(
        input_block_tables=[torch.tensor([[40], [44], [-1], [-1]])],
        slot_mappings=torch.tensor([[12, 13, -1, -1]]),
    )
    draft.attn_groups = draft.kv_cache_config = None
    draft.draft_max_seq_len = 20
    observed = {}

    def build(**kwargs):
        observed.update(kwargs)
        return {"built": True}

    monkeypatch.setattr(base, "build_attn_metadata", build)
    assert draft._build_draft_attn_metadata(2, 4, 4) == {"built": True}
    phase = observed["model_specific_attn_metadata"]
    assert (
        phase.get_extra_common_attn_kwargs(0, 4)["is_prefilling"].tolist()
        == [False] * 4
    )
    assert observed["query_start_loc_cpu"].tolist() == [0, 1, 2, 2, 2]
    assert observed["block_tables"][0][:2].tolist() == [[40], [44]]


def test_later_drafts_build_metadata_and_slots_in_replicated_mode(monkeypatch):
    from vllm_hcu.model_executor.layers.attention.pcp import effective_pcp_world_size
    from vllm_hcu.v1.spec_decode import pcp_mtp as module

    draft = object.__new__(module.HcuPCPMTPSpeculator)
    draft.input_buffers = SimpleNamespace(
        positions=torch.tensor([10, 20]), query_start_loc=torch.tensor([0, 1, 2])
    )
    draft.idx_mapping = torch.tensor([2, 0])
    draft.current_draft_step = torch.tensor(0)
    draft.num_speculative_steps = 3
    draft.kv_cache_config = None
    phases = []

    def record(name):
        phases.append((name, effective_pcp_world_size(16)))
        return {}

    draft.block_tables = SimpleNamespace(
        compute_slot_mappings=lambda *args: record("slots")
    )
    draft._build_draft_attn_metadata = lambda **kwargs: record("metadata")
    draft._generate_draft = lambda *args, **kwargs: record("forward")
    monkeypatch.setattr(module.ar, "build_slot_mappings_by_layer", lambda *args: {})
    desc = SimpleNamespace(num_tokens=2, num_reqs=2, cg_mode=module.CUDAGraphMode.NONE)
    draft._multi_step_decode(2, False, desc, None)
    assert phases == [("slots", 1), ("metadata", 1), ("forward", 1)] * 2
    assert effective_pcp_world_size(16) == 16


def test_runner_binds_local_first_speculator_to_the_real_manager(
    pcp_runner_module, monkeypatch
):
    module, _ = pcp_runner_module
    manager = object()
    monkeypatch.setattr(module, "maybe_build_pcp_manager", lambda *args: manager)
    runner = module.HcuGPUModelRunnerV2(_config(2), "cpu")
    runner.speculator = SimpleNamespace(pcp_local_prefill=True)
    runner.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[object()]))
    assert runner.speculator.pcp_manager is manager


def test_runner_keeps_pre_hc_residual_local_until_the_first_draft(
    pcp_runner_module, monkeypatch
):
    from vllm_hcu.model_executor.layers.attention.pcp import in_replicated_mtp_batch

    module, _ = pcp_runner_module
    global_batch = SimpleNamespace(num_tokens_after_padding=4)
    local_batch = SimpleNamespace(num_tokens_after_padding=2)
    local_hidden = torch.tensor([[1.0], [2.0]])
    global_hidden = torch.tensor([[1.0], [3.0], [4.0], [2.0]])
    residual = torch.tensor(
        [[10.0, 20.0], [30.0, 40.0], [999.0, 999.0], [999.0, 999.0]]
    )
    manager = SimpleNamespace(
        restore_for_sampling=lambda hidden: (global_hidden, global_batch),
        prepare_global_attn=lambda: ("global_blocks", "global_slots"),
    )
    observed = []

    def consume(runner, grammar):
        assert not in_replicated_mtp_batch()
        assert runner.execute_model_state.attn_metadata == "target_local_metadata"
        assert runner.execute_model_state.hidden_states is global_hidden
        observed.append(manager.draft_target_hidden_states.clone())
        return "sampled"

    monkeypatch.setattr(module.GPUModelRunner, "sample_tokens", consume)
    monkeypatch.setattr(module, "synchronize_pp_spec_draft_tokens", lambda *args: False)
    monkeypatch.setattr(module, "build_slot_mappings_by_layer", lambda *args: {})
    runner = module.HcuGPUModelRunnerV2(_config(2), "cpu")
    runner.is_last_pp_rank = True
    runner.speculator = SimpleNamespace(pcp_local_prefill=True)
    runner.pcp_manager = manager
    runner.model = SimpleNamespace(get_mtp_target_hidden_states=lambda: residual)
    runner.model_state = SimpleNamespace(prepare_attn=lambda *args: "global_metadata")
    runner.attn_groups = runner.kv_cache_config = None
    runner.execute_model_state = _MTPExecuteModelState(
        local_batch, "target_local_metadata", {}, local_hidden, None, set()
    )
    assert runner.sample_tokens(None) == "sampled"
    assert observed[0].tolist() == [[10.0, 20.0], [30.0, 40.0]]
    assert manager.draft_target_hidden_states is None
