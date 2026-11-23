# SPDX-License-Identifier: Apache-2.0
"""Draft-only capacity checks with production AST and installed Core/Torch.

Real Core allocates InputBuffers. Model loading, attention, target class lookup,
and proposal math are explicit spies: these tests do not run a model forward.
"""

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_hcu/models/hy_v4_flash/nvidia/dfly_speculator.py"


def production_class(monkeypatch):
    core = pytest.importorskip("vllm.v1.worker.gpu.spec_decode.speculator")
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    monkeypatch.setattr(core, "_target_feeds_hc_residual", lambda config: False)

    def set_attn(self, state, kv_config, tables, target_inputs, target_groups):
        self.kv_cache_config = kv_config
        self.block_tables = tables
        self.target_input_buffers = target_inputs
        self.target_attn_groups = target_groups

    monkeypatch.setattr(core.DraftModelSpeculator, "set_attn", set_attn)
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HYV4DFlySpeculator"
    )
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in (
            "__init__",
            "set_attn",
            "propose",
            "attn_vllm_config",
            "_dummy_propose_inputs",
        )
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ns = dict(
        Base=core.DraftModelSpeculator,
        copy=copy,
        torch=torch,
        CUDAGraphMode=CUDAGraphMode,
        BatchExecutionDescriptor=BatchExecutionDescriptor,
        PAD_SLOT_ID=-1,
        DFlyProposeInputs=lambda **kw: NS(**kw),
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    spec_cls = ns[cls.name]
    # Omitted abstract model-loading/capture methods are not invoked in this test.
    spec_cls.__abstractmethods__ = frozenset()
    return spec_cls, CUDAGraphMode, BatchExecutionDescriptor


def config(target_tokens, requests, steps, mrope=True):
    from vllm.config import SchedulerConfig

    scheduler = SchedulerConfig(
        max_model_len=8192,
        is_encoder_decoder=False,
        max_num_batched_tokens=target_tokens,
        max_num_seqs=requests,
        enable_chunked_prefill=True,
    )
    draft = NS(
        get_hidden_size=lambda: 16,
        get_vocab_size=lambda: 32,
        uses_mrope=mrope,
        mrope_num_dims=3,
    )
    return NS(
        cache_config=NS(cache_dtype="auto"),
        kv_transfer_config=None,
        lora_config=None,
        scheduler_config=scheduler,
        model_config=NS(
            max_model_len=8192, dtype=torch.bfloat16, use_fp64_gumbel=False
        ),
        speculative_config=NS(
            method="dfly",
            num_speculative_tokens=steps,
            draft_model_config=draft,
            kv_cache_dtype=None,
            use_local_argmax_reduction=False,
            enable_adaptive_verification=False,
            draft_sample_method="greedy",
        ),
        parallel_config=NS(
            use_ubatching=False,
            data_parallel_size=1,
            data_parallel_rank=0,
            decode_context_parallel_size=1,
            is_moe_model=True,
        ),
        attention_config=NS(sparse_mla_force_mqa=False, use_non_causal=True),
        compilation_config=NS(),
        watermark_config=None,
    )


def allocate_draft_topk(draft_config, device):
    """Run the actual two allocation statements, without constructing layers."""
    path = ROOT / "vllm_hcu/models/hy_v4_flash/nvidia/dfly.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HYV4DFlyModel"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    i = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "max_rows" for t in n.targets)
    )
    module = ast.Module(body=init.body[i : i + 2], type_ignores=[])
    model = torch.nn.Module().to(device)
    # Production allocates under its model/device context; reproduce that scope.
    with torch.device(device):
        exec(
            compile(module, str(path), "exec"),
            dict(
                self=model,
                torch=torch,
                vllm_config=draft_config,
                config=NS(num_hidden_layers=2, index_topk=128),
            ),
        )
    return model.topk_indices


@pytest.mark.parametrize(
    "target_tokens,requests,steps",
    [
        (512, 128, 5),
        (512, 128, 1),
        (4096, 128, 5),
        (512, 2, 5),
        (512, 128, 16),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_all_draft_capacities_share_expanded_budget(
    monkeypatch, target_tokens, requests, steps, device
):
    if device != "cpu" and (not torch.version.hip or not torch.cuda.is_available()):
        pytest.skip("requires HIP")
    cls, _, _ = production_class(monkeypatch)
    cfg = config(target_tokens, requests, steps)
    scheduler = cfg.scheduler_config
    draft = cls(cfg, torch.device(device))
    capacity = max(target_tokens, requests * (steps + 1))
    assert draft.vllm_config is not cfg
    assert cfg.scheduler_config is scheduler
    assert cfg.scheduler_config.max_num_batched_tokens == target_tokens
    assert draft.scheduler_config is not scheduler
    assert draft.scheduler_config.max_num_batched_tokens == capacity
    assert draft.max_num_tokens == capacity
    assert draft.input_buffers.max_num_tokens == capacity
    for value in (
        draft.input_buffers.input_ids,
        draft.input_buffers.positions,
        draft.input_buffers.is_padding,
    ):
        assert value.shape == (capacity,)
    assert draft.mrope_positions.shape == (3, capacity)
    assert draft.attn_vllm_config.scheduler_config.max_num_batched_tokens == capacity
    assert draft.attn_vllm_config.cache_config is cfg.cache_config
    assert cfg.parallel_config.is_moe_model is True
    assert cfg.attention_config.use_non_causal is True
    draft.reuse = NS(validate_groups=lambda kv: None)
    kv_config = NS(kv_cache_groups=[object(), object()])
    draft.set_attn(None, kv_config, None, object(), [])
    assert draft._slot_mappings.shape == (2, capacity)
    topk = allocate_draft_topk(draft.attn_vllm_config, device)
    assert topk.shape == (2, capacity, 128)
    assert (topk == -1).all()
    # Target snapshots still have target-token rows; only draft output expands.
    assert draft.max_num_reqs == requests
    assert draft.draft_tokens.shape == (requests, steps)
    # Model length auto-fit still propagates through the shared model config.
    cfg.model_config.max_model_len = 4096
    assert draft.max_model_len == 4096


def proposal_fixture(monkeypatch, target_tokens=512):
    cls, mode, descriptor = production_class(monkeypatch)
    cfg = config(target_tokens, 128, 5)
    draft = cls(cfg, torch.device("cpu"))
    called = []
    draft.reuse = NS(validate_storage=lambda: called.append("validate"))
    draft._prepare_query = lambda *args, **kwargs: called.append((args[1:], kwargs))
    draft._query_attention_state = lambda *args: ({}, {})
    draft._run_query_model = lambda *args: (
        called.append(args[0]),
        draft.draft_tokens.fill_(7),
    )
    batch = NS(
        num_reqs=128,
        query_start_loc=torch.arange(129) * 4,
        positions=torch.arange(512),
        idx_mapping=torch.arange(128),
        seq_lens_cpu_upper_bound=torch.full((128,), 4, dtype=torch.int32),
    )
    args = dict(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.zeros(512, 16),
        aux_hidden_states=None,
        num_sampled=torch.ones(128, dtype=torch.int32),
        num_rejected=torch.zeros(128, dtype=torch.int32),
        last_sampled=torch.zeros(128, 1, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(1, 128, dtype=torch.int64),
        temperature=torch.zeros(128),
        seeds=torch.zeros(128, dtype=torch.int64),
    )
    return draft, args, called, mode, descriptor


def test_first_proposal_after_short_prefill_can_expand_512_to_768(monkeypatch):
    draft, args, called, _, _ = proposal_fixture(monkeypatch)
    result = draft.propose(**args)
    assert result.shape == (128, 5)
    assert torch.equal(result, torch.full_like(result, 7))
    assert 768 in called
    assert draft.input_buffers.input_ids.numel() >= 768


def test_zero_speculative_tokens_stays_target_only(monkeypatch):
    draft, args, called, _, _ = proposal_fixture(monkeypatch)
    result = draft.propose(**args, num_speculative_tokens=0)
    assert (result == -1).all()
    assert not called


def test_request_overflow_fails_before_query_write(monkeypatch):
    # Token capacity alone would allow 129*6; exercise the request guard.
    draft, args, called, _, _ = proposal_fixture(monkeypatch, target_tokens=4096)
    args["input_batch"].num_reqs = 129
    with pytest.raises(ValueError, match="allocated draft capacity"):
        draft.propose(**args)
    assert not called


def test_padded_overflow_fails_before_query_write(monkeypatch):
    draft, args, called, mode, descriptor = proposal_fixture(
        monkeypatch, target_tokens=4096
    )
    draft.cudagraph_manager = NS(
        dispatch=lambda *a, **k: descriptor(mode.NONE, 774, 129)
    )
    with pytest.raises(ValueError, match="padded query block"):
        draft.propose(**args)
    assert not called


@pytest.mark.parametrize("skip_attn", [False, True])
def test_profile_uses_worst_case_draft_not_target_dummy_count(monkeypatch, skip_attn):
    draft, args, called, _, _ = proposal_fixture(monkeypatch)
    args["input_batch"].num_reqs = 85  # Core target dummy cap: floor(512/6).
    args["input_batch"].seq_lens_cpu_upper_bound = torch.full(
        (85,), 6, dtype=torch.int32
    )
    draft._dummy_propose_inputs = lambda: object()
    result = draft.propose(
        **args, dummy_run=True, is_profile=True, skip_attn_for_dummy_run=skip_attn
    )
    assert result.shape == (128, 5)
    assert 768 in called
    assert "validate" not in called  # No production storage accesses in dummy.


def test_nonprofile_dummy_keeps_core_requested_count(monkeypatch):
    draft, args, called, _, _ = proposal_fixture(monkeypatch)
    args["input_batch"].num_reqs = 85
    args["input_batch"].seq_lens_cpu_upper_bound = torch.full(
        (85,), 6, dtype=torch.int32
    )
    draft._dummy_propose_inputs = lambda: object()
    result = draft.propose(
        **args, dummy_run=True, is_profile=False, skip_attn_for_dummy_run=False
    )
    assert result.shape == (85, 5)
    assert 510 in called
