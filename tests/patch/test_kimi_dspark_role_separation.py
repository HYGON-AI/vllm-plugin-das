"""Contract tests for the DSpark draft-logits role-separation patch.

Port of the vLLM functional-branch fix (kimi-k3-dspark-liuwc 877e6996):
keep the draft-logits all_gather out of the captured DSpark draft graph.
CPU-only fakes cover the contract surface -- interposition, idempotence,
feature gating, shard install/fallback, capture bypass, and the eager
replay tail (no real CUDA graphs or collectives required).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.patch import worker as worker_dispatcher
from vllm_hcu.patch.worker.framework_opt import patch_dspark_role_separation
from vllm_hcu.patch.worker.framework_opt.patch_dspark_role_separation import (
    _GEN_WRAPPER,
    _MARKER,
    _REPLAY_WRAPPER,
    _SHARD_STATE,
    _gather_and_markov,
    _shard_logits_in_graph,
)


def _fake_dspark_module():
    from vllm.config.compilation import CUDAGraphMode

    class DSparkSpeculator:
        def load_draft_model(self, target_model, target_attn_layer_names):
            return "model"

        def _run_model(
            self,
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode=None,
        ):
            return ("ran", num_tokens, cudagraph_runtime_mode)

        def _generate_draft(
            self,
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            head_hidden = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            self._sample_sequential(num_reqs, head_hidden)

        def _sample_sequential(self, num_reqs, head_hidden):
            return ("sampled", num_reqs, head_hidden)

        def init_cudagraph_manager(self, cudagraph_mode):
            self.init_mode = cudagraph_mode

    module = SimpleNamespaceType(
        __name__=patch_dspark_role_separation.TARGET_MODULE,
        DSparkSpeculator=DSparkSpeculator,
    )
    return module


class SimpleNamespaceType(SimpleNamespace):
    """Namespace that behaves like a module for load_exact_module()."""


def _speculator(shard_state=True):
    spec = SimpleNamespace(
        num_speculative_steps=2,
        draft_tokens=None,
        max_num_tokens=8,
        max_num_reqs=4,
        vocab_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
        draft_logits=None,
        _d2t_scatter_index=None,
        _draft_scatter_buf=None,
        model=None,
    )
    setattr(spec, _SHARD_STATE, shard_state)
    return spec


def test_dispatcher_inventory_contains_role_separation_patch():
    name = patch_dspark_role_separation.PATCH_ID
    assert name == "worker.framework_opt.spec_decode.dspark_role_separation"
    assert (name, patch_dspark_role_separation.TARGET_MODULE) in (
        worker_dispatcher.worker_callback_names()
    )
    assert worker_dispatcher._patch_features()[name] == "always"


def test_interposes_three_methods_and_is_idempotent(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        patch_dspark_role_separation,
        "_guard_captured_draft_all_gather",
        lambda where: warnings.append(where),
    )
    module = _fake_dspark_module()
    assert patch_dspark_role_separation.apply_to_module(module) is True
    cls = module.DSparkSpeculator
    assert getattr(module, _MARKER) is True
    assert getattr(cls._generate_draft, _GEN_WRAPPER, False) is True
    # Second application must be a validated no-op.
    assert patch_dspark_role_separation.apply_to_module(module) is False

    inst = cls()
    # load_draft_model on a fake without the shard API follows the guarded
    # fallback path; every other method keeps its wrapped behavior.
    assert inst.load_draft_model("target", set()) == "model"
    assert getattr(inst, _SHARD_STATE) is False
    assert warnings, "fallback must report the captured-AG hazard"
    assert inst._generate_draft(2, 8, None, None, None, "NONE") is None
    inst.init_cudagraph_manager("FULL")
    assert inst.init_mode == "FULL"


def test_feature_off_keeps_manager_unpatched():
    module = _fake_dspark_module()
    patch_dspark_role_separation.apply_to_module(module)
    cls = module.DSparkSpeculator

    class FakeMgr:
        def run_fullgraph(self, desc):
            return desc

    inst = cls()
    inst.query_cudagraph_manager = FakeMgr()
    setattr(inst, _SHARD_STATE, False)
    cls.init_cudagraph_manager(inst, "FULL")
    replay = inst.query_cudagraph_manager.run_fullgraph
    assert getattr(replay, _REPLAY_WRAPPER, False) is False
    assert replay("desc") == "desc"
    assert inst.init_mode == "FULL"


def test_generate_wrapper_shard_path_only_while_capturing(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode

    module = _fake_dspark_module()
    patch_dspark_role_separation.apply_to_module(module)
    cls = module.DSparkSpeculator

    shard_calls = []
    monkeypatch.setattr(
        patch_dspark_role_separation,
        "_shard_logits_in_graph",
        lambda speculator, num_reqs, hidden: shard_calls.append(num_reqs),
    )

    # Eager (mode NONE) falls through to the original sampling loop.
    inst = cls()
    monkeypatch.setattr(
        patch_dspark_role_separation,
        "torch_is_capturing",
        lambda: False,
    )
    assert inst._generate_draft(2, 8, None, None, None, "NONE") is None
    assert shard_calls == []

    # Capture-time execution (mode NONE inside the capture block) stops at
    # the collective-free shard write.
    setattr(inst, _SHARD_STATE, True)
    monkeypatch.setattr(
        patch_dspark_role_separation,
        "torch_is_capturing",
        lambda: True,
    )
    assert inst._generate_draft(2, 8, None, None, None, "NONE") is None
    assert shard_calls == [2]

    # Replay-mode python invocation (mode FULL, not capturing) MUST take
    # the shard path too: the sampling already ran in the replay tail, and
    # falling through to _sample_sequential would join the captured-role
    # c10d comm eagerly (the 10:09 wedge).
    monkeypatch.setattr(
        patch_dspark_role_separation,
        "torch_is_capturing",
        lambda: False,
    )
    assert (
        inst._generate_draft(2, 8, None, None, None, CUDAGraphMode.FULL)
        is None
    )
    assert shard_calls == [2, 2]


def test_replay_tail_appends_eager_gather(monkeypatch):
    module = _fake_dspark_module()
    patch_dspark_role_separation.apply_to_module(module)
    cls = module.DSparkSpeculator

    markov_reqs = []
    monkeypatch.setattr(
        patch_dspark_role_separation,
        "_gather_and_markov",
        lambda speculator, num_reqs: markov_reqs.append(num_reqs),
    )

    class FakeMgr:
        def __init__(self):
            self.replays = 0

        def run_fullgraph(self, desc):
            self.replays += 1
            return desc

    inst = cls()
    setattr(inst, _SHARD_STATE, True)
    setattr(inst, "max_num_reqs", 4)
    mgr = FakeMgr()
    inst.query_cudagraph_manager = mgr

    cls.init_cudagraph_manager(inst, "FULL")
    assert getattr(mgr.run_fullgraph, _REPLAY_WRAPPER, False) is True

    mgr.run_fullgraph(SimpleNamespace(num_reqs=3, num_tokens=6))
    assert mgr.replays == 1
    assert markov_reqs == [3]

    # desc.num_reqs=None mirrors the capture-driver padding rule
    # (dflash/cudagraph.py create_forward_fn): min(num_tokens, max_num_reqs).
    mgr.run_fullgraph(SimpleNamespace(num_reqs=None, num_tokens=9))
    assert markov_reqs[-1] == 4


def test_shard_install_enables_with_sharded_head():
    class FakeLMHead:
        num_embeddings_per_partition = 8
        replicated = False

    class FakeModel:
        lm_head = FakeLMHead()
        compute_draft_logits_shard = staticmethod(lambda hidden: hidden)

    spec = _speculator()
    spec.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(draft_vocab_size=12)
    )
    spec.draft_logits = SimpleNamespace(device=spec.device)

    patch_dspark_role_separation._install_shard_outcome(spec, FakeModel())
    assert getattr(spec, _SHARD_STATE) is True
    assert getattr(spec, "_vllm_hcu_shard_width") == 8
    assert getattr(spec, "_vllm_hcu_draft_vocab_size") == 12
    buf = getattr(spec, "_vllm_hcu_shard_buf")
    assert buf.shape == (spec.max_num_tokens, 8)
    assert buf.dtype == torch.float32
    assert buf.device == spec.device


def test_shard_install_falls_back_without_shard_api(monkeypatch):
    warnings = []

    class ReplicatedHead:
        num_embeddings_per_partition = 8
        replicated = True

    class FakeModel:
        lm_head = ReplicatedHead()

    spec = _speculator()
    monkeypatch.setattr(
        patch_dspark_role_separation,
        "_guard_captured_draft_all_gather",
        lambda where: warnings.append(where),
    )
    patch_dspark_role_separation._install_shard_outcome(spec, FakeModel())
    assert getattr(spec, _SHARD_STATE) is False
    assert len(warnings) == 1


def test_shard_install_honors_role_sep_off(monkeypatch):
    allocations = []
    spec = _speculator()

    class FakeModel:
        lm_head = SimpleNamespace(
            num_embeddings_per_partition=8, replicated=False
        )

    monkeypatch.setenv("VLLM_ROLE_SEP", "0")
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *a, **k: allocations.append(a) or torch.empty(0),
    )
    patch_dspark_role_separation._install_shard_outcome(spec, FakeModel())
    assert getattr(spec, _SHARD_STATE) is False
    assert allocations == []


class _ShardModel:
    def __init__(self, num_reqs: int):
        self._num_reqs = num_reqs
        self._bias_shapes = []

    def compute_draft_logits_shard(self, hidden):
        rows = hidden.shape[0]
        return torch.arange(rows * 4, dtype=torch.float32).view(rows, 4)

    def markov_embed(self, token_ids):
        return token_ids

    def markov_bias(self, markov_embed):
        self._bias_shapes.append(tuple(markov_embed.shape))
        return torch.zeros(self._num_reqs, 1)

    def map_draft_to_target(self, draft_ids):
        return draft_ids


def _shard_speculator(num_reqs: int):
    n_spec = 2
    num_sample = num_reqs * n_spec
    spec = _speculator()
    spec.num_speculative_steps = n_spec
    setattr(spec, "_vllm_hcu_shard_width", 4)
    setattr(spec, "_vllm_hcu_draft_vocab_size", 8)
    spec.sample_indices = torch.arange(num_sample)
    spec.sample_idx_mapping = torch.zeros(num_sample, dtype=torch.int64)
    spec.sample_pos = torch.zeros(num_sample, dtype=torch.int64)
    spec._vllm_hcu_shard_buf = torch.zeros(num_sample + 4, 4)
    spec._anchor_idx = torch.arange(num_reqs) * n_spec
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(8, dtype=torch.int64)
    )
    spec.draft_tokens = torch.zeros(num_reqs, n_spec, dtype=torch.int64)
    spec.model = _ShardModel(num_reqs)
    return spec


def test_shard_write_lands_in_persistent_buffer():
    num_reqs = 2
    spec = _shard_speculator(num_reqs)
    head_hidden = torch.zeros(num_reqs * spec.num_speculative_steps, 6)

    _shard_logits_in_graph(spec, num_reqs, head_hidden)
    buf = spec._vllm_hcu_shard_buf
    num_sample = num_reqs * spec.num_speculative_steps
    # The arange layout the fake shard head produced survived the copy into
    # the persistent buffer exactly as the captured segment wrote it.
    assert buf[0, 1].item() == 1.0
    assert buf[num_sample - 1, 3].item() == float(num_sample * 4 - 1)


def test_eager_gather_then_markov_matches_shape_contract(monkeypatch):
    num_reqs = 2
    tp = 2
    spec = _shard_speculator(num_reqs)
    _shard_logits_in_graph(
        spec, num_reqs, torch.zeros(num_reqs * spec.num_speculative_steps, 6)
    )

    import vllm_hcu.models.kimi_k3.role_separation as role_separation

    monkeypatch.setattr(
        role_separation,
        "eager_all_gather",
        lambda tensor, dim=-1: torch.cat([tensor] * tp, dim=dim),
    )
    _gather_and_markov(spec, num_reqs)
    # full[..., :draft_vocab] fed the per-step Markov slices with
    # (num_reqs,) semantics -- identical shapes to the in-graph path.
    assert spec.model._bias_shapes == [
        (num_reqs,),
    ] * spec.num_speculative_steps
    assert spec.draft_tokens.shape == (num_reqs, spec.num_speculative_steps)


def test_tail_all_gather_docks_on_aux_comm_not_c10d(monkeypatch):
    """Placement contract: the replay tail calls the role_separation
    docking point, never ``tensor_model_parallel_all_gather`` directly --
    the c10d TP comm is captured-role in this runtime."""
    import vllm.distributed

    import vllm_hcu.models.kimi_k3.role_separation as role_separation

    num_reqs = 2
    spec = _shard_speculator(num_reqs)
    _shard_logits_in_graph(
        spec, num_reqs, torch.zeros(num_reqs * spec.num_speculative_steps, 6)
    )

    dock_calls = []

    def _docking_gather(tensor, dim=-1):
        dock_calls.append((tuple(tensor.shape), dim))
        return torch.cat([tensor] * 2, dim=dim)

    monkeypatch.setattr(role_separation, "eager_all_gather", _docking_gather)

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "eager tail joined the c10d TP comm (captured-role) directly"
        )

    monkeypatch.setattr(
        vllm.distributed, "tensor_model_parallel_all_gather", _boom
    )
    _gather_and_markov(spec, num_reqs)
    assert dock_calls and dock_calls[0][1] == -1
    assert spec.draft_tokens.shape == (
        num_reqs,
        spec.num_speculative_steps,
    )


def test_generate_wrapper_guards_capture_without_shard(monkeypatch):
    """Fork parity: with the shard path unavailable, a capture must still
    report the in-graph all_gather hazard instead of recording silently."""
    warnings = []
    monkeypatch.setattr(
        patch_dspark_role_separation,
        "_guard_captured_draft_all_gather",
        lambda where: warnings.append(where),
    )
    module = _fake_dspark_module()
    patch_dspark_role_separation.apply_to_module(module)
    cls = module.DSparkSpeculator
    inst = cls()
    setattr(inst, _SHARD_STATE, False)

    monkeypatch.setattr(
        patch_dspark_role_separation, "torch_is_capturing", lambda: True
    )
    assert inst._generate_draft(2, 8, None, None, None, "NONE") is None
    assert warnings, "capture without shard path must report the hazard"

    monkeypatch.setattr(
        patch_dspark_role_separation, "torch_is_capturing", lambda: False
    )
    assert inst._generate_draft(2, 8, None, None, None, "NONE") is None
    assert len(warnings) == 1, "eager path must stay silent"


def test_eager_all_gather_publishes_on_aux_comm(monkeypatch):
    import vllm.distributed

    from vllm_hcu.models.kimi_k3 import role_separation as rs

    monkeypatch.setenv("VLLM_ROLE_SEP", "1")
    monkeypatch.setattr(rs, "_disabled", False)
    monkeypatch.setattr(rs, "_tp_group", object())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: False)
    monkeypatch.setattr(
        vllm.distributed,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )

    calls = []

    class _Comm:
        # pynccl-truth stub: ncclAllGather has NO dim semantics -- rank r's
        # chunk lands at flat offset r*numel (concat-style along dim 0).
        # Each rank's payload is made distinct (base + r) so a misread
        # layout cannot alias to the right answer.  The first version of
        # this stub implemented dim-concat itself and thereby blessed the
        # original layout bug (runtime evidence 20261010: acceptance ~0
        # with healthy throughput and accuracy).
        @staticmethod
        def all_gather(output_tensor, input_tensor):
            ws = 2
            chunks = [input_tensor + r for r in range(ws)]
            output_tensor.reshape(ws, *input_tensor.shape).copy_(
                torch.stack(chunks)
            )
            calls.append(
                (tuple(output_tensor.shape), tuple(input_tensor.shape))
            )

    monkeypatch.setattr(rs, "_aux_ar_comm", _Comm())

    tensor = torch.arange(6, dtype=torch.float32).view(2, 3)
    out = rs.eager_all_gather(tensor, dim=-1)
    # pynccl concat-style device buffer: (ws * rows, cols), NOT dim-scaled.
    assert calls == [((4, 3), (2, 3))]
    assert out.shape == (2, 6)
    # Official DeviceCommunicatorBase.all_gather contract: the exact
    # dim-concat of the per-rank chunks.
    assert out.tolist() == torch.cat(
        [tensor + r for r in range(2)], dim=-1
    ).tolist(), out


def test_eager_all_gather_falls_back_when_role_sep_off(monkeypatch):
    import vllm.distributed

    from vllm_hcu.models.kimi_k3 import role_separation as rs

    monkeypatch.setenv("VLLM_ROLE_SEP", "0")
    comm_calls = []
    monkeypatch.setattr(
        rs, "_aux_ar_comm",
        SimpleNamespace(all_gather=lambda out, inp: comm_calls.append(1)),
    )
    fallback = []

    def _fake_c10d(tensor, dim=-1):
        fallback.append(1)
        return torch.cat([tensor] * 2, dim=dim)

    monkeypatch.setattr(
        vllm.distributed, "tensor_model_parallel_all_gather", _fake_c10d
    )
    out = rs.eager_all_gather(torch.zeros(2, 3))
    assert out.shape == (2, 6)
    assert fallback and comm_calls == []


def test_eager_all_gather_falls_back_while_capturing(monkeypatch):
    import vllm.distributed

    from vllm_hcu.models.kimi_k3 import role_separation as rs

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: True)
    comm_calls = []
    monkeypatch.setattr(
        rs, "_aux_ar_comm",
        SimpleNamespace(
            all_gather=lambda out, inp: comm_calls.append(1)
        ),
    )
    monkeypatch.setattr(
        vllm.distributed,
        "tensor_model_parallel_all_gather",
        lambda tensor, dim=-1: torch.cat([tensor] * 2, dim=dim),
    )
    out = rs.eager_all_gather(torch.zeros(2, 3))
    assert out.shape == (2, 6)
    assert comm_calls == [], "capture must never touch the auxiliary comm"


def test_apply_lm_head_shard_dispatches_per_runtime_seam():
    # Real-module import must happen in a clean subprocess: under pytest the
    # hcu platform patch timing is not guaranteed for the heavy dspark
    # import chain (deepseek_v2 -> sparse_attn_indexer).
    import os as _os
    import subprocess
    import sys
    from pathlib import Path

    code = r"""
import torch
from types import SimpleNamespace
from vllm_hcu.models.kimi_k3.dspark import apply_lm_head_shard

gemm_calls = []
head = SimpleNamespace(
    quant_method=SimpleNamespace(
        apply=lambda lm, hidden, bias=None: (
            gemm_calls.append((hidden, bias)) or ("gemm", hidden)
        )
    )
)
hidden = torch.zeros(3, 5)

class ForkLP:
    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        return ("head", lm_head, hidden_states, embedding_bias)

assert apply_lm_head_shard(ForkLP(), head, hidden) == (
    "head", head, hidden, None,
)
assert gemm_calls == []

class WheelLP:
    pass

assert apply_lm_head_shard(WheelLP(), head, hidden) == ("gemm", hidden)
assert gemm_calls == [(hidden, None)]

try:
    apply_lm_head_shard(SimpleNamespace(), SimpleNamespace(), hidden)
except RuntimeError as exc:
    assert "shard-only" in str(exc), exc
else:
    raise AssertionError("missing seams must raise RuntimeError")

print("SEAM DISPATCH PASS")
"""
    repo = Path(__file__).resolve().parents[2]
    env = {
        key: value
        for key, value in _os.environ.items()
        if not key.startswith(
            ("CUDA_", "HIP_", "ROCR_", "VLLM_", "PYTEST_")
        )
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    assert "SEAM DISPATCH PASS" in proc.stdout


def test_strict_guard_raises_only_in_strict_mode(monkeypatch):
    monkeypatch.setenv("VLLM_ROLE_SEP_STRICT", "1")
    with pytest.raises(RuntimeError, match="captured draft graph"):
        patch_dspark_role_separation._guard_captured_draft_all_gather("unit")
    monkeypatch.setenv("VLLM_ROLE_SEP_STRICT", "0")
    patch_dspark_role_separation._guard_captured_draft_all_gather("unit")


def _test_head(replicated=False, weight=None):
    head = SimpleNamespace(num_embeddings_per_partition=8, replicated=replicated)
    if weight is not None:
        head.weight = weight
    return head


def test_head_share_fork_parity_resolves_conditional_generation_target():
    # K3 target is a *ForConditionalGeneration wrapper: the wheel share block
    # resolves target_model.lm_head (absent) and silently skips. The parity
    # helper must reach the nested language-model head so the guard premise
    # (non-replicated sharded head) holds -- the exact 19:26 boot failure.
    shared_head = _test_head()

    class FakeLanguageModel:
        lm_head = shared_head

    class FakeConditionalTarget:
        def get_language_model(self):
            return FakeLanguageModel()

    class FakeDraft:
        lm_head = None
        has_own_lm_head = False
        compute_draft_logits_shard = staticmethod(lambda hidden: hidden)

    draft = FakeDraft()
    patch_dspark_role_separation._share_target_lm_head_fork_parity(
        draft, FakeConditionalTarget()
    )
    assert draft.lm_head is shared_head

    spec = _speculator(shard_state=None)
    spec.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(draft_vocab_size=12)
    )
    spec.draft_logits = SimpleNamespace(device=spec.device)
    patch_dspark_role_separation._install_shard_outcome(spec, draft)
    assert getattr(spec, _SHARD_STATE) is True


def test_head_share_respects_own_mismatched_head():
    # A draft with its own differing head keeps it (wheel _should_share gate).
    own_head = _test_head(weight=torch.zeros(4, 2))
    target_head = _test_head(weight=torch.ones(4, 2))

    class FakeTarget:
        lm_head = target_head

    class FakeDraft:
        lm_head = own_head
        has_own_lm_head = True

    draft = FakeDraft()
    patch_dspark_role_separation._share_target_lm_head_fork_parity(
        draft, FakeTarget()
    )
    assert draft.lm_head is own_head


def test_head_share_is_noop_without_any_target_head():
    class FakeDraft:
        lm_head = None
        has_own_lm_head = False

    draft = FakeDraft()
    patch_dspark_role_separation._share_target_lm_head_fork_parity(
        draft, object()
    )
    assert draft.lm_head is None
