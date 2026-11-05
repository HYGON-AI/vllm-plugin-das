"""CPU tests with a fake broadcast; NOT multi-GPU correctness validation."""
import ast
import copy
import importlib.util
import json
import pickle
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import pytest
import torch

module_path = Path(__file__).resolve().parents[2]/"vllm_hcu/v1/glm53_pcp_mtp_sync.py"
spec = importlib.util.spec_from_file_location("glm53_pcp_mtp_sync_candidate", module_path)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class FakeGroup:
    def __init__(self, wire, leader):
        self.wire = wire
        self.leader = leader
        self.calls = 0
        self.world_size = 2
        self.rank_in_group = 0 if leader else 1
        self.sequence = []

    def broadcast(self, tensor, src):
        assert src == 0
        self.calls += 1
        self.sequence.append((tuple(tensor.shape), tensor.dtype, src))
        if self.leader:
            self.wire.append(tensor.clone())
            return tensor
        # Intentionally return another buffer to test return-value handling.
        return self.wire.pop(0).clone()


def output(tokens, counts, rejected):
    counts = torch.tensor(counts, dtype=torch.int32)
    rejected = torch.tensor(rejected, dtype=torch.int32)
    return (SimpleNamespace(sampled_token_ids=torch.tensor(tokens, dtype=torch.int64),
                            num_sampled=counts, num_rejected=rejected), counts, rejected)


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.scope = patch.object(sync, "_in_replicated_mtp_batch", return_value=True)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def test_draft_synchronized_before_feedback(self):
        wire=[]
        leader=SimpleNamespace(sample_draft=lambda x: x+1)
        follower=SimpleNamespace(sample_draft=lambda x: x+9)
        sync._install_draft_sync(leader, FakeGroup(wire, True))
        sync._install_draft_sync(follower, FakeGroup(wire, False))
        a=b=torch.tensor([10,20],dtype=torch.int64)
        for _ in range(3):
            a=leader.sample_draft(a)
            b=follower.sample_draft(b)
            torch.testing.assert_close(a,b)
        self.assertEqual(a.tolist(),[13,23])

    def test_sync_counts_tokens_and_aliases(self):
        wire=[];a=output([[279,1565,999]], [2], [2]);b=output([[279,1565,50541]], [3], [1])
        original_counts=b[1];original_rejected=b[2];original_sampled=b[0].sampled_token_ids
        sync.sync_glm53_pcp_sampler_output(SimpleNamespace(_hcu_glm53_pcp_mtp_sync_group=FakeGroup(wire,True)),a)
        result=sync.sync_glm53_pcp_sampler_output(SimpleNamespace(_hcu_glm53_pcp_mtp_sync_group=FakeGroup(wire,False)),b)
        self.assertIs(result,b)
        self.assertIs(b[0].num_sampled,original_counts)
        self.assertIs(b[0].num_rejected,original_rejected)
        self.assertIs(b[0].sampled_token_ids,original_sampled)
        self.assertEqual(b[0].sampled_token_ids.tolist(),[[279,1565,-1]])
        self.assertEqual(b[1].tolist(),[2]);self.assertEqual(b[2].tolist(),[2])

    def test_noncontiguous_output_and_dtype(self):
        wire=[];a=output([[1,2,3],[4,5,6]],[1,3],[2,0]);b=output([[9,9,9],[9,9,9]],[3,3],[0,0])
        b[0].sampled_token_ids=torch.full((2,6),9,dtype=torch.int32)[:,::2]
        for leader,data in [(True,a),(False,b)]:
            sync.sync_glm53_pcp_sampler_output(SimpleNamespace(_hcu_glm53_pcp_mtp_sync_group=FakeGroup(wire,leader)),data)
        self.assertEqual(b[0].sampled_token_ids.tolist(),[[1,-1,-1],[4,5,6]])
        self.assertEqual(b[0].sampled_token_ids.dtype,torch.int32)
        self.assertEqual(b[1].dtype,torch.int32)

    def test_other_paths_unchanged(self):
        data=output([[4,5]],[2],[0])
        self.assertIs(sync.sync_glm53_pcp_sampler_output(SimpleNamespace(),data),data)
        group=FakeGroup([],False)
        with patch.object(sync,"_in_replicated_mtp_batch",return_value=False):
            self.assertIs(sync.sync_glm53_pcp_sampler_output(SimpleNamespace(_hcu_glm53_pcp_mtp_sync_group=group),data),data)
        self.assertEqual(group.calls,0)

    def test_draft_install_idempotent(self):
        speculator=SimpleNamespace(sample_draft=lambda:torch.tensor([7]))
        group=FakeGroup([],True)
        sync._install_draft_sync(speculator,group)
        first=speculator.sample_draft
        sync._install_draft_sync(speculator,group)
        self.assertIs(first,speculator.sample_draft)
        speculator.sample_draft()
        self.assertEqual(group.calls,1)

    def test_empty_batch(self):
        group=FakeGroup([],True)
        counts=torch.empty(0,dtype=torch.int32)
        out=(SimpleNamespace(sampled_token_ids=torch.empty((0,4),dtype=torch.int64)),counts,counts.clone())
        sync.sync_glm53_pcp_sampler_output(SimpleNamespace(_hcu_glm53_pcp_mtp_sync_group=group),out)
        self.assertEqual(group.calls,0)


@pytest.mark.parametrize("is_profiling", [False, True])
@pytest.mark.parametrize("is_step5", [False, True])
@pytest.mark.parametrize("ubatch_state", ["missing", "none", "present"])
def test_runner_initialization_keeps_glm53_sync_and_step5_dbo_independent(
    monkeypatch, is_profiling, is_step5, ubatch_state
):
    """Execute the actual merged initialization tail with CPU-only doubles."""
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/v1/hcu_model_runner_v2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HcuGPUModelRunnerV2"
    )
    method = copy.deepcopy(next(
        node for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_kv_cache"
    ))
    start = next(
        index for index, node in enumerate(method.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_init_host_seq_lens"
    )
    method.body = method.body[start:]
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))

    calls = []
    dbo = ModuleType("vllm_hcu.v1.hcu_dbo")
    dbo.install_runner = lambda runner: calls.append("step5_dbo")
    monkeypatch.setitem(sys.modules, dbo.__name__, dbo)
    namespace = {
        "is_step5_config": lambda config: config.step5,
        "install_glm53_pcp_mtp_sync": lambda runner: calls.append("glm53_sync"),
    }
    exec(compile(module, str(path), "exec"), namespace)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(step5=is_step5),
        _init_host_seq_lens=lambda: calls.append("host_seq_lens"),
    )
    if ubatch_state != "missing":
        runner.ubatch_runner = object() if ubatch_state == "present" else None
    namespace["initialize_kv_cache"](runner, None, is_profiling=is_profiling)

    expected = ["host_seq_lens"]
    if not is_profiling:
        expected.append("glm53_sync")
    if is_step5 and ubatch_state == "present":
        expected.append("step5_dbo")
    assert calls == expected


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("head_enabled", ["0", "1"])
@pytest.mark.parametrize("serialization", [json, pickle], ids=["json", "pickle"])
@pytest.mark.parametrize("worker_environments", [("0", "1"), (None, "0"), ("invalid", "1")])
def test_serialized_policy_keeps_pcp_workers_collective_sequence_equal(
    monkeypatch, enabled, head_enabled, serialization, worker_environments,
):
    from vllm import distributed
    from test_platform_hcu_config import _glm53_head_hash_config
    from vllm_hcu.patch.platform.core_fix.patch_vllm_config import validate_and_update_hcu_config

    variable = "VLLM_HCU_GLM53_PCP_MTP_SYNC"
    monkeypatch.setenv(variable, str(int(enabled)))
    monkeypatch.setenv("VLLM_HCU_GLM53_KDA_CP_HEAD_SHARD", head_enabled)
    engine = _glm53_head_hash_config(pcp=2, speculative=True)
    validate_and_update_hcu_config(engine)
    wire = []
    groups = [FakeGroup(wire, rank == 0) for rank in range(2)]
    runners = []
    installed = []
    group_lookups = []
    monkeypatch.setattr(sync, "_in_replicated_mtp_batch", lambda: True)
    for rank, worker_value in enumerate(worker_environments):
        worker = copy.copy(engine)
        worker.additional_config = serialization.loads(serialization.dumps(engine.additional_config))
        if worker_value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, worker_value)
        validate_and_update_hcu_config(worker)
        group = groups[rank]

        def get_group():
            group_lookups.append(rank)
            return group

        monkeypatch.setattr(distributed, "get_pcp_group", get_group)
        increment = 1 if rank == 0 else 9
        runner = SimpleNamespace(vllm_config=worker, speculator=SimpleNamespace(
            sample_draft=lambda tokens, step=increment: tokens + step,
        ))
        installed.append(sync.install_glm53_pcp_mtp_sync(runner))
        runners.append(runner)
    assert installed == [enabled, enabled]
    assert group_lookups == ([0, 1] if enabled else [])

    drafts = [torch.tensor([10, 20], dtype=torch.int64) for _ in range(2)]
    for _ in range(3):
        for rank, runner in enumerate(runners):
            drafts[rank] = runner.speculator.sample_draft(drafts[rank])
        if enabled:
            torch.testing.assert_close(drafts[0], drafts[1])
    decisions = [output([[1, 2, 3]], [2], [1]), output([[9, 9, 9]], [3], [0])]
    for runner, decision in zip(runners, decisions):
        sync.sync_glm53_pcp_sampler_output(runner, decision)
    assert [group.calls for group in groups] == ([4, 4] if enabled else [0, 0])
    assert groups[0].sequence == groups[1].sequence
    assert wire == []
    if enabled:
        for leader_tensor, follower_tensor in zip(
            (decisions[0][0].sampled_token_ids, *decisions[0][1:]),
            (decisions[1][0].sampled_token_ids, *decisions[1][1:]),
        ):
            torch.testing.assert_close(leader_tensor, follower_tensor)
    else:
        assert drafts[0].tolist() != drafts[1].tolist()


@pytest.mark.parametrize("worker_value", [None, "0", "1", "invalid"])
def test_eligible_worker_rejects_unresolved_sync_policy_before_group_lookup(monkeypatch, worker_value):
    from vllm import distributed
    from test_platform_hcu_config import _glm53_head_hash_config

    variable = "VLLM_HCU_GLM53_PCP_MTP_SYNC"
    if worker_value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, worker_value)

    def unexpected_group():
        raise AssertionError("unresolved policy must fail before collective group lookup")

    monkeypatch.setattr(distributed, "get_pcp_group", unexpected_group)
    runner = SimpleNamespace(vllm_config=_glm53_head_hash_config(pcp=2, speculative=True),
                             speculator=SimpleNamespace(sample_draft=lambda: None))
    with pytest.raises(RuntimeError, match="sync policy is unresolved"):
        sync.install_glm53_pcp_mtp_sync(runner)


@pytest.mark.parametrize("scope", ["tp2_pcp1", "unrelated", "no_mtp", "no_speculator"])
def test_ineligible_workers_keep_native_path_without_group_lookup(monkeypatch, scope):
    from vllm import distributed
    from test_platform_hcu_config import _glm53_head_hash_config

    monkeypatch.setenv("VLLM_HCU_GLM53_PCP_MTP_SYNC", "invalid")
    config = _glm53_head_hash_config(pcp=1 if scope == "tp2_pcp1" else 2,
                                    speculative=(scope != "no_mtp"))
    if scope == "tp2_pcp1":
        config.parallel_config.tensor_parallel_size = 2
    elif scope == "unrelated":
        config.model_config.architectures = ["OtherModel"]

    def unexpected_group():
        raise AssertionError("ineligible path must not acquire the PCP group")

    monkeypatch.setattr(distributed, "get_pcp_group", unexpected_group)
    speculator = None if scope == "no_speculator" else SimpleNamespace(sample_draft=lambda: None)
    assert sync.install_glm53_pcp_mtp_sync(SimpleNamespace(
        vllm_config=config, speculator=speculator,
    )) is False


@pytest.mark.parametrize("attribute, value, message", [
    ("draft_sample_method", "probabilistic", "greedy draft sampling only"),
    ("enable_adaptive_verification", True, "does not cover adaptive verification"),
])
def test_worker_rechecks_sampling_contract_before_group_lookup(monkeypatch, attribute, value, message):
    from vllm import distributed
    from test_platform_hcu_config import _glm53_head_hash_config
    from vllm_hcu.patch.config import get_hcu_config, set_hcu_config

    config = _glm53_head_hash_config(pcp=2, speculative=True)
    set_hcu_config(config, get_hcu_config(config).with_updates(glm53_pcp_mtp_sync=True))
    setattr(config.speculative_config, attribute, value)

    def unexpected_group():
        raise AssertionError("unsupported sampling must fail before group acquisition")

    monkeypatch.setattr(distributed, "get_pcp_group", unexpected_group)
    runner = SimpleNamespace(vllm_config=config, speculator=SimpleNamespace(sample_draft=lambda: None))
    with pytest.raises(ValueError, match=message):
        sync.install_glm53_pcp_mtp_sync(runner)


if __name__=="__main__":
    unittest.main()
