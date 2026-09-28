"""Exercise DP-local TP ownership without a model service or HCU allocation."""

from types import SimpleNamespace
import importlib.util
from pathlib import Path
import ast
from datetime import timedelta

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm_hcu.models.kimi_k3.amd import linear as kimi


@pytest.fixture
def hcu_linear(monkeypatch):
    # Both official and HCU Linear declare the same pluggable names. Production
    # exchanges the module before import; isolate registration in this CPU test.
    from vllm.model_executor import custom_op
    monkeypatch.setattr(custom_op, "op_registry", {})
    monkeypatch.setattr(custom_op, "op_registry_oot", {})
    path = Path(__file__).resolve().parents[3] / "vllm_hcu/model_executor/layers/linear.py"
    spec = importlib.util.spec_from_file_location("_kimi_test_hcu_linear", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_real_parallel_initializer_keeps_tp_inside_dp(monkeypatch, tp):
    """Use the production group planner, replacing only group construction."""
    import vllm.config

    groups = {}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2 * tp)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_world_group",
                        lambda: SimpleNamespace(local_rank=0))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2,
            enable_elastic_ep=False, enable_eplb=False), model_config=None)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    for name in ("_TP", "_DCP", "_PCP", "_PP", "_DP", "_EP", "_EPLB"):
        monkeypatch.setattr(parallel_state, name, None)

    def make_group(ranks, local_rank, backend, **kwargs):
        groups[kwargs["group_name"]] = ranks
        return SimpleNamespace(rank_in_group=0)

    monkeypatch.setattr(parallel_state, "init_model_parallel_group", make_group)
    parallel_state.initialize_model_parallel(tp, backend="gloo")
    assert groups["tp"] == [list(range(tp)), list(range(tp, 2 * tp))]
    assert groups["ep"] == [list(range(2 * tp))]
    assert groups["dp"] == [[i, tp + i] for i in range(tp)]


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("rows", [(1, 7), (0, 5), (3, 0)])
def test_real_kimi_mlp_tp_shards_preserve_each_dp_batch(monkeypatch, hcu_linear, shared, rows):
    """Real Kimi/HCU Linear classes; emulate the TP sum, not the GEMMs.

    Full FP32 MLP is the independent reference. No DP gather is supplied:
    a production attempt to gather DP tokens would fail this test.
    """
    monkeypatch.setenv("VLLM_USE_NN", "0")
    monkeypatch.setattr(kimi, "MergedColumnParallelLinear",
                        hcu_linear.MergedColumnParallelLinear)
    monkeypatch.setattr(kimi, "RowParallelLinear", hcu_linear.RowParallelLinear)
    monkeypatch.setattr(hcu_linear, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(hcu_linear, "dispatch_unquantized_gemm",
                        lambda: lambda layer, x, w, bias: F.linear(x, w, bias))
    torch.manual_seed(228)
    gate, up, down = torch.randn(3, 12, 8) * .1
    down = down.T.contiguous()
    for dp_rank, count in enumerate(rows):
        x = torch.randn(count, 8) + dp_rank * 2
        reference = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
        partials, modules = [], []
        with set_current_vllm_config(VllmConfig()):
            for tp_rank in range(2):
                monkeypatch.setattr(parallel_state, "_TP",
                    SimpleNamespace(rank_in_group=tp_rank, world_size=2))
                monkeypatch.setattr(hcu_linear, "get_tensor_model_parallel_rank",
                                    lambda rank=tp_rank: rank)
                mlp = kimi.KimiMLP(8, 12, "silu", reduce_results=not shared)
                lo, hi = 6 * tp_rank, 6 * (tp_rank + 1)
                # Exercise the production checkpoint loader with fused shard IDs.
                mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, gate, 0)
                mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, up, 1)
                mlp.down_proj.weight_loader(mlp.down_proj.weight, down)
                torch.testing.assert_close(mlp.gate_up_proj.weight,
                    torch.cat((gate[lo:hi], up[lo:hi])))
                torch.testing.assert_close(mlp.down_proj.weight, down[:, lo:hi])
                calls = []
                monkeypatch.setattr(hcu_linear, "tensor_model_parallel_all_reduce",
                                    lambda value: calls.append(value.clone()) or value)
                partials.append(mlp(x))
                assert len(calls) == int(not shared)
                modules.append(mlp)
        torch.testing.assert_close(sum(partials), reference, atol=1e-6, rtol=1e-5)
        assert all(part.shape == (count, 8) for part in partials)


def _gloo_worker(rank, rendezvous, hcu_linear):
    """Real DP-local TP collectives, with an already-combined EP stand-in."""
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=rendezvous,
        rank=rank, world_size=4, timeout=timedelta(seconds=45))
    try:
        groups = [torch.distributed.new_group([0, 1]),
                  torch.distributed.new_group([2, 3])]
        group = groups[rank // 2]
        tp_rank = rank % 2
        path = (Path(__file__).resolve().parents[3] /
                "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py")
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "MoERunner")
        names = {"_maybe_reduce_shared_expert_output", "_maybe_reduce_final_output"}
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in names]
        calls = []

        def reduce(value):
            calls.append(tuple(value.shape))
            torch.distributed.all_reduce(value, group=group)
            return value

        namespace = {"torch": torch, "tensor_model_parallel_all_reduce": reduce}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods,
            type_ignores=[])), str(path), "exec"), namespace)
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("VLLM_USE_NN", "0")
            patch.setattr(kimi, "MergedColumnParallelLinear", hcu_linear.MergedColumnParallelLinear)
            patch.setattr(kimi, "RowParallelLinear", hcu_linear.RowParallelLinear)
            patch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=tp_rank, world_size=2))
            patch.setattr(hcu_linear, "get_tensor_model_parallel_world_size", lambda: 2)
            patch.setattr(hcu_linear, "get_tensor_model_parallel_rank", lambda: tp_rank)
            patch.setattr(hcu_linear, "tensor_model_parallel_all_reduce", reduce)
            patch.setattr(hcu_linear, "dispatch_unquantized_gemm",
                          lambda: lambda layer, x, w, bias: F.linear(x, w, bias))
            torch.manual_seed(112)
            gate, up, down = torch.randn(3, 12, 8) * .1
            down = down.T.contiguous()
            with set_current_vllm_config(VllmConfig()):
                for shared in (False, True):
                    mlp = kimi.KimiMLP(8, 12, "silu", reduce_results=not shared)
                    mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, gate, 0)
                    mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, up, 1)
                    mlp.down_proj.weight_loader(mlp.down_proj.weight, down)
                    for rows in ((1, 7), (0, 5), (3, 0)):
                        # Inputs agree within TP but differ across DP, including
                        # an entirely idle replica. A global reduction is wrong.
                        count = rows[rank // 2]
                        x = torch.arange(count * 8).reshape(count, 8).float() / 20 + rank // 2
                        expected = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
                        calls.clear()
                        out = mlp(x)
                        if shared:
                            runner = SimpleNamespace(moe_config=SimpleNamespace(
                                is_sequence_parallel=False, skip_final_all_reduce=False,
                                tp_size=2, ep_size=4))
                            out = namespace["_maybe_reduce_shared_expert_output"](runner, out, True)
                            routed = torch.full_like(out, rank // 2 + .25)
                            out = namespace["_maybe_reduce_final_output"](runner, out + routed, None, True)
                            expected += routed
                        assert len(calls) == 1, calls
                        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    finally:
        torch.distributed.destroy_process_group()


def test_gloo_dp2_tp2_dense_and_shared_with_idle_replica(tmp_path, hcu_linear):
    torch.multiprocessing.start_processes(_gloo_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", hcu_linear),
        nprocs=4, join=True, start_method="fork")
