"""Foreign models and unset recipes must never enter Step5-specific paths."""
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


def config(model="qwen3"):
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model)),
                           parallel_config=SimpleNamespace(data_parallel_size=4, tensor_parallel_size=4, enable_dbo=False))


def step5(c):
    return c is not None and c.model_config.hf_config.model_type == "step5"


@pytest.mark.parametrize("model", ["qwen3", "deepseek_v3", "glm", "unknown"])
def test_foreign_worker_ignores_inherited_dbo_flag(model):
    fn = function("vllm_hcu/v1/hcu_dbo.py", "enable_in_worker", dict(
        is_step5_config=step5, DBO_MIN_TOKENS=4096,
        henvs=SimpleNamespace(optional_custom_op_enabled=lambda x: x)))
    c = config(model)
    assert fn(c) is False and not c.parallel_config.enable_dbo


@pytest.mark.parametrize("model,threshold", [("qwen3",4096), ("qwen3",0), ("step5",0)])
def test_native_ubatch_runner_is_not_wrapped(model, threshold):
    ur = SimpleNamespace(prepare=object(), run=object())
    original = dict(vars(ur))
    fn = function("vllm_hcu/v1/hcu_dbo.py", "install_runner", dict(
        DBO_MIN_TOKENS=threshold, is_step5_config=step5,
        henvs=SimpleNamespace(optional_custom_op_enabled=lambda x: x)))
    fn(SimpleNamespace(vllm_config=config(model), ubatch_runner=ur))
    assert vars(ur) == original


@pytest.mark.parametrize("owner", [None, config(), config("glm")])
def test_fabric_rejects_foreign_or_unknown_owner_before_device_probe(owner):
    fn = function("vllm_hcu/distributed/device_communicators/step5_dp_fabric.py", "maybe_install", dict(
        os=SimpleNamespace(environ={"HCU_STEP5_DP_FABRIC":"1"}),
        is_step5_config=step5, henvs=SimpleNamespace(optional_custom_op_enabled=lambda x:x)))
    native = object()
    comm = SimpleNamespace(world_size=4, pynccl_comm=native)
    fn(comm, None, None, "dp:test", vllm_config=owner)
    assert comm.pynccl_comm is native


def test_other_gpu_never_reads_step5_moe_table():
    fn = function("vllm_hcu/model_executor/layers/fused_moe/ep_fused_moe.py", "pick_configs", dict(
        _device_arch=lambda _:"gfx942",
        lib_configs=lambda *a:"native"))
    assert fn(SimpleNamespace(shape=(22,3072,4096),device=SimpleNamespace(index=0)),object(),8,65536)=="native"


def load_tuning():
    spec=importlib.util.spec_from_file_location("step5_tuning_subject", ROOT/"vllm_hcu/ops/step5/tuning.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_recipe_has_no_dense_or_asm_choices(monkeypatch):
    monkeypatch.delenv("HCU_STEP5_TUNING_CONFIG",raising=False)
    t=load_tuning()
    assert t.dense_recipe() is None and t.asm_recipe(22) is None


def test_external_recipe_is_data_and_does_not_execute_code(monkeypatch,tmp_path):
    path=tmp_path/"recipe.json"
    path.write_text(json.dumps(dict(schema_version=1,architecture="gfx938",
        dense=dict(hipblaslt_sha256="a"*64,algorithms={"8,16,32":7}),
        asm_g1=dict(block_m=128,solution=7,persist_by_local_experts={"22":16}))))
    monkeypatch.setenv("HCU_STEP5_TUNING_CONFIG",str(path))
    t=load_tuning()
    assert t.dense_recipe()==({(8,16,32):7},"a"*64)
    assert t.asm_recipe(22)==(7,16)
    path.write_text("__import__('os').abort()")
    with pytest.raises(json.JSONDecodeError):load_tuning().tuning_data()


@pytest.mark.parametrize("build", [False,True])
def test_dense_build_dependency_is_explicit(monkeypatch,build):
    monkeypatch.setenv("VLLM_HCU_BUILD_STEP5_DENSE",str(int(build)))
    tree=ast.parse((ROOT/"setup.py").read_text(encoding="utf-8"))
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id in {"_BUILD_STEP5_DENSE","ext_modules"} for t in n.targets)]
    ns=dict(os=os,CUDAExtension=lambda **kw:kw)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"setup.py","exec"),ns)
    extension=ns["ext_modules"][0]
    assert ("hipblaslt" in extension["libraries"]) is build
    assert ("vllm_hcu/csrc/step5_fp8_dense.cpp" in extension["sources"]) is build


def test_retired_runtime_and_tuning_artifacts_are_absent():
    for name in ["vllm_hcu/ops/step5/prefill_g1.py",
                 "vllm_hcu/distributed/static_eplb_diagnostics.py",
                 "vllm_hcu/model_executor/layers/fused_moe/ep_table.json",
                 "benchmarks/step5/ep_table_decode_candidate.json"]:
        assert not (ROOT/name).exists()


def test_threshold_hook_uses_parallel_owner_marker_and_keeps_native_dbo():
    fn = function("vllm_hcu/v1/hcu_dbo.py", "check_ubatch_thresholds", dict(
        original_threshold=lambda *args: "native", DBO_MIN_TOKENS=4096,
        _MIN_TAIL=64, sys=SimpleNamespace(_getframe=lambda _:SimpleNamespace(
            f_locals={"num_tokens_across_dp":SimpleNamespace(max=lambda:8192)}))))
    assert fn(SimpleNamespace(),8192,False)=="native"
    assert fn(SimpleNamespace(_hcu_step5_dbo=True),8192,False) is True
    assert fn(SimpleNamespace(_hcu_step5_dbo=True),8192,True) is False


def test_foreign_shared_experts_ignore_inherited_late_stream_threshold():
    fn = function("vllm_hcu/model_executor/layers/fused_moe/shared_experts.py", "_late_stream", dict(
        _LATE_STREAM_MIN_TOKENS=1, dbo_enabled=lambda: False))
    hidden=SimpleNamespace(shape=(65536,4096))
    assert fn(SimpleNamespace(_hcu_step5_owner=False,_stream=object()),hidden) is False
    assert fn(SimpleNamespace(_hcu_step5_owner=True,_stream=object()),hidden) is True


@pytest.mark.parametrize("owner", [False, True])
def test_combine_hooks_install_only_for_step5_constructor(owner):
    path=ROOT/"vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="MoERunner")
    ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    index=next(i for i,n in enumerate(ctor.body) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Attribute) and t.attr=="_ep_fused_combine" for t in n.targets))
    calls=[]
    ns=dict(self=SimpleNamespace(_hcu_step5_owner=owner),_EP_FUSED_COMBINE=True,
        shared_experts=object(),moe_config=SimpleNamespace(dp_size=4,pcp_size=1,
        is_sequence_parallel=False,skip_final_all_reduce=False),routed_scaling_factor=1.0,
        routed_input_transform=None,routed_output_transform=None,
        _install_step5_combine_hooks=lambda:calls.append("installed"))
    exec(compile(ast.Module(body=ctor.body[index:index+2],type_ignores=[]),str(path),"exec"),ns)
    assert calls==(["installed"] if owner else [])
    assert not any(isinstance(n,ast.If) and isinstance(n.test,ast.Name)
                   and n.test.id=="_EP_FUSED_COMBINE" for n in tree.body)
