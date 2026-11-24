# SPDX-License-Identifier: Apache-2.0
"""CPU selection/layout contracts for the public nightly WASP128 API."""
import ast
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location("wasp_envs", ROOT / "vllm_hcu/platforms/envs.py")
HENVS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HENVS)


def load():
    path = ROOT / "vllm_hcu/model_executor/layers/fused_moe/step5_wasp128.py"
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)]
    scope = dict(os=os, lru_cache=lru_cache, henvs=HENVS)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


@pytest.fixture
def case():
    scope = load()
    gemm, pack = Mock(), Mock()
    api = {"deepgemm": NS(m_grouped_fp8_gemm_nt_contiguous_wasp_pair32=gemm),
           "deepgemm.group_gemm_weight_pack": NS(pack_fp8_weight_pair32=pack)}
    scope["import_module"] = Mock(side_effect=api.__getitem__)
    model = NS(_hcu_step5_owner=True, num_experts=352, hidden_dim_unpadded=4096,
               intermediate_size_per_partition_unpadded=1536,
               moe_parallel_config=NS(use_deepep_ht_kernels=True, tp_size=1, ep_size=16))
    launch = NS(data_parallel_size=16, data_parallel_size_local=8, enable_dbo=False)
    config = ModuleType("vllm.config")
    config.get_current_vllm_config_or_none = lambda: NS(parallel_config=launch)
    hcu = ModuleType("vllm_hcu.platforms.hcu")
    hcu.on_gfx938 = lambda: True
    step = ModuleType("vllm_hcu.runtime_compat.step5")
    step.is_step5_owner = lambda obj: getattr(obj, "_hcu_step5_owner", False)
    log = ModuleType("vllm.logger")
    log.init_logger = lambda _: NS(info_once=Mock())
    flags = dict(VLLM_HCU_USE_CUSTOM_OPS="1", HCU_STEP5_WASP128="1",
                 DEEP_EP_NORMAL_MNVL="1", HCU_DBO_MIN_TOKENS="4096",
                 HCU_STEP5_DEEPEP_PROFILE_DBO_OUTPUT="1",
                 HCU_STEP5_DEEPEP_DBO_BUFFER_PAIR="1", HCU_STEP5_DEEPEP_FUSED_SCATTER="1")
    with patch.dict(os.environ, flags, clear=True), patch.dict("sys.modules", {
        "vllm.config": config, "vllm_hcu.platforms.hcu": hcu,
        "vllm_hcu.runtime_compat.step5": step, "vllm.logger": log,
    }):
        yield NS(scope=scope, model=model, launch=launch, api=api, gemm=gemm, pack=pack)


@pytest.mark.parametrize("experts", [352, 368])
def test_ep16_selects_explicit_pair32_packing(case, experts):
    case.model.num_experts = experts
    assert case.scope["nightly_weight_packer"](case.model) is case.pack


@pytest.mark.parametrize("flag,value", [("VLLM_HCU_USE_CUSTOM_OPS", "0"), ("HCU_STEP5_WASP128", "0")])
def test_off_does_not_import_nightly(case, flag, value):
    os.environ[flag] = value
    assert case.scope["nightly_weight_packer"](case.model) is None
    case.scope["import_module"].assert_not_called()


def test_other_models_do_not_select_step5_layout(case):
    case.model._hcu_step5_owner = False
    assert case.scope["nightly_weight_packer"](case.model) is None
    case.scope["import_module"].assert_not_called()


@pytest.mark.parametrize("change", ["tp4", "ep8", "local4", "no_mnvL", "intranode", "no_scatter", "no_dbo"])
def test_requested_unsupported_profiles_fail_before_loading_vendor(case, change):
    if change == "tp4": case.model.moe_parallel_config.tp_size = 4
    elif change == "ep8": case.model.moe_parallel_config.ep_size = 8
    elif change == "local4": case.launch.data_parallel_size_local = 4
    elif change == "no_mnvL": os.environ["DEEP_EP_NORMAL_MNVL"] = "0"
    elif change == "intranode": os.environ["VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE"] = "1"
    elif change == "no_scatter": os.environ["HCU_STEP5_DEEPEP_FUSED_SCATTER"] = "0"
    elif change == "no_dbo": os.environ["HCU_DBO_MIN_TOKENS"] = "0"
    with pytest.raises(ValueError, match="TP1/DP16/EP16"):
        case.scope["nightly_weight_packer"](case.model)
    case.scope["import_module"].assert_not_called()


def test_missing_explicit_packer_cannot_use_ordinary_layout(case):
    case.api["deepgemm.group_gemm_weight_pack"].pack_fp8_weight_pair32 = None
    with pytest.raises(RuntimeError, match="explicit Pair32"):
        case.scope["nightly_weight_packer"](case.model)


def tensor(shape):
    return NS(shape=shape, size=lambda dim: shape[dim], numel=lambda: __import__("math").prod(shape))


@pytest.mark.parametrize("experts", [22, 23])
@pytest.mark.parametrize("k,n", [(4096, 3072), (1536, 4096)])
def test_both_gemms_use_mode1002_and_ep16_layout(case, experts, k, n):
    a = (tensor((128, k)), object())
    b = (tensor((experts, k // 64, n // 32, 2, 4, 16, 16)), object())
    out, ids = tensor((128, n)), tensor((128,))
    case.scope["grouped_fp8"](a, b, out, ids)
    case.gemm.assert_called_once_with(a, b, out, ids, config={"MODE": 1002})


def test_wrong_weight_layout_is_rejected_before_gemm(case):
    with pytest.raises(ValueError, match="Pair32 weights"):
        case.scope["grouped_fp8"]((tensor((128, 4096)), None),
                                 (tensor((22, 3072, 4096)), None),
                                 tensor((128, 3072)), tensor((128,)))
    case.gemm.assert_not_called()
