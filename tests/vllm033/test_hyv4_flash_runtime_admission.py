# SPDX-License-Identifier: Apache-2.0
"""Portable startup contracts; no native checkpoint/kernel qualification."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

MODEL = (Path(__file__).resolve().parents[2]
         / "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash")

def validate(config):
    tree = ast.parse((MODEL / "model.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "validate_runtime")
    calls = []
    namespace = {"VllmConfig": object, "torch": NS(bfloat16="bf16"),
                 "is_hcu_platform": lambda: True,
                 "_check_quantization": lambda *a: calls.append(a)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]),
                 "<active-runtime-admission>", "exec"), namespace)
    namespace[fn.name](config)
    return calls


def configuration(*, hisparse=False, eager=True):
    config = NS(validate_flash_config=lambda: None, index_topk=2048,
                sliding_window=128, engram_layer_ids=[], enable_ihc=False,
                num_attention_heads=64, swa_num_attention_heads=64)
    return NS(model_config=NS(hf_text_config=config, dtype="bf16",
                              quantization=None, enforce_eager=eager),
              parallel_config=NS(data_parallel_size=1,pipeline_parallel_size=1,
                  prefill_context_parallel_size=1,decode_context_parallel_size=1,
                  enable_eplb=False,enable_dbo=False,tensor_parallel_size=4),
              use_v2_model_runner=True, kv_transfer_config=None,lora_config=None,
              speculative_config=None,cache_config=NS(cache_dtype="fp8_ds_mla"),
              attention_config=NS(hisparse_config=object() if hisparse else None),
              quant_config=None)


@pytest.mark.parametrize("eager", [True,False])
def test_active_model_rejects_unqualified_orphan_hisparse(eager):
    with pytest.raises(NotImplementedError, match="HiSparse"):
        validate(configuration(hisparse=True,eager=eager))


@pytest.mark.parametrize("eager", [True,False])
def test_ordinary_target_admission_retains_eager_and_graph(eager):
    assert len(validate(configuration(eager=eager))) == 1


def test_core_moe_collects_without_core_root_environment():
    import os, subprocess, sys
    env=dict(os.environ)
    env.pop("HYV4_CORE033_ROOT",None)
    env.pop("VLLM_TARGET_ROOT",None)
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",HCU_ADAPTER_HOST="none",
               HIP_VISIBLE_DEVICES="",ROCR_VISIBLE_DEVICES="",CUDA_VISIBLE_DEVICES="")
    result=subprocess.run([sys.executable,"-m","pytest","--collect-only","-q",
         str(Path(__file__).with_name("test_hyv4_flash_core_moe.py"))],
         env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
