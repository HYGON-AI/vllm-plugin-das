# LightOp Categorized API Adaptation for v0.25.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace production legacy LightOp/LMSlim kernel entry points on `v0.25.1` with LightOp 0.6 categorized APIs while preserving only ABI-compatible fallbacks and failing closed for changed ABIs.

**Architecture:** Migrate each call in its current owning module so lazy imports, runtime adapters, and plugin initialization order stay intact. Categorized modules are primary; deprecated top-level, `lightop.op`, `lightop.gemmopt`, or LMSlim paths remain only where signatures and data flow are compatible. DeepSeek V4 is reflowed so QR is normalized before projection and raw KV is normalized exactly once inside the new fused kernel.

**Tech Stack:** Python 3.10, PyTorch, vLLM v0.25.1 runtime adapters, LightOp 0.6 categorized modules, pytest, AST/source contract tests, optional HCU/ROCm kernel tests.

**Spec:** `docs/superpowers/specs/2026-08-25-lightop-categorized-api-v0251-design.md`

## Global Constraints

- Base all work on `origin/v0.25.1` commit `8c3d880b1b1f0b73ff7313a37d1a33b4693fc01b`.
- Preserve the current lazy/eager import boundary of each production owner.
- Catch only `ImportError` and `AttributeError` while selecting a compatibility path; never retry a failed kernel call through a legacy API.
- Retain legacy fallback only for ABI-compatible interfaces and log it with `logger.warning_once`.
- Require categorized APIs for Gemma RMSNorm, FP8 per-token quant, dynamic RMS quant, MoE align-out, and DeepSeek V4.
- Leave `lightop.sampling` unchanged.
- Pass all LightOp MQA weights as `weights.float().contiguous()`.
- Keep existing AITER, Triton, and PyTorch portable fallbacks outside the LightOp compatibility selection unchanged.
- Do not introduce a central LightOp facade.
- CPU/mock tests prove routing and ABI contracts only; report HCU numerical coverage separately.

## File Structure

### New tests

- `tests/runtime_patch/test_lightop_categorized_api.py`: installed categorized-export contract and static legacy-call inventory.
- `tests/runtime_patch/test_lightop_attention_api.py`: attention namespace selection, MQA ABI, weight layout, and compatible fallback contracts.
- `tests/runtime_patch/test_lightop_ops_api.py`: activation, norm, quant, Gemma, and concat routing contracts.
- `tests/runtime_patch/test_lightop_deepseek_v4_api.py`: DeepSeek V4 normalization and fused-kernel argument flow.

### Existing tests extended

- `tests/runtime_patch/test_moe_deepep.py`: EP gather/scatter, fused gate, and MoE align-out.
- `tests/runtime_patch/test_quant_gemm_aiter.py`: FP8/INT8 quant, channelwise GEMM, and Marlin categorized routing.
- `tests/patch/test_module_exchange.py`: whole-module replacement surface remains unchanged.
- `tests/accuracy/test_hcu_kernel_accuracy.py`: live HCU tests import the same categorized APIs as production.

### Production owners modified

- Attention: `vllm_hcu/model_executor/layers/attention_runtime.py`, `vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py`.
- Activation/GEMM: `vllm_hcu/ops/fuse_silu_mul_quant.py`, `vllm_hcu/ops/silu_and_mul.py`, and the three DeepGEMM expert modules.
- MoE: `deep_gemm_utils.py`, `router_runtime.py`, `ops/fuse_moe_gate.py`, `compressed_tensors_moe_marlin.py`, and `patch_moe_align_block_size.py`.
- Norm/quant: `ops/rms_norm.py`, `ops/gemma_rms_norm.py`, `ops/fuse_rms_norm_quant.py`, `lightop_fp8_runtime.py`, and `int8_runtime.py`.
- DeepSeek V4: `vllm_hcu/model_executor/layers/deepseek_v4_attention.py`.
- Tensor utility: `vllm_hcu/ops/test_concat.py`.

---

### Task 1: Lock the LightOp 0.6 categorized export contract

**Files:**
- Create: `tests/runtime_patch/test_lightop_categorized_api.py`

**Interfaces:**
- Consumes: installed `lightop.activation`, `attention`, `gemm_ops`, `moe`, `norm`, `quant`, and `tensor` modules.
- Produces: `REQUIRED_EXPORTS: dict[str, set[str]]`, used as the environment contract for every later task.

- [ ] **Step 1: Add the exact categorized-export test**

```python
from importlib import import_module

import pytest


REQUIRED_EXPORTS = {
    "lightop.activation": {
        "fuse_silu_mul_fp8_quant",
        "fuse_silu_mul_fp8_quant_ep",
        "fuse_silu_mul_per_token_quant",
        "fuse_silu_mul_quant",
        "fuse_silu_mul_quant_ep",
        "silu_and_mul_opt",
    },
    "lightop.attention": {
        "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32",
        "get_paged_mqa_logits_metadata",
        "mqa_logits",
        "paged_mqa_logits",
        "split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant",
        "top_k_per_row_decode",
        "top_k_per_row_prefill",
    },
    "lightop.gemm_ops": {
        "hipblaslt_w8a8_channelwise_gemm",
        "m_grouped_w8a8_gemm_nt_contig_asm",
        "m_grouped_w8a8_gemm_nt_masked",
    },
    "lightop.moe": {
        "ep_gather",
        "ep_scatter",
        "fused_experts_impl_fp8_marlin",
        "fused_experts_impl_int8_marlin",
        "moe_align_block_size_out",
        "moe_fused_gate",
    },
    "lightop.norm": {
        "fused_add_rms_norm",
        "gemma_fused_add_rmsnorm",
        "gemma_rmsnorm",
        "rms_norm_dynamic_per_token_quant",
        "rmsnorm_forward_autograd",
    },
    "lightop.quant": {"per_token_quant_fp8", "per_token_quant_int8"},
    "lightop.tensor": {"ds_cat"},
}


@pytest.mark.parametrize("module_name", sorted(REQUIRED_EXPORTS))
def test_categorized_lightop_exports(module_name: str) -> None:
    pytest.importorskip("lightop")
    module = import_module(module_name)
    missing = sorted(name for name in REQUIRED_EXPORTS[module_name]
                     if not hasattr(module, name))
    assert not missing, f"{module_name} is missing required exports: {missing}"
```

- [ ] **Step 2: Run the external API contract**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_categorized_api.py`

Expected: PASS with the installed LightOp 0.6 wheel. If it fails, stop because the implementation target is not installed.

- [ ] **Step 3: Commit the environment contract**

```bash
git add tests/runtime_patch/test_lightop_categorized_api.py
git commit -m "test: lock LightOp categorized API exports"
```

---

### Task 2: Migrate attention and sparse MLA APIs

**Files:**
- Create: `tests/runtime_patch/test_lightop_attention_api.py`
- Modify: `vllm_hcu/model_executor/layers/attention_runtime.py:190-225`
- Modify: `vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py:1-150,700-830,888-1065,1200-1280`

**Interfaces:**
- Consumes: `lightop.attention.mqa_logits(Q, K, Weights, ks, ke, kv_scale=None, clean_logit=True, D_out=None)` and `paged_mqa_logits(q, cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits=True)`.
- Produces: module-local `lightop_attention` with categorized methods; deprecated aggregate fallback uses top-level `mqa_logits`, `gemmopt.paged_mqa_logits`, and `op.top_k_per_row_*` only.

- [ ] **Step 1: Write failing attention source/ABI tests**

```python
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_attention_runtime_prefers_categorized_split_qkv() -> None:
    source = (REPO / "vllm_hcu/model_executor/layers/attention_runtime.py").read_text()
    assert "from lightop.attention import (" in source
    assert "Using deprecated top-level lightop split QKV" in source


def test_sparse_mla_uses_categorized_attention_and_new_mqa_abi() -> None:
    source = (REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py").read_text()
    assert "from lightop import attention as lightop_attention" in source
    assert "lightop_attention.mqa_logits(" in source
    assert "lightop_attention.paged_mqa_logits(" in source
    assert "lightop_attention.top_k_per_row_prefill" in source
    assert "lightop_attention.top_k_per_row_decode" in source
    assert "q_slice.shape[0], # logical lengths" not in source


def test_every_lightop_mqa_weight_is_fp32_contiguous() -> None:
    source = (REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py").read_text()
    assert "weights_slice.float().contiguous()" in source
    assert "weights.float().contiguous()" in source
```

- [ ] **Step 2: Run the tests and confirm the legacy imports fail them**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_attention_api.py`

Expected: FAIL because production still imports top-level `lightop`, `op`, and `gemmopt`.

- [ ] **Step 3: Add categorized-first split-QKV selection**

Use the existing lazy call boundary in `attention_runtime.py`:

```python
try:
    from lightop.attention import (
        split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant,
    )
except (ImportError, AttributeError):
    from lightop import (
        split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant,
    )

    logger.warning_once(
        "Using deprecated top-level lightop split QKV API because "
        "lightop.attention is unavailable; upgrade LightOp."
    )
```

Add `init_logger` and `logger = init_logger(__name__)` if the file does not already own a logger. Preserve its existing required-LightOp `RuntimeError` if neither import succeeds.

- [ ] **Step 4: Replace the sparse MLA aggregate import**

```python
from types import SimpleNamespace

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from lightop import attention as lightop_attention
except (ImportError, AttributeError):
    from lightop import gemmopt as _legacy_gemmopt
    from lightop import mqa_logits as _legacy_mqa_logits
    from lightop import op as _legacy_op

    lightop_attention = SimpleNamespace(
        mqa_logits=_legacy_mqa_logits,
        paged_mqa_logits=_legacy_gemmopt.paged_mqa_logits,
        top_k_per_row_decode=_legacy_op.top_k_per_row_decode,
        top_k_per_row_prefill=_legacy_op.top_k_per_row_prefill,
    )
    logger.warning_once(
        "Using deprecated top-level lightop, lightop.op, and "
        "lightop.gemmopt attention APIs because lightop.attention is "
        "unavailable; upgrade LightOp."
    )
```

Replace all sparse MLA calls with `lightop_attention.*`. In the chunked call, remove the four explicit logical-size arguments and call:

```python
lightop_attention.mqa_logits(
    q_slice,
    k_fp8,
    weights_slice.float().contiguous(),
    ks_slice,
    ke_slice,
    chunk_k_scale,
    True,
    logits_slice_view,
)
```

For non-paged and paged calls, pass `weights.float().contiguous()` immediately before invoking LightOp. Do not change AITER or PyTorch branches.

- [ ] **Step 5: Run focused attention tests**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_attention_api.py tests/runtime_patch/test_sparse_indexer_loading.py`

Expected: PASS.

- [ ] **Step 6: Commit attention migration**

```bash
git add tests/runtime_patch/test_lightop_attention_api.py \
  tests/runtime_patch/test_sparse_indexer_loading.py \
  vllm_hcu/model_executor/layers/attention_runtime.py \
  vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py
git commit -m "refactor: use categorized LightOp attention APIs"
```

---

### Task 3: Migrate activation and grouped GEMM APIs

**Files:**
- Create: `tests/runtime_patch/test_lightop_ops_api.py`
- Modify: `vllm_hcu/ops/fuse_silu_mul_quant.py`
- Modify: `vllm_hcu/ops/silu_and_mul.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:410-485`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py:455-510`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py:45-60`
- Test: `tests/patch/test_module_exchange.py`

**Interfaces:**
- Consumes: activation exports from `lightop.activation` and grouped INT8 GEMMs from `lightop.gemm_ops`.
- Produces: unchanged expert class and function signatures; deprecated top-level LightOp exports remain compatible fallbacks with one warning per module.

- [ ] **Step 1: Add failing categorized activation/GEMM source contracts**

Append to `test_lightop_ops_api.py`:

```python
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def test_activation_operators_prefer_lightop_activation() -> None:
    assert "from lightop.activation import" in _source(
        "vllm_hcu/ops/fuse_silu_mul_quant.py"
    )
    assert "from lightop import activation as op" in _source(
        "vllm_hcu/ops/silu_and_mul.py"
    )


def test_deep_gemm_experts_use_categorized_activation_and_gemm() -> None:
    for relative in (
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
        "vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py",
        "vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py",
    ):
        source = _source(relative)
        assert "lightop.activation" in source
    assert "lightop.gemm_ops" in _source(
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
    )
    assert "lightop.gemm_ops" in _source(
        "vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py"
    )
```

- [ ] **Step 2: Run the tests and confirm they fail on top-level imports**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_ops_api.py -k 'activation or deep_gemm'`

Expected: FAIL.

- [ ] **Step 3: Migrate the two operator modules**

In `fuse_silu_mul_quant.py`, keep the import inside the real implementation:

```python
try:
    from lightop.activation import (
        fuse_silu_mul_per_token_quant as fuse_silu_mul_quant_lightop,
    )
except (ImportError, AttributeError):
    from lightop import (
        fuse_silu_mul_per_token_quant as fuse_silu_mul_quant_lightop,
    )
    logger.warning_once(
        "Using deprecated lightop.fuse_silu_mul_per_token_quant because "
        "lightop.activation is unavailable; upgrade LightOp."
    )
```

In `silu_and_mul.py`, select the module without changing the custom-op ABI:

```python
try:
    from lightop import activation as op
except (ImportError, AttributeError):
    import lightop.op as op
    logger.warning_once(
        "Using deprecated lightop.op activation API because "
        "lightop.activation is unavailable; upgrade LightOp."
    )
```

- [ ] **Step 4: Migrate all three expert modules at their current import sites**

For contiguous INT8 experts import:

```python
try:
    from lightop.activation import fuse_silu_mul_quant
    from lightop.gemm_ops import m_grouped_w8a8_gemm_nt_contig_asm
except (ImportError, AttributeError):
    from lightop import (
        fuse_silu_mul_quant,
        m_grouped_w8a8_gemm_nt_contig_asm,
    )
    logger.warning_once(
        "Using deprecated top-level LightOp activation/GEMM APIs; "
        "upgrade LightOp."
    )
```

For masked experts use `fuse_silu_mul_quant_ep` and `m_grouped_w8a8_gemm_nt_masked`. For FP8 branches use `fuse_silu_mul_fp8_quant` or `_ep` from `lightop.activation`. In `dpsk_v4_deep_gemm_moe.py`, keep the current module-level import timing and add only the categorized-first fallback block.

- [ ] **Step 5: Run focused and replacement-surface tests**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_ops_api.py tests/patch/test_module_exchange.py`

Expected: PASS, including unchanged whole-module signatures.

- [ ] **Step 6: Commit activation/GEMM migration**

```bash
git add tests/runtime_patch/test_lightop_ops_api.py \
  tests/patch/test_module_exchange.py \
  vllm_hcu/ops/fuse_silu_mul_quant.py \
  vllm_hcu/ops/silu_and_mul.py \
  vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py \
  vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py \
  vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py
git commit -m "refactor: use categorized LightOp activation and GEMM APIs"
```

---

### Task 4: Migrate MoE routing, EP, align-out, and Marlin APIs

**Files:**
- Modify: `tests/runtime_patch/test_moe_deepep.py:1278-1420`
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py:330-510`
- Modify: `vllm_hcu/model_executor/layers/fused_moe/router_runtime.py:85-110`
- Modify: `vllm_hcu/ops/fuse_moe_gate.py`
- Modify: `vllm_hcu/patch/worker/op_opt/moe/patch_moe_align_block_size.py:105-150`
- Modify: `vllm_hcu/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py:330-360,740-775`

**Interfaces:**
- Consumes: `lightop.moe.ep_scatter`, `ep_gather`, `moe_fused_gate`, `moe_align_block_size_out`, and Marlin expert implementations.
- Produces: unchanged EP/MoE public contracts. `moe_align_block_size_out` is strict and receives preallocated outputs plus `is_ep=False, is_fuse_fill=False`.

- [ ] **Step 1: Change the MoE tests to install categorized fakes first**

Add a fake package helper and strict align assertion:

```python
def _install_lightop_moe(monkeypatch, **exports):
    lightop = _module("lightop")
    lightop.__path__ = []
    moe = _module("lightop.moe", **exports)
    lightop.moe = moe
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.moe", moe)
    return moe


def moe_align_block_size_out(*args, **kwargs):
    calls.append((args, kwargs))

# After invoking the patched adapter:
assert calls[0][0][3:6] == (sorted_ids, expert_ids, num_tokens_post_pad)
assert calls[0][1] == {"is_ep": False, "is_fuse_fill": False}
```

Add a negative case that installs only `lightop.op.moe_align_block_size` and expects `RuntimeError` naming `lightop.moe.moe_align_block_size_out`.

- [ ] **Step 2: Run focused MoE tests and confirm they fail**

Run: `python -m pytest -q tests/runtime_patch/test_moe_deepep.py -k 'lightop or moe_align or router'`

Expected: FAIL because production still resolves `lightop.op` and the old align name.

- [ ] **Step 3: Migrate compatible EP and fused-gate calls**

At each current lazy boundary, use:

```python
try:
    from lightop import moe as lightop_moe
except (ImportError, AttributeError):
    from lightop import op as lightop_moe
    logger.warning_once(
        "Using deprecated lightop.op MoE APIs because lightop.moe is "
        "unavailable; upgrade LightOp."
    )
```

Call `lightop_moe.ep_scatter`, `ep_gather`, and `moe_fused_gate`. Preserve the existing feature flags and Triton fallback in `deep_gemm_utils.py`.

- [ ] **Step 4: Make MoE align-out strict**

Replace the old import and call with:

```python
try:
    from lightop.moe import moe_align_block_size_out
except (ImportError, AttributeError) as exc:
    raise RuntimeError(
        "VLLM_HCU_USE_LIGHTOP_MOE_ALIGN requires "
        "lightop.moe.moe_align_block_size_out; upgrade LightOp"
    ) from exc

moe_align_block_size_out(
    topk_ids,
    num_experts,
    block_size,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
    expert_map if ignore_invalid_experts else None,
    None,
    None,
    is_ep=False,
    is_fuse_fill=False,
)
```

Do not catch `TypeError` from the kernel call and do not retry the legacy function.

- [ ] **Step 5: Migrate Marlin implementations with LMSlim fallback**

At both existing lazy call sites:

```python
try:
    from lightop.moe import fused_experts_impl_fp8_marlin
except (ImportError, AttributeError):
    from lmslim.layers.fused_moe.fuse_moe_fp8_marlin import (
        fused_experts_impl_fp8_marlin,
    )
    logger.warning_once(
        "Using deprecated LMSlim FP8 Marlin API because lightop.moe is "
        "unavailable; upgrade LightOp."
    )
```

Use the corresponding INT8 symbol and LMSlim module in the second path. Do not change call keywords or output handling.

- [ ] **Step 6: Run MoE and quant routing tests**

Run: `python -m pytest -q tests/runtime_patch/test_moe_deepep.py tests/runtime_patch/test_quant_gemm_aiter.py -k 'lightop or moe_align or marlin or int8'`

Expected: PASS.

- [ ] **Step 7: Commit MoE migration**

```bash
git add tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py \
  vllm_hcu/model_executor/layers/fused_moe/router_runtime.py \
  vllm_hcu/ops/fuse_moe_gate.py \
  vllm_hcu/patch/worker/op_opt/moe/patch_moe_align_block_size.py \
  vllm_hcu/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py
git commit -m "refactor: use categorized LightOp MoE APIs"
```

---

### Task 5: Migrate norm, FP8/INT8 quant, and channelwise GEMM APIs

**Files:**
- Modify: `tests/runtime_patch/test_lightop_ops_api.py`
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py:3850-3940`
- Modify: `tests/accuracy/test_portable_operator_accuracy.py`
- Modify: `vllm_hcu/ops/rms_norm.py`
- Modify: `vllm_hcu/ops/gemma_rms_norm.py`
- Modify: `vllm_hcu/ops/fuse_rms_norm_quant.py:55-80`
- Modify: `vllm_hcu/model_executor/layers/quantization/lightop_fp8_runtime.py:20-45`
- Modify: `vllm_hcu/model_executor/layers/quantization/int8_runtime.py:100-180`

**Interfaces:**
- Consumes: `lightop.norm.*`, `lightop.quant.per_token_quant_fp8`, `per_token_quant_int8`, and `lightop.gemm_ops.hipblaslt_w8a8_channelwise_gemm`.
- Produces: unchanged vLLM custom-op and linear adapter APIs. Strict calls use categorized argument/return contracts; compatible norm and LMSlim fallbacks warn once.

- [ ] **Step 1: Add failing strict-ABI tests**

Add tests that record arguments rather than executing HCU kernels:

```python
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization import lightop_fp8_runtime
from vllm_hcu.ops.fuse_rms_norm_quant import fused_rmsquant_impl


def _install_lightop_submodule(monkeypatch, name, **exports):
    lightop = ModuleType("lightop")
    lightop.__path__ = []
    submodule = ModuleType(f"lightop.{name}")
    submodule.__dict__.update(exports)
    setattr(lightop, name, submodule)
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, f"lightop.{name}", submodule)
    return submodule


def test_fp8_quant_uses_categorized_output_keywords(monkeypatch) -> None:
    calls = []

    def per_token_quant_fp8(x, *, dtype, out_q, out_scale):
        calls.append((x, dtype, out_q, out_scale))
        return out_q, out_scale

    _install_lightop_submodule(monkeypatch, "quant", per_token_quant_fp8=per_token_quant_fp8)
    monkeypatch.setattr(
        lightop_fp8_runtime, "_FP8_DTYPE", torch.float8_e4m3fn
    )
    x = torch.ones((2, 8), dtype=torch.bfloat16)
    output, scale = lightop_fp8_runtime._lightop_per_token_quant_fp8(x)
    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is torch.float8_e4m3fn
    assert calls[0][2] is output
    assert calls[0][3] is scale


def test_fp8_quant_rejects_legacy_output_first_api(monkeypatch) -> None:
    lightop = ModuleType("lightop")
    lightop.__path__ = []
    lightop.op = SimpleNamespace(per_token_quant_fp8=lambda out, x, scale: None)
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.delitem(sys.modules, "lightop.quant", raising=False)
    monkeypatch.setattr(
        lightop_fp8_runtime, "_FP8_DTYPE", torch.float8_e4m3fn
    )
    with pytest.raises(
        lightop_fp8_runtime.HcuLightOpRegistrationError,
        match="lightop.quant.per_token_quant_fp8",
    ):
        lightop_fp8_runtime._lightop_per_token_quant_fp8(
            torch.ones((2, 8), dtype=torch.bfloat16)
        )


def test_dynamic_rms_quant_consumes_returned_tensors(monkeypatch) -> None:
    expected_q = torch.ones((2, 4), dtype=torch.int8)
    expected_s = torch.ones((2, 1), dtype=torch.float32)
    _install_lightop_submodule(
        monkeypatch,
        "norm",
        rms_norm_dynamic_per_token_quant=lambda *args, **kwargs: (
            expected_q,
            expected_s,
        ),
    )
    actual_q, actual_s = fused_rmsquant_impl(
        torch.ones((2, 4)), torch.ones(4), 1e-6, torch.int8
    )
    assert actual_q is expected_q
    assert actual_s is expected_s


def _load_gemma_forward(gemma_rmsnorm):
    import ast
    import copy
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] /
              "vllm_hcu/ops/gemma_rms_norm.py").read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef)
               and node.name == "HcuGemmaRMSNorm")
    method = copy.deepcopy(next(node for node in cls.body
                                if isinstance(node, ast.FunctionDef)
                                and node.name == "forward_hip"))
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "henvs": SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=True,
            VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=True,
        ),
        "gemma_rmsnorm": gemma_rmsnorm,
        "gemma_fused_add_rmsnorm": lambda *args: None,
    }
    exec(compile(module, "gemma_forward_contract", "exec"), namespace)
    return namespace["forward_hip"]


def test_gemma_rmsnorm_uses_new_out_keyword() -> None:
    calls = []

    def gemma_rmsnorm(x, weight, epsilon, *, out):
        calls.append((x, weight, epsilon, out))

    forward = _load_gemma_forward(gemma_rmsnorm)
    owner = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    x = torch.ones((2, 4))
    actual = forward(owner, x)
    assert calls[0][0] is x
    assert calls[0][1] is owner.weight
    assert calls[0][2] == owner.variance_epsilon
    assert calls[0][3] is actual


def test_gemma_rmsnorm_rejects_legacy_only_installation() -> None:
    forward = _load_gemma_forward(None)
    owner = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    with pytest.raises(RuntimeError, match="lightop.norm.gemma_rmsnorm"):
        forward(owner, torch.ones((2, 4)))
```

- [ ] **Step 2: Run strict-ABI tests and confirm they fail**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_ops_api.py tests/runtime_patch/test_quant_gemm_aiter.py -k 'fp8_quant or dynamic_rms or gemma or int8_linear'`

Expected: FAIL on output-first legacy calls and LMSlim-first routing.

- [ ] **Step 3: Migrate compatible norm functions and strict Gemma RMSNorm**

In `rms_norm.py`:

```python
try:
    from lightop.norm import fused_add_rms_norm, rmsnorm_forward_autograd
except (ImportError, AttributeError):
    from lightop import fused_add_rms_norm
    from lightop.op import rmsnorm_forward_autograd
    logger.warning_once(
        "Using deprecated top-level lightop and lightop.op RMSNorm APIs "
        "because lightop.norm is unavailable; upgrade LightOp."
    )
```

In `gemma_rms_norm.py`, resolve `gemma_fused_add_rmsnorm` with compatible fallback, but set `gemma_rmsnorm = None` when the categorized import is absent. The strict call is:

```python
if gemma_rmsnorm is None:
    raise RuntimeError(
        "lightop.norm.gemma_rmsnorm is required because its ABI differs "
        "from lightop.op.gemma_rmsnorm; upgrade LightOp"
    )
gemma_rmsnorm(x, self.weight, self.variance_epsilon, out=out)
```

- [ ] **Step 4: Adapt changed quant ABIs**

In `lightop_fp8_runtime.py`:

```python
try:
    from lightop.quant import per_token_quant_fp8
except (ImportError, AttributeError) as exc:
    raise HcuLightOpRegistrationError(
        "lightop.quant.per_token_quant_fp8 is required; upgrade LightOp"
    ) from exc

try:
    per_token_quant_fp8(
        x,
        dtype=_FP8_DTYPE,
        out_q=out,
        out_scale=scale,
    )
except Exception as exc:
    raise HcuLightOpRegistrationError(
        "LightOp per_token_quant_fp8 kernel execution failed"
    ) from exc
```

Keep the adapter's existing execution-error translation but do not retry a legacy kernel. In `fuse_rms_norm_quant.py`, remove preallocation from the real implementation and return:

```python
from lightop.norm import rms_norm_dynamic_per_token_quant

return rms_norm_dynamic_per_token_quant(
    input,
    weight,
    epsilon,
    quant_dtype,
    residual=residual,
    update_input=bool(update_input),
)
```

Translate only missing-import errors; allow kernel errors to propagate.

- [ ] **Step 5: Make LightOp primary for W8A8 quant and GEMM**

In `int8_runtime.py`, retain the existing lazy boundaries:

```python
try:
    from lightop.quant import per_token_quant_int8
except (ImportError, AttributeError):
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8
    logger.warning_once(
        "Using deprecated LMSlim per-token INT8 quantization because "
        "lightop.quant is unavailable; upgrade LightOp."
    )

try:
    from lightop.gemm_ops import hipblaslt_w8a8_channelwise_gemm
except (ImportError, AttributeError):
    from lmslim import quant_ops
    hipblaslt_w8a8_channelwise_gemm = quant_ops.hipblaslt_w8a8_gemm
    logger.warning_once(
        "Using deprecated LMSlim W8A8 GEMM because lightop.gemm_ops is "
        "unavailable; upgrade LightOp."
    )
```

Keep the current `(status, output)` validation, dimensional arguments, layout `"NT"`, and bias handling.

Update the portable numerical test helper so its primary path is a fake
categorized package rather than the LMSlim fallback:

```python
def _install_fake_lightop_quant_gemm(monkeypatch):
    def per_token_quant_int8(value):
        absmax = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
        scale = absmax / 127.0
        quantized = torch.round(value.float() / scale).clamp(-127, 127).to(torch.int8)
        return quantized, scale.to(torch.float32)

    def hipblaslt_w8a8_gemm(
        activation, weight, activation_scale, weight_scale,
        m, n, k, layout, output_dtype,
    ):
        assert activation.shape == (m, k)
        assert weight.shape == (n, k)
        assert layout == "NT"
        output = (activation.float() * activation_scale) @ (
            weight.float() * weight_scale
        ).t()
        return True, output.to(output_dtype)

    lightop = _module("lightop")
    lightop.__path__ = []
    quant = _module(
        "lightop.quant",
        per_token_quant_int8=per_token_quant_int8,
    )
    gemm_ops = _module(
        "lightop.gemm_ops",
        hipblaslt_w8a8_channelwise_gemm=hipblaslt_w8a8_gemm,
    )
    lightop.quant = quant
    lightop.gemm_ops = gemm_ops
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.quant", quant)
    monkeypatch.setitem(sys.modules, "lightop.gemm_ops", gemm_ops)
```

Retain one explicit LMSlim fallback test by making imports of
`lightop.quant` and `lightop.gemm_ops` raise `ImportError`, installing the
existing LMSlim fakes, and asserting one deprecation warning plus the same
numerical result.

- [ ] **Step 6: Run focused routing and portable numerical tests**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_ops_api.py tests/runtime_patch/test_quant_gemm_aiter.py tests/accuracy/test_portable_operator_accuracy.py -k 'lightop or rms or gemma or int8 or w8a8'`

Expected: PASS.

- [ ] **Step 7: Commit norm/quant migration**

```bash
git add tests/runtime_patch/test_lightop_ops_api.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/accuracy/test_portable_operator_accuracy.py \
  vllm_hcu/ops/rms_norm.py \
  vllm_hcu/ops/gemma_rms_norm.py \
  vllm_hcu/ops/fuse_rms_norm_quant.py \
  vllm_hcu/model_executor/layers/quantization/lightop_fp8_runtime.py \
  vllm_hcu/model_executor/layers/quantization/int8_runtime.py
git commit -m "refactor: adapt categorized LightOp norm and quant APIs"
```

---

### Task 6: Reflow DeepSeek V4 for the categorized fused KVNorm kernel

**Files:**
- Create: `tests/runtime_patch/test_lightop_deepseek_v4_api.py`
- Modify: `vllm_hcu/model_executor/layers/deepseek_v4_attention.py:20-55,450-470,575-615`
- Test: `tests/patch/test_module_exchange.py:412-470`

**Interfaces:**
- Consumes: `lightop.attention.fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(q, raw_kv, kv_norm_weight, cache, slot_mapping, positions, cos_sin_cache, epsilon, block_size)`.
- Produces: QR normalized by `self.q_norm`; raw KV passed to the fused categorized kernel; KV is normalized exactly once inside that kernel.

- [ ] **Step 1: Add failing AST/data-flow tests**

```python
import ast
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] /
          "vllm_hcu/model_executor/layers/deepseek_v4_attention.py").read_text()


def test_deepseek_v4_normalizes_qr_only_before_projection() -> None:
    assert "qr = self.q_norm(qr)" in SOURCE
    assert "qr, kv = fused_q_kv_rmsnorm(" not in SOURCE


def test_deepseek_v4_uses_strict_categorized_kvnorm_insert() -> None:
    assert "from lightop.attention import (" in SOURCE
    assert "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32" in SOURCE
    assert "lightop.op.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert" not in SOURCE
    assert "self.kv_norm.weight.data" in SOURCE
    assert "swa_metadata.slot_mapping" in SOURCE
    assert "positions.to(torch.int64)" in SOURCE


def test_deepseek_v4_fused_kernel_argument_order() -> None:
    tree = ast.parse(SOURCE)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32"
    )
    assert [ast.unparse(arg) for arg in call.args] == [
        "q",
        "kv",
        "self.kv_norm.weight.data",
        "swa_kv_cache_2d",
        "swa_metadata.slot_mapping",
        "positions.to(torch.int64)",
        "self.rotary_emb.cos_sin_cache",
        "self.eps",
        "swa_metadata.block_size",
    ]
```

- [ ] **Step 2: Run tests and confirm old two-stage flow fails**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_deepseek_v4_api.py`

Expected: FAIL because `fused_q_kv_rmsnorm` and the legacy kernel are still present.

- [ ] **Step 3: Change QR/KV normalization ownership**

Replace:

```python
qr, kv = fused_q_kv_rmsnorm(
    qr, kv, self.q_norm.weight.data, self.kv_norm.weight.data, self.eps
)
```

with:

```python
qr = self.q_norm(qr)
```

Remove the now-unused `fused_q_kv_rmsnorm` import. Do not mutate or clone `kv` before the fused cache-insert call.

- [ ] **Step 4: Require and call the categorized DeepSeek kernel**

Inside `_fused_qnorm_rope_kv_insert`, retain the current lazy boundary:

```python
try:
    from lightop.attention import (
        fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32,
    )
except (ImportError, AttributeError) as exc:
    raise RuntimeError(
        "DeepSeek V4 requires lightop.attention."
        "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32; "
        "upgrade LightOp"
    ) from exc

fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(
    q,
    kv,
    self.kv_norm.weight.data,
    swa_kv_cache_2d,
    swa_metadata.slot_mapping,
    positions.to(torch.int64),
    self.rotary_emb.cos_sin_cache,
    self.eps,
    swa_metadata.block_size,
)
```

Do not add a legacy kernel fallback.

- [ ] **Step 5: Run DeepSeek and replacement ownership tests**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_deepseek_v4_api.py tests/patch/test_module_exchange.py -k 'deepseek_v4 or lightop'`

Expected: PASS.

- [ ] **Step 6: Commit DeepSeek V4 reflow**

```bash
git add tests/runtime_patch/test_lightop_deepseek_v4_api.py \
  tests/patch/test_module_exchange.py \
  vllm_hcu/model_executor/layers/deepseek_v4_attention.py
git commit -m "refactor: adapt DeepSeek V4 categorized LightOp kernel"
```

---

### Task 7: Add tensor fallback and align live HCU accuracy imports

**Files:**
- Modify: `tests/runtime_patch/test_lightop_ops_api.py`
- Modify: `tests/accuracy/test_hcu_kernel_accuracy.py:20-370`
- Modify: `vllm_hcu/ops/test_concat.py:1-30,190-205`

**Interfaces:**
- Consumes: `lightop.tensor.ds_cat(input_A, input_B, output, mode)` and categorized activation/norm/quant exports.
- Produces: concat helpers that always return a valid tensor through new API, compatible legacy API, or `torch.cat`.

- [ ] **Step 1: Add failing concat fallback tests**

```python
import torch

from vllm_hcu.ops import test_concat


def test_decode_concat_falls_back_to_torch_cat(monkeypatch) -> None:
    monkeypatch.setattr(test_concat, "ds_cat", None)
    left = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    right = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    torch.testing.assert_close(
        test_concat.concat_helper_decode(left, right, dim=2),
        torch.cat((left, right), dim=2),
    )


def test_prefill_concat_falls_back_to_torch_cat(monkeypatch) -> None:
    monkeypatch.setattr(test_concat, "ds_cat", None)
    left = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    right = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    torch.testing.assert_close(
        test_concat.lightop_concat_prefill_helper(left, right, dim=2),
        torch.cat((left, right), dim=2),
    )
```

- [ ] **Step 2: Run tests and confirm the unresolved-callable failure**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_ops_api.py -k concat`

Expected: FAIL because the current import failure leaves `ds_cat` undefined and helpers do not consistently use `torch.cat`.

- [ ] **Step 3: Implement the three-level concat resolution**

```python
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from lightop.tensor import ds_cat
except (ImportError, AttributeError):
    try:
        from lightop import ds_cat
    except (ImportError, AttributeError):
        ds_cat = None
        logger.warning_once(
            "LightOp ds_cat is unavailable; using torch.cat."
        )
    else:
        logger.warning_once(
            "Using deprecated top-level lightop.ds_cat because "
            "lightop.tensor is unavailable; upgrade LightOp."
        )
```

In both concat helpers, return `torch.cat((A, B), dim=dim)` immediately when `ds_cat is None`; otherwise preserve the existing output allocation and `mode` mapping.

- [ ] **Step 4: Migrate live HCU test imports**

Update `test_hcu_kernel_accuracy.py` to import the same symbols production uses:

```python
from lightop.activation import silu_and_mul_opt
from lightop.norm import (
    fused_add_rms_norm,
    gemma_rmsnorm,
    rms_norm_dynamic_per_token_quant,
    rmsnorm_forward_autograd,
)
from lightop.quant import per_token_quant_fp8
```

Adapt test calls to the categorized signatures, including `gemma_rmsnorm(..., out=actual)`, returned tensors from dynamic RMS quant, and `per_token_quant_fp8(..., out_q=actual, out_scale=scale)`. Keep `pytest.mark.hcu` and numerical tolerances unchanged.

- [ ] **Step 5: Run portable tests and collect HCU tests**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_ops_api.py -k concat`

Expected: PASS.

Run: `python -m pytest -q --collect-only tests/accuracy/test_hcu_kernel_accuracy.py`

Expected: collection succeeds without executing HCU kernels.

- [ ] **Step 6: Commit tensor and accuracy-import migration**

```bash
git add tests/runtime_patch/test_lightop_ops_api.py \
  tests/accuracy/test_hcu_kernel_accuracy.py \
  vllm_hcu/ops/test_concat.py
git commit -m "refactor: use categorized LightOp tensor APIs"
```

---

### Task 8: Audit residual legacy calls and run final verification

**Files:**
- Modify: `tests/runtime_patch/test_lightop_categorized_api.py`

**Interfaces:**
- Consumes: all categorized migrations produced by Tasks 2-7.
- Produces: an explicit allowlist for deprecated compatibility blocks and final test evidence for the pull request.

- [ ] **Step 1: Add a static production inventory test**

```python
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PRODUCTION = REPO / "vllm_hcu"

ALLOWED_LEGACY_FILES = {
    "model_executor/layers/attention_runtime.py",
    "model_executor/layers/fused_moe/deep_gemm_utils.py",
    "model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py",
    "model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
    "model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py",
    "model_executor/layers/fused_moe/router_runtime.py",
    "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe_marlin.py",
    "model_executor/layers/quantization/int8_runtime.py",
    "ops/fuse_moe_gate.py",
    "ops/fuse_silu_mul_quant.py",
    "ops/gemma_rms_norm.py",
    "ops/rms_norm.py",
    "ops/silu_and_mul.py",
    "ops/test_concat.py",
    "v1/attention/ops/rocm_aiter_mla_sparse.py",
}


def test_legacy_lightop_references_are_confined_to_compatible_fallbacks() -> None:
    offenders = set()
    for path in PRODUCTION.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in (
            "lightop.op", "lightop.gemmopt", "from lmslim", "import lmslim"
        )):
            relative = path.relative_to(PRODUCTION).as_posix()
            if relative not in ALLOWED_LEGACY_FILES:
                offenders.add(relative)
    assert not offenders, sorted(offenders)
```

Remove an allowlist entry when the corresponding file no longer contains a legacy compatibility block. `lightop.sampling` is not a legacy token and needs no exception.

- [ ] **Step 2: Run the categorized contract and legacy inventory**

Run: `python -m pytest -q tests/runtime_patch/test_lightop_categorized_api.py`

Expected: PASS.

- [ ] **Step 3: Run whitespace and syntax checks**

Run: `git diff --check origin/v0.25.1...HEAD`

Expected: no output.

Run: `python -m compileall -q vllm_hcu tests/runtime_patch`

Expected: exit code 0.

- [ ] **Step 4: Run all focused LightOp tests together**

Run:

```bash
python -m pytest -q \
  tests/runtime_patch/test_lightop_categorized_api.py \
  tests/runtime_patch/test_lightop_attention_api.py \
  tests/runtime_patch/test_lightop_ops_api.py \
  tests/runtime_patch/test_lightop_deepseek_v4_api.py \
  tests/runtime_patch/test_sparse_indexer_loading.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/patch/test_module_exchange.py
```

Expected: PASS.

- [ ] **Step 5: Run the full portable contract suite**

Run: `python tools/run_patch_tests.py --suite contract`

Expected: all tests pass; compare against the pre-change baseline of 947 passed, 37 deselected, and 14 warnings.

- [ ] **Step 6: Run or explicitly classify live HCU validation**

If a live HCU/ROCm device is available, run:

```bash
python tools/run_patch_tests.py --suite accuracy-hcu -- -k lightop
```

Expected: categorized LightOp kernel numerical tests pass. If the device is unavailable, record the tests as not run due to hardware absence; do not call them passing.

- [ ] **Step 7: Review the complete diff and residual imports**

Run:

```bash
git diff --stat origin/v0.25.1...HEAD
rg -n --glob '*.py' \
  'lightop\.op|lightop\.gemmopt|from lmslim|import lmslim|from lightop import' \
  vllm_hcu tests
```

Expected: every production result is either a documented ABI-compatible fallback or unchanged `lightop.sampling`; no strict changed-ABI path references its legacy symbol.

- [ ] **Step 8: Commit the final audit test**

```bash
git add tests/runtime_patch/test_lightop_categorized_api.py
git commit -m "test: verify categorized LightOp migration"
```

- [ ] **Step 9: Prepare the pull request evidence**

Use branch `feat/lightop-categorized-api-v0251`, push it, and create a pull request targeting `v0.25.1`. The description must include:

```markdown
## Summary
- migrate LightOp calls to categorized 0.6 APIs
- retain warning-backed fallbacks only for compatible ABIs
- adapt strict quant, MoE align, and DeepSeek V4 data flows

## References
- OpenDAS/vllm-hcu v0.21.0 commit 059fc449
- MQA weight-layout follow-up 1d4a0d7

## Verification
- focused LightOp contract tests: PASS; include the exact pytest summary from Step 4
- portable contract suite: PASS; include the exact runner summary from Step 5
- HCU accuracy tests: include the exact Step 6 summary, or state that no live HCU was available
```

Do not merge until required checks are green and repository review policy permits it.
