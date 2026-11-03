# HYV4 INT8 Quantization Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse INT8 activations throughout HYV4 and ship the optimization only when accuracy drops by at most 2 percentage points and controlled decode performance improves.

**Architecture:** Gate the optimization with the existing fused-RMS-quant environment switch and INT8 consumer detection. Produce one INT8 tensor/scale pair at each reusable boundary, then thread that pair through attention, indexer, dense MLP, and MoE APIs without changing disabled-path behavior. The final implementation deliberately keeps decoder RMSNorm unfused because decoder-level pairs enlarged graph capture and regressed throughput.

**Tech Stack:** Python, PyTorch, vLLM 0.25.1, LightOp HCU custom operators, pytest, vLLM benchmark serving.

**Spec:** `docs/superpowers/specs/2026-10-24-hyv4-int8-quant-reuse-design.md`

## Global Constraints

- Do not modify checkpoint files, LightOp, DeepGEMM configuration, or the installed Python environment.
- Preserve existing behavior unless `VLLM_HCU_USE_FUSED_RMS_QUANT` and HCU custom ops are enabled for INT8 consumers.
- HumanEval 0--63 must score at least 63/64 against the existing 64/64 baseline.
- Performance must use DP8, EP8, MTP3, DeepEP low latency, FP8 E4M3 KV cache, FULL_AND_PIECEWISE, input 1, output 1024, concurrency 128, 16 warmups, and 128 measured requests.
- Open a separate stacked PR only when controlled throughput improves or TPOT decreases.

## Review Focus

- Disabled fusion or a non-INT8 projection must execute the original RMSNorm and per-linear quantization path.
- BF16 activations updated by fused RMSNorm must remain correct for unquantized gate/indexer consumers.
- The q-LoRA pair must be reused only by projections consuming the same normalized `q_c` tensor.
- Chunked MoE execution must slice activation, quantized tensor, and scale on identical token boundaries.
- Graph capture and MTP must not retain stale quantized tensors between batches.

---

### Task 1: Attention quantized-input reuse

**Files:**
- Modify: `vllm_hcu/models/hy_v4/attention.py`
- Create: `tests/models/hy_v4/test_quant_reuse.py`

**Interfaces:**
- Produces: `HYV4MLAAttention.forward(..., x_and_scale_quanted: tuple[Tensor, Tensor] | None = None)`.
- Produces: `Indexer.forward/prepare_inputs(..., qr_and_scale_quanted: tuple[Tensor, Tensor] | None = None)`.
- Consumes: linear layer `_forward_with_hcu_quanted(input, pair)` and `FusedRMSNormAndQuant`.

- [ ] Write tests proving one input pair reaches both q-a and kv-a projections, one q-LoRA pair reaches q-b and indexer wq-b, and the disabled path supplies no pair.
- [ ] Run `pytest -q tests/models/hy_v4/test_quant_reuse.py -k attention` and verify the new tests fail because the optional interfaces are absent.
- [ ] Implement the optional pair plumbing and INT8-gated fused q-LoRA normalization with no behavior change when disabled.
- [ ] Run `pytest -q tests/models/hy_v4/test_quant_reuse.py -k attention` and verify all selected tests pass.
- [ ] Commit the attention implementation and tests.

### Task 2: Decoder and MoE quantized-input reuse

**Files:**
- Modify: `vllm_hcu/models/hy_v4/model.py`
- Modify: `vllm_hcu/models/hy_v4/moe.py`
- Modify: `tests/models/hy_v4/test_quant_reuse.py`

**Interfaces:**
- Consumes: Task 1 `HYV4MLAAttention.forward(..., x_and_scale_quanted=None)`.
- Produces: `HYV4FeedForward.forward(x, x_and_scale_quanted=None)` and `HYV4MoEFused.forward(hidden_states, x_and_scale_quanted=None)`.

- [ ] Write tests proving iHC fused norm updates BF16 input and forwards its pair, MoE forwards/slices the pair correctly, dense MLP consumes the pair, and fusion-off behavior is unchanged.
- [ ] Run `pytest -q tests/models/hy_v4/test_quant_reuse.py -k 'decoder or moe or feed_forward'` and verify failures identify the missing behavior.
- [ ] Implement INT8-gated fused decoder norms, dense fused activation quantization, and MoE pair forwarding/chunk slicing.
- [ ] Run the selected tests and the existing HYV4 model tests; verify they pass.
- [ ] Commit the decoder/MoE implementation and tests.

### Task 3: Kernel and repository verification

**Files:**
- Modify: `tests/accuracy/test_hcu_kernel_accuracy.py`
- Modify: `docs/hy4_w4a8_pcp.md` only if a launch flag or measured result requires documentation.

**Interfaces:**
- Consumes: Tasks 1--2 fused RMSNorm and reuse paths.
- Produces: reproducible kernel-accuracy and regression evidence.

- [ ] Add an HCU accuracy test comparing fused RMSNorm INT8 output, scale, and updated BF16 activation with the unfused reference for token counts 1, 16, 32, 64, and 128.
- [ ] Run the new test without the implementation commit checked out and verify it fails for the missing HYV4 integration contract, then restore the implementation and verify it passes.
- [ ] Run HYV4, runtime-patch, and accuracy suites, followed by the full repository test command; record exact pass/fail counts.
- [ ] Commit the accuracy test and any required launch documentation.

### Task 4: End-to-end accuracy and performance gates

**Files:**
- Create artifacts under `/model/hy4-pr264-validation-20261024/int4-quant-reuse/` (not committed).

**Interfaces:**
- Consumes: optimized branch and the unoptimized `30ba18d` baseline.
- Produces: HumanEval and controlled performance JSON/log evidence used for the delivery decision.

- [ ] Run baseline and optimized HumanEval cases 0--63 with identical deterministic settings; require optimized score at least 63/64.
- [ ] Run fresh baseline and optimized controlled serving benchmarks with the Global Constraints configuration.
- [ ] Compare throughput, TPOT, E2E, MTP acceptance, and per-card memory; require throughput improvement or TPOT reduction before delivery.
- [ ] If the gate fails, use systematic profiling to identify the next bottleneck and do not open a PR; if it passes, record the comparison report and commit any code changes required by findings through a new RED-GREEN cycle.

### Task 5: Review and independent PR

**Files:**
- Review all changes since `30ba18dd035ad425b74010c2c5316ca2b197f559`.

**Interfaces:**
- Consumes: Tasks 1--4 evidence.
- Produces: one pushed branch and one PR targeting `fix/hy4-w4a8-dp8-ep8-mtp3`.

- [ ] Build the final review package and run a fresh whole-branch code review; fix Critical/Important findings with RED-GREEN tests.
- [ ] Re-run the accuracy and performance gates after any review fix.
- [ ] Push `perf/hy4-w4a8-quant-reuse` using token authentication and open a separate PR against `fix/hy4-w4a8-dp8-ep8-mtp3`.
- [ ] Report commit, PR URL, exact test counts, HumanEval score, and before/after performance.
