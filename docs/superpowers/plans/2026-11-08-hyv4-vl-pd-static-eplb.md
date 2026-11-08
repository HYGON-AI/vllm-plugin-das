# HYV4-VL PD and Static EPLB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the supplied W4A8 image checkpoint with real P PCP16 / D DP16 separation and fresh offline P EPLB, validating correctness before throughput.

**Architecture:** Extend native MRV2 model state, prepare multimodal inputs globally and gather them through the existing PCP token map. Retain canonical KV positions and generic static loaders; add narrow architecture gates and explicit vision memory profiling.

**Tech Stack:** Python, PyTorch/DTK, fixed vLLM752a3a5, HCU plugin v0.25.1 + MR338, DeepEP HT/LL, Mooncake RDMA.

**Spec:** `docs/superpowers/specs/2026-11-08-hyv4-vl-pd-static-eplb.md`

## Global Constraints

- Model `/data_hy4_w4a8/Hy4-testing-Channel-INT4-w4a8`; preserve its configuration and weights.
- P PP1/TP1/PCP16/DP1/DCP1/EP16, HT/DeepGEMM, eager; D PP1/TP1/PCP1/DP16/EP16, LL; no MTP.
- Images only; preserve unrelated topology/model/unsupported-input rejections.
- No frozen-core edits, routing simulation, old text map reuse, arbitrary profiling bypass, formal KV priming or intrusive formal counters.
- Reuse the existing independent local source checkout on branch `feat/hyv4-vl-pd-static-eplb`; preserve benchmark baseline snapshots and owned-session fencing.

## Review Focus

- Chunking through visual rows, separators and multiple images preserves full-prompt offsets.
- Rotary delta never changes logical KV or sparse causal positions.
- Request reuse, dummy-after-real and padding cannot expose stale state; decode replicas remain live.
- D cached-image skip and uncached image suffix both use native encoder lifecycle.
- Real vision/cache peak and failure cleanup are included before automatic KV sizing.

### Task 1: PCP token-aligned model inputs

**Files:** `vllm_hcu/v1/pcp_manager.py`, `tests/models/hy_v4/test_pcp_multimodal_inputs.py` (new).

**Interfaces:** `get_global_batch(local_batch)`, `get_local_token_mapping(local_batch)`, `gather_global_token_tensor(tensor, local_batch, *, token_dim, out)`.

- [ ] Add CPU tests using real PCP layout for token counts1/15/16/17/31/32/33/63/64/65, mixed requests and decode replicas.
- [ ] Run and confirm missing-interface failures.
- [ ] Implement checked nonaliasing gather with separate dim0/dim1 outputs and padding mask; do not change canonical positions/slot ownership.
- [ ] Verify exact restoration, chunks/cache offsets, zero tails, stale-layout and bad-shape rejection; run related PCP regressions.

### Task 2: Native MM model-state adaptation

**Files:** new `vllm_hcu/models/hy_v4/vl_mrv2.py`; `vl.py`; `vllm_hcu/v1/hcu_model_runner_v2.py`; new `tests/models/hy_v4/test_vl_mrv2.py`.

**Interfaces:** HYV4VL `get_model_state_cls()` selects a DefaultModelState subclass; bind manager after initialization, carry explicit scoped dummy status; use Task1 gather APIs.

- [ ] Write failing tests that exercise actual native global merge with partial images, custom embed masks, multiple images, reorder and prefix-cache skips.
- [ ] Add dummy-before-manager/dummy-after-real and exception/nested-scope tests; canonical position16 must stay slot16 while model RoPE is13 in the rectangular-image fixture.
- [ ] Implement the narrow state adapter, persistent local buffers and native PCP1 delegation without recreating cache/request lifecycle.
- [ ] Run state/PCP/runner tests; preserve native persistent addresses for PCP1 graph inputs and keep P graphs unsupported.

### Task 3: Vision and encoder-cache profiling

**Files:** new `vllm_hcu/v1/hyv4_vl_profile.py`; new `tests/models/hy_v4/test_vl_mrv2_profile.py`; root integrates the context in `hcu_model_runner_v2.py`.

**Interfaces:** image-scoped profiling context around the existing decoder profile path, no-op for other architectures. Reuse native registry/dummy-input/budget utilities.

- [ ] Write failing tests for real encoder invocation, resident-cache accounting, full per-rank peak, no double PCP division, and cleanup after encoder/decoder exceptions.
- [ ] Implement a budget-derived real vision/cache profile, with explicit bounds and checked output cardinality; avoid invented fixed KV caps.
- [ ] Verify no mutation/clearing of live cache entries and no untracked tensors after exit; run native-compatible smoke on hardware before making memory claims.

### Task 4: Narrow route/config/scratch gates

**Files:** `patch/platform/core_fix/patch_vllm_config.py`, `model_executor/layers/fused_moe/deepep_runtime.py`, `model_executor/layers/quantization/slimquant_w4a8.py`; existing config/DeepEP/static-scratch tests.

- [ ] Add red tests for exact VL no-MTP P scope, fixed HT and D LL predicates, and static HT scratch ownership.
- [ ] Preserve negative coverage for other architectures/topologies/connectors, MTP, PP/DP/DCP, graphs, LoRA and offload.
- [ ] Add the exact supported architecture/model_type; leave generic loader/parser and communicator identity guards unchanged.
- [ ] Run relevant positive/negative route/config/static-map regressions.

### Task 5: Integrated reference, calibration and PD acceptance

**Files:** run operators/artifacts under `work/hy4_mm_20261108` and `outputs/HY4_MM_20261108`; runtime documentation after validation.

- [ ] Complete the existing minimal W4A8 image reference; collect clean raw evidence and release its owned sessions.
- [ ] Self-review combined source and run affected CPU tests plus text regressions.
- [ ] Stage an immutable candidate with actual source hashes, create a VL identity R0 JSON and start bounded P/D eager services.
- [ ] Verify images affect outputs, exact placement/rotary contracts, real KV completion/token accounting, mixed/concurrent image requests and safe cleanup.
- [ ] Calibrate fresh image-workload expert counts; build a VL-only R16 map, validate272→256 replica accounting and load samples, print held-out balance.
- [ ] Run original RealWorldQA precision and raw-output comparison before formal throughput; stop/investigate new failures.
- [ ] Run documented cold-KV formal image benchmark, report absolute latency/throughput/cache/limitations, collect commands and logs, release owned GPU processes.
