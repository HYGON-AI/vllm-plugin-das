# vLLM Main 58ad1f3 HCU and Plugin Migration Design

## Objective

Build and validate a reproducible HCU stack against official vLLM commit
`58ad1f3b8973b23943107b51230d594050b42ec3` from 2026-09-07, then publish:

- OpenDAS vLLM branch `main-58ad1f3-hcu` and its DTK wheel.
- vLLM HCU plugin branch `main-58ad1f3` and its matching wheel.
- An upgrade skill that makes the next upstream migration repeatable.

The fixed commit, rather than the moving `main` ref, is the compatibility
contract. A later upstream commit requires a new branch and a new validation
record.

## Inputs

- Official vLLM: `upstream/main@58ad1f3b8973b23943107b51230d594050b42ec3`.
- OpenDAS build references: `origin/vllm-latest-191cecd` and
  `origin/v0.29.0rc4-hcu`.
- Plugin functional baseline:
  `origin/chore/qwen-max-model-len-40960-v0251-clean@9aa55078`.
- Additional proven 0.25.1 work from `origin/v0.25.1`, especially commits
  `021625d`, `353b973`, `cb1fe83`, `85a4ad5`, `8615729`, `d115798`, and
  `325dee8`.
- RC4 worktree as read-only migration evidence. Its uncommitted changes are
  not an implementation base and are not copied wholesale.

## Governing Principles

1. Upstream-first interfaces: use the official implementation when main has
   the required behavior. Keep HCU code only for hardware-specific behavior
   or features absent upstream.
2. Minimal core delta: OpenDAS vLLM contains only DTK/HCU compilation,
   packaging, and unavoidable platform compatibility changes. Model and
   optimization behavior remains in the plugin.
3. Evidence before fallback: a fallback is valid only when logs and tests
   identify the unsupported shape, dtype, topology, or kernel contract.
4. No feature masking: do not make validation pass by disabling prefix cache,
   CUDA Graph, MRV2, MTP, native FP8 KV cache, or the requested backend.
5. No moving artifacts: every wheel records the upstream SHA, OpenDAS SHA,
   plugin SHA, DTK version, torch version, and SHA256 checksum.
6. One compatibility owner: each upstream API is either used directly or
   adapted at one plugin boundary. Parallel monkey patches for the same API
   are removed.
7. Test the delivery form: final model tests use installed wheels in isolated
   directories, never the source tree through an accidental `PYTHONPATH`.

## RC4 Lessons Converted to Requirements

### KV-cache physical layout

RC4 changed attention allocation from the 0.25.1 backend-specific
`[B, 2, N, H, D]` contract to generic packed state. Shape-only splitting
created non-contiguous K/V views, while the self-developed FlashAttention
kernel ignored unsupported strides. The native cache writer itself was
correct, but writer-only and byte-replay tests did not validate the reader.

Main adaptation must trace the current allocation, layer binding, block copy,
connector registration, cache writer, and FlashAttention reader as one data
flow. It must add a real writer-to-reader test for both NHD and HND, including
non-default block stride, storage offset, BF16, and native FP8. A view
reinterpretation is acceptable only when the external kernel demonstrably
reads that physical layout and prefix reuse returns the same output.

### Prefix cache and CUDA Graph

A successful graph capture and correct cache-hit counters do not prove
correctness. Every prefix-cache case must run the identical long prompt at
least three times. Required evidence is a non-zero cache hit on later runs,
byte-identical deterministic output, and sensible text. Eager, PIECEWISE, and
FULL_AND_PIECEWISE are separate cases. Qwen3-8B is the non-hybrid control;
Qwen3.5 is the hybrid attention/GDN control.

### MTP and hybrid metadata

QSA multi-step metadata rebuild and Mamba draft KV-group/prefix-boundary
handling remain required, but they are not assumed to explain every hybrid
failure. First reproduce without speculative decoding, then add MTP3. Draft
acceptance metrics, target output, cache hits, and per-step metadata rebuild
must all be checked.

### AITER MoE

`--moe-backend aiter` means the AITER selector is requested; it does not prove
an AITER expert kernel ran. The self-developed AITER package must load its
tuned CSV and search the exact model shape. A matching config uses AITER. A
missing config falls back to TritonExperts, with an explicit one-time log.
Selection evidence records requested backend, config path, shape key, match
result, and final expert implementation. No dependency is added on upstream
AITER behavior when the container package differs.

### Validation harness reliability

Local HTTP tests must bypass `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY`, or
set `NO_PROXY=127.0.0.1,localhost`; proxy-generated 502 responses are not model
failures. Test prompt generation must remain below `max_model_len`. Service
shutdown waits for API, EngineCore, and workers before releasing devices;
shared-memory `resource_tracker` errors are recorded as shutdown defects, not
silently ignored.

## Architecture

### OpenDAS vLLM layer

Start from the fixed official main SHA. Port the RC4 build contract as a small
reviewable series:

- DTK/DCU CMake detection and compiler definitions.
- HCU-compatible libtorch stable kernel compilation guards.
- DAS local wheel version metadata.
- A build wrapper that creates
  `/usr/lib/x86_64-linux-gnu/librt.so -> librt.so.1` only when required and
  verifies the target before building.
- Build-contract tests and a delta manifest listing every deviation from
  official main.

The RC4 core commit is a reference, not a blind cherry-pick. Each hunk is
checked against main because official CMake and packaging code may already
contain an equivalent fix.

### Plugin layer

Start from the requested clean 0.25.1 branch. Build a compatibility inventory
against main and migrate by owner:

- Platform registration, environment handling, and compatibility gate.
- Attention, FlashAttention/FlashMLA, KV-cache, GDN, and Mamba.
- MRV2 scheduling, CUDA Graph, QSA, and MTP3.
- AITER/Triton MoE routing, DeepEP, and DeepGEMM.
- SlimQuant, channelwise FP8/W8A8, and native FP8 KV cache.
- Hy3, Qwen3.8 Flash Next, Qwen3.5, GLM-5, and related model loaders.
- CI selection, doctor diagnostics, packaging, and lifecycle handling.

For each owner, compare three implementations: official main, proven 0.25.1,
and the RC4 experiment. Adopt official main when semantics match; otherwise
port the smallest HCU implementation and its tests. Obsolete RC4 shims are not
carried forward.

## Execution Gates

### Gate 1: Official baseline and OpenDAS wheel

- Branch HEAD equals the fixed official SHA before HCU commits.
- Core delta manifest contains every changed official file and rationale.
- Build-contract tests pass.
- DTK wheel builds and installs in an isolated target directory.
- Import and platform discovery smoke tests pass before plugin work begins.
- Branch and wheel are pushed to SourceFind with checksums recorded.

### Gate 2: Plugin API compatibility

- Plugin imports against the installed main wheel.
- Compatibility gate, doctor, packaging, lifecycle, and patch ownership tests
  pass.
- Removed/renamed official APIs are handled at explicit adapters, not broad
  exception swallowing.

### Gate 3: Functional domains

- Attention/KV writer-to-reader tests pass for NHD and HND.
- AITER selector tests distinguish config hit from Triton fallback.
- SlimQuant, MoE routing, channelwise quantization, and native FP8 KV cache
  unit and integration tests pass.
- DeepEP and DeepGEMM route selection is observable and validated.
- MRV2, QSA metadata rebuild, Mamba KV-group recognition, and MTP3 tests pass.

### Gate 4: Model validation

Use `--attention-backend FLASH_ATTN`, `--moe-backend aiter`, MRV2, prefix
caching, and CUDA Graph unless a model does not contain the corresponding
component. Reduce environment variables to platform discovery and device
selection; feature behavior is configured by CLI.

| Model | Minimum required coverage |
| --- | --- |
| `/models/Qwen3-8B` | BF16, FLASH_ATTN, prefix reuse, eager and graph control |
| `/models/Qwen3.5-35B-A3B` | AITER lookup/fallback evidence, hybrid prefix reuse, native FP8 KV, MTP3, graph |
| `/models/Qwen3.8-Flash-Next-FP8-Channelwise` | channelwise FP8, FLASH_ATTN/GDN, AITER MoE, MTP3, graph |
| `/models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3` | W8A8/channel FP8, SlimQuant, MoE routing, DeepEP/DeepGEMM where supported |
| `/models/GLM-5-W8A8` | W8A8, FlashMLA, MoE, MTP3, graph, DeepEP/DeepGEMM where supported |

Independent models may run concurrently only on disjoint device sets and
ports. A model passes only when startup, deterministic generation, backend
evidence, prefix metrics, and shutdown all pass. HumanEval is run after the
per-model functional matrix, not as a substitute for it.

## Upgrade Skill Deliverable

Create a Codex skill for subsequent HCU vLLM upgrades. It must include:

- How to freeze and name an upstream baseline.
- How to inventory official API drift before porting patches.
- The official-first and single-owner principles.
- DTK wheel build prerequisites, including the conditional `librt.so` link.
- The self-developed AITER config lookup and Triton fallback contract.
- NHD/HND KV-cache writer-to-reader validation.
- MRV2, QSA, Mamba, MTP3, CUDA Graph, prefix-cache, FP8, DeepEP, and DeepGEMM
  validation gates.
- Local proxy bypass, service lifecycle, log capture, artifact checksums, and
  failure triage.
- RC4 failed approaches and why shape-only KV-cache adaptation is insufficient.

The skill is updated only with behavior demonstrated during this migration.
Unverified assumptions are documented as open risks rather than instructions.

## Deliverables

- SourceFind branch `main-58ad1f3-hcu`.
- Installed and archived OpenDAS vLLM wheel with checksum and build manifest.
- GitHub plugin branch `main-58ad1f3`.
- Installed and archived plugin wheel with checksum and compatibility manifest.
- Unit, integration, model, backend-routing, prefix-cache, graph, MTP3, and
  HumanEval reports.
- Upgrade skill and a concise upgrade retrospective.

## Non-goals

- Do not claim an unreleased version such as `v0.30.0` for official main.
- Do not modify the external self-developed FlashAttention or AITER packages
  unless an isolated package defect is proven and separately approved.
- Do not merge RC4 experimental history into the main branches wholesale.
- Do not declare completion when only startup or graph capture succeeds.
