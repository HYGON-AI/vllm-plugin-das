# Hy4 and Common HCU Runtime Alignment for v0.28.1-dev

## Outcome and frozen inputs

Create one plugin MR against `v0.28.1-dev` that makes the supplied Hy4
Channel-FP8 checkpoint serve on the HCU stack and carries compatible,
still-missing common runtime improvements from `v0.25.1`. Inventory every
source-only commit, but do not equate a unique commit with a missing feature.
Update `/models/upgrading-vllm-hcu` only with conclusions supported by the
final validation evidence.

The initial comparison is frozen at these refs:

- Plugin source: `origin/v0.25.1@6ea7b12f3d77d4564d613aac31c5e6c5e7a8641d`.
- Plugin target: `origin/v0.28.1-dev@93021650a5c768121507d5b8241d0e848756e44e`.
- OpenDAS vLLM target: `v0.28.1-dev@77acaf633d4b1bfdc31390a012cfc470441df88f`.
- Model: `/models/Hy4-preview-Channel-FP8-w8a8-v2`, with `config.json`
  SHA-256 `d4a648cb09bb89f4b8778e60629e43618f1abb581ef3aa38bd67b2b2cd998441`.

Refresh both remote plugin refs immediately before implementation and again
before MR publication. If either ref advances, record the new SHA and recheck
the affected dispositions and validation; do not silently change the base.
Record Python, Torch/HIP, DTK, vLLM/plugin wheel versions, checksums, import
roots, and proprietary provider versions in the run evidence. The installed
OpenDAS vLLM reports the pinned `77acaf6` lineage; its exact delivery artifact
and import root still require verification. Use an isolated wheel installation
when provenance cannot be established. Never install over the global stack.

## Scope decision

The user selected Hy4 plus reusable/common runtime work first, with one
separate MR. Kimi and other independent model-specific work is inventoried for
follow-up rather than combined with Hy4. CI, Docker, and packaging-only commits
are listed for visibility, not copied into this runtime MR. The supplied
checkpoint is Channel-FP8, so its tests cannot prove packed W4A8 checkpoint
arithmetic. Two-node Mooncake and dynamic/record EPLB are also not acceptance
claims for this MR.

The 28 source-only commits fall into these audit lanes:

| Lane | Source commits | Required decision |
| --- | --- | --- |
| Hy4 runtime | `5e68a65`, `7cdd654`, `2b7de5f`, `df99a09`, `3f30e7c` | Adapt final behavior at target interfaces; include only paths with owned tests and hardware evidence. |
| Common runtime candidates | `021625d`, `b701be3`, `4f1f266`, and the generic sparse-indexer portion of `9218102` | Compare target behavior and owner first; port the smallest missing safe behavior or document equivalent/inapplicable with evidence. |
| Target-equivalence candidates | `325dee8`, `8565e54` | Check GDN/FLA and LightOp signatures, guards, and fallbacks; do not duplicate a working target route. |
| Independent model-specific | `1ea04f2`, DeepSeek-specific parts of `9218102`, `e9f4e2d`, `904f4e2` | Inventory for a later model MR unless a narrow generic dependency is demonstrated. |
| CI/build/release | `50c8f55`, `88f8a07`, `39bc3a3`, `15b7cd8`, `1993931`, `11029ca`, `bf8633d`, `8aebc86`, `bc93290`, `2c73085`, `ce2b65f`, `534162d`, `4f1275d`, `6ea7b12` | Record separately; no unrelated workflow or container churn. |

This is a disposition inventory, not a promise to cherry-pick whole commits.
For every candidate, the implementation record must name its functional owner,
target equivalent or gap, exact files selected, tests, and whether a kernel
path became reachable. Generic scheduling from `4f1f266` requires an explicit
same-workload correctness/performance comparison before changing the default;
the target scheduler must not be replaced wholesale. `df99a09` contains
intermediate experiments and a removed regressive mask-TopK route: use its
final state and benchmarks, not every historical step.

## Interface ownership

OpenDAS vLLM already provides the Hy4 config, speculative `hy_v4_mtp`
conversion, reasoning parser, tool parser, and registry entries. Its bundled
Hy4 model implementation explicitly rejects ROCm. The plugin therefore owns
the HCU Hy4 model registration, hardware model layers, sparse-MLA/indexer
adapters, channel-FP8 quantized loading, HCU MoE/attention execution, and any
remaining narrow checkpoint/MTP compatibility behavior. It must reuse the
target core's config, parser, scheduler, KV allocation, and speculative APIs
where their semantics are already correct. Do not register competing parser or
config patches merely because the old plugin did.

Keep each adaptation behind one exact current-owner interface. Check target
signatures and downstream imported aliases before changing callback order or
monkey patches. Missing provider symbols or unsupported shapes take a
documented fallback only when the target implementation preserves semantics;
otherwise fail closed. Reject incompatible ABIs before weight mutation or
cache writes. Preserve native HCU/HIPC KV writing and
backend separation: AITER selects MoE, not sparse attention or KV writes.
Do not force CUDA Graph eager/PIECEWISE as a compatibility shortcut.

Target-only TP8 is the first operational topology. Then add native MTP3 and
FP8 E4M3 KV under the same model and default Graph policy. PCP/EP, DCP, static
EPLB, W4A8, or an optional LightOp path may enter this MR only with a named
contract test and relevant device gate; otherwise exclude the path and report
the gap. In particular, the existing v0.28.1 sparse-MLA PCP Graph rejection
remains authoritative; an eager PCP result cannot be described as Graph
support. Any required OpenDAS core change is a separate scope decision with
its own paired wheel and compatibility review, not an implicit plugin patch.

## Validation and acceptance

1. Establish import provenance and a clean baseline before runtime changes.
   The target branch's `tests/patch/test_plugin_lifecycle.py` currently passes
   `34` tests. `tests/patch/test_platform_dispatcher.py` currently fails at
   collection because it defaults to the absent legacy
   `VLLM_V0251_SOURCE_ROOT`; migrate that test to an installed-target-root
   contract as part of the planned test work, not as proof of a runtime defect.
2. Start from failing target-interface tests for each selected behavior.
   Cover model/config import, registration, loader extents and scales,
   quantized arithmetic, sparse-indexer and cache layout, MoE routing,
   MTP/PCP metadata where applicable, default/fallback operator selection,
   and callback ownership. Run every test file changed by the full MR diff,
   the relevant runtime-patch suite, and installed-artifact bootstrap checks.
3. Run a fresh foreground TP8 server on the supplied checkpoint with AITER
   MoE, prefix caching, and the target default Graph policy. Confirm actual
   Channel-FP8 kernel and AITER config selection or explicit per-shape Triton
   fallback, target PIECEWISE/FULL capture, sparse-attention route, healthy
   HTTP responses, and coherent repeated long-prefix output with nonzero
   later cache hits. Record the exact command and logs.
4. Repeat with native MTP3, then MTP3 plus `--kv-cache-dtype fp8_e4m3`.
   Confirm target/draft capture, nonzero drafted and accepted tokens by
   position, correct output, and no silent FP8 cache fallback. Do not claim
   precision from startup, graph capture, or acceptance counters alone.
5. Evaluate exactly HumanEval/0-7 with the same checkpoint, deterministic
   API sampling, `reasoning_effort=no_think`, the same generation limit,
   prompt/template, evaluator, and fresh output directories for target-only
   and MTP3. Record eight unique predictions and reviews, pass@1, per-item
   flips, finish reasons, errors, and generation lengths. Investigate
   `max_tokens` truncation separately from kernel accuracy. If an optional
   topology or newly defaulted kernel is included, run its own comparable
   accuracy gate instead of borrowing TP8 evidence.
6. Use proxy-free localhost clients and owned foreground PTY service sessions.
   After each run, verify API/EngineCore/worker teardown, listener removal,
   and selected-device memory release. Retain failures and retry evidence.

The historical v0.25.1 TP8 result (target, MTP3, and FP8 E4M3 KV) is a
comparison protocol, not proof for v0.28.1. No MR push or success claim occurs
until the complete intended diff is reviewed, blocking findings are fixed,
relevant gates rerun, and the committed diff is reviewed again. If a required
hardware gate fails, report the MR as unverified instead of weakening its
topology, graph mode, or backend to obtain a green result.

## Deliverables

- One clean plugin branch/MR against the refreshed `v0.28.1-dev` base, with
  the commit-disposition matrix, review notes, exact server commands, and
  validation artifacts in its description or comments.
- A concise, evidence-backed update to `/models/upgrading-vllm-hcu/SKILL.md`
  and its relevant reference file, preserving failure states and the final
  skill path in the handoff.
- A list of excluded model-specific or unvalidated optional paths so a later
  MR does not mistake omission for target equivalence.
