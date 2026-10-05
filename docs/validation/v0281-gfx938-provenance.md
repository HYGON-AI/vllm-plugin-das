# vLLM 0.28.1 gfx938 validation provenance

Recorded at `2026-10-03T18:29:55+00:00`. This record freezes the wheel pair
used to start the gfx938 model-validation campaign. A final plugin wheel is
rebuilt after any runtime fixes; this pre-validation wheel remains immutable
evidence for the initial harness and compatibility gates.

## Source topology

- Plugin repository branch: `codex/validate-v0281-gfx938-models`
- Plugin wheel source commit:
  `f189798e80b34a94b1577de19a70fab549236661`
- Rebased PR 181 stack head:
  `592b38cf9ce9f6e36ab1c846ecd2483f03e20fd9`
- PR 180 base head:
  `f24e99be620c3a0d3611a7d96079bb7be7e63de8`
- `origin/v0.28.1-dev` baseline:
  `9788dc633fb3d0a0a0a124b5886f62c07b3de765`

The plugin wheel was built from a clean tracked worktree with:

```bash
ROCM_PATH=/opt/dtk MAX_JOBS=16 python setup.py bdist_wheel
```

## Frozen wheels

| Package | Filename | SHA256 |
| --- | --- | --- |
| vLLM | `vllm-0.28.1+dtk2604.torch2110.2609171627.g77acaf-cp310-cp310-linux_x86_64.whl` | `2d6b392dcff0d5064c754e838d9ac5ea3ffd5d6c7cac86ed2b5b1b2811cb9b3b` |
| vLLM HCU plugin | `vllm_hcu-0.28.1rc1.dev491+das.f189798.dtk26041-cp310-cp310-linux_x86_64.whl` | `407b68d3a37a1148c835e3b12a10c7c59cd5d184c2f95ceb0a0484a3a67c8fd5` |

The vLLM wheel is stored at `/models/artifacts/v0281-gfx938/`. The plugin
wheel is the `dist/` artifact from the source commit above. The `dtk26041`
local-version component is the plugin build system's encoding of the installed
DTK `26.04.1` release.

The wheels were installed without dependencies into these immutable targets:

- vLLM: `/models/.installs/vllm-v0281-gfx938-g77acaf`
- plugin: `/models/.installs/vllm-plugin-v0281-gfx938-f189798`

`importlib.metadata` reported the following distributions from those exact
roots:

- `vllm==0.28.1+dtk2604.torch2110.2609171627.g77acaf`
- `vllm-hcu==0.28.1rc1.dev491+das.f189798.dtk26041`

## Parent and child import roots

The import probe ran with `PYTHONNOUSERSITE=1`, the vLLM target and plugin
target at the front of `PYTHONPATH`, and path assertions in both the parent and
a fresh child interpreter. Both processes resolved the same files:

```text
vllm      /models/.installs/vllm-v0281-gfx938-g77acaf/vllm/__init__.py
vllm_hcu  /models/.installs/vllm-plugin-v0281-gfx938-f189798/vllm_hcu/__init__.py
hcu_ops   /models/.installs/vllm-plugin-v0281-gfx938-f189798/vllm_hcu/hcu_ops.cpython-310-x86_64-linux-gnu.so
```

The following vLLM native modules were also resolved below the isolated vLLM
root:

```text
vllm._C                         vllm/_C.abi3.so
vllm._rocm_C                    vllm/_rocm_C.abi3.so
vllm._C_stable_libtorch         vllm/_C_stable_libtorch.abi3.so
vllm._moe_C_stable_libtorch     vllm/_moe_C_stable_libtorch.abi3.so
```

## Runtime environment

- Host kernel: `Linux 6.6.92-34.1.tl4.x86_64`
- Python: `3.10.12` (`/usr/bin/python`)
- DTK release files: `26.04.1`
- HIP compiler version: `6.3.26255`
- Torch: `2.11.0+dtk2604.2609291046.ge12687`
- Hardware: 8 x `BW1100`, `gfx938:sramecc+:xnack-`, 147440 MiB VRAM per
  device
- `/usr/lib/x86_64-linux-gnu/librt.so`: absent
- `/usr/lib/x86_64-linux-gnu/librt.so.1`: present

Installed operator/provider distributions:

| Distribution | Version |
| --- | --- |
| `aiter` | `0.1.6+dtk2604.torch2110.2609300837.gf9b9c8` |
| `boltops` | `0.1.0+dtk2604.torch2110.2609301634.g137f5d` |
| `flash-attn` | `2.8.4+dtk2604.torch2110.2609241509.g624d7b` |
| `flash-mla` | `1.2.0+dtk2604.torch2110.2609291807.gc24e3f` |
| `lightop` | `0.6.0+dtk2604.torch2110.2609301833.g9d6ed8` |
| `deep-ep` | `1.1.0+dtk2604.torch2110.2609231914.g49988b` |
| `deepgemm` | `2.1.0+dtk2604.torch2110.2609301620.g9dc3b6` |

## Artifact verification

The installed-root compatibility, packaging, and lifecycle gates were run from
outside the checkout with pytest importlib mode so the source tree could not
shadow either wheel:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/models/.installs/vllm-v0281-gfx938-g77acaf:/models/.installs/vllm-plugin-v0281-gfx938-f189798 \
VLLM_TARGET_ROOT=/models/.installs/vllm-v0281-gfx938-g77acaf \
VLLM_HCU_TARGET_ROOT=/models/.installs/vllm-plugin-v0281-gfx938-f189798 \
VLLM_PLUGINS=__disabled__ \
python -m pytest --import-mode=importlib -q \
  /models/.worktrees/vllm-plugin-das-v0281-gfx938-validation/tests/patch/test_compatibility_gate.py \
  /models/.worktrees/vllm-plugin-das-v0281-gfx938-validation/tests/patch/test_setup_packaging.py \
  /models/.worktrees/vllm-plugin-das-v0281-gfx938-validation/tests/patch/test_plugin_lifecycle.py
```

Result: `58 passed, 14 warnings in 107.64s`.

## Source-tree model matrix

The following results were produced from the validation worktree with the
pinned vLLM runtime above. All successful service launches used
HcuGPUModelRunnerV2 and the resolved default Graph policy; no accuracy result
was rescued with `--enforce-eager`. Exact arguments, ports, request protocol,
and owned artifact directories are defined in
`tests/models/v0281_gfx938_humaneval16.yaml`.

| Profile | Topology and feature route | HumanEval16 | Disposition |
| --- | --- | ---: | --- |
| `deepseek_v32_channel_fp8_tp8` | TP8, sparse MLA, channel FP8 W8A8, AITER, prefix, default Graph | 16/16 | Pass; native LBNHC sparse-MLA KV layout retained |
| `deepseek_v32_channel_fp8_mtp3_kvfp8_tp8` | TP8, sparse MLA, channel FP8 W8A8, AITER, MTP3, E4M3 sparse KV, prefix, default target/speculator Graphs | 16/16 | Pass; public E4M3 mapped to `fp8_ds_mla`, MTP draft acceptance observed |
| `deepseek_r1_channel_fp8_tp8` | TP8, regular MLA, channel FP8 W8A8, AITER, prefix, default Graph | 16/16 | Pass; current OpenAI response `reasoning` field accepted by the prefix probe |
| `deepseek_r1_channel_fp8_mtp3_tp8` | TP8, regular MLA, channel FP8 W8A8, AITER, MTP3, prefix, default target/speculator Graphs | 16/16 | Pass; MTP draft acceptance observed throughout the concurrent run |
| `deepseek_v4_flash_tp8` | TP8, sparse MLA, AITER, DSpark7, prefix, default Graph | 2/16 concurrent; 1/3 serial diagnostic | Local checkpoint is the previously documented incomplete asset; DSpark smoke passed, but no full accuracy claim |
| `glm5_w8a8_tp8` | TP8, sparse MLA, AITER INT8 MoE, MTP3, prefix | 16/16 | Pass; repeated on the final shared-expert code with target/draft FULL plus PIECEWISE Graphs |
| `glm52_channel_int8_tp8` | TP8, sparse MLA, Channel INT8, AITER INT8 MoE, E4M3 sparse KV, MTP3, prefix, default target/speculator Graphs | 16/16 | Pass; public E4M3 mapped to `fp8_ds_mla`, final-window MTP draft acceptance 97.2% |
| `glm53_channel_fp8_tp8` | TP8, sparse MLA, AITER FP8 MoE, E4M3 public KV mapped to `fp8_ds_mla`, MTP3 | 16/16 | Pass; repeated on the final shared-expert code and extended to HumanEval 32/32 |
| `glm51_channel_fp8_tp8` | TP8, sparse MLA, AITER FP8 MoE, MTP3, prefix | 16/16 | Pass; EvalScope artifact ID collapses repeated underscores |
| `hy3_channel_fp8_mtp2_kvfp8_tp8` | TP8, FLASH_ATTN, channel FP8 W8A8, AITER, MTP2, E4M3 KV, prefix | 16/16 | Pass; HND selection resolved to the physical LBHNC cache and default target/speculator Graphs |
| `hy3_channel_fp8_dp8_ep8_mtp2_kvfp8` | DP8/TP1/EP8, FLASH_ATTN, channel FP8 W8A8, DeepEP low-latency/DeepGEMM, MTP2, E4M3 KV, prefix | 16/16 | Pass; nine-request probe demonstrated rank-local prefix reuse |
| `hy4_preview_channel_fp8_tp8` | TP8, sparse MLA, AITER FP8 MoE, MTP3, prefix | 16/16 | Pass after `indexed_attention` and `reasoning_effort=no_think` fixes |
| `minimax_m25_int8_tp4` | TP4, FLASH_ATTN, HND/BHSD kernel view, AITER, prefix | 16/16 | Pass |
| `qwen2_57b_tp2` | TP2, FLASH_ATTN, HND/BHSD kernel view, prefix | 15/16 | Deterministic HumanEval/10 miss under both AITER and Triton MoE; retained as checkpoint/model outcome |
| `qwen3_30b_int8_tp2` | TP2, FLASH_ATTN, HND/BHSD kernel view | 5/16 | AITER and Triton MoE both 5/16; dense route 6/16; direct INT8 kernel probes pass, so no backend-specific fix was justified |
| `qwen3_8b_tp2` | TP2, FLASH_ATTN, native FP8 E4M3 KV, prefix | 16/16 | Pass; BF16 control also 16/16 |
| `qwen35_35b_tp2` | TP2, FLASH_ATTN, AITER BF16 MoE, MTP3, prefix | 16/16 | Pass |
| `qwen35_35b_w8a8_tp2` | TP2, FLASH_ATTN, AITER INT8 MoE, E4M3 KV, MTP3 | 16/16 | Pass |
| `qwen36_27b_w8a8_tp2` | TP2, FLASH_ATTN, W8A8, prefix | 15/16 | Deterministic HumanEval/8 miss retained |
| `qwen38_27b_int8_tp2` | TP2, FLASH_ATTN, INT8, prefix | 16/16 | Pass |
| `qwen38_flash_next_fp8_tp4` | TP4, hybrid BLNHC layout, AITER FP8 MoE, E4M3 KV, MTP3 | 16/16 | Pass; prefix hits observed |
| `qwen38_flash_next_w4a8_tp4` | TP4, hybrid BLNHC layout, SlimQuant W4A8, AITER, MTP3 | 16/16 | Pass |

The final shared-expert stream-safety change was followed by fresh live
regression runs for both large GLM routes. `/models/GLM-5-W8A8` used TP8,
regular BF16/auto sparse MLA KV, AITER INT8 MoE, MTP3, prefix caching, and the
default FULL_AND_PIECEWISE target/draft Graph policy. It passed raw and
normalized HumanEval 16/16; all 18 API requests, including the two prefix
probes, returned HTTP 200 with no ERROR or Traceback. The final metric window
reported 35.2% prefix hits and 93.0% draft-token acceptance. The report
observed 27.41 output tokens/s, 4.782 s mean latency, 591.7 ms TTFT, and
32.1 ms TPOT. The route allocated 400,512 KV tokens; evidence remains under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-glm5-w8a8-tp8`.

`/models/GLM-5.3-Channel-FP8-w8a8` used TP8, `FLASHMLA_SPARSE`, AITER FP8
MoE, MTP3, public E4M3 KV resolved to `fp8_ds_mla`, prefix caching, and the
default FULL_AND_PIECEWISE target/draft Graph policy. A fresh HumanEval16 gate
passed raw and normalized 16/16. The extended accuracy gate then passed raw
and normalized HumanEval 32/32 with 34/34 HTTP 200 responses, no ERROR or
Traceback, 737,920 KV tokens, and a final report of 25.6 output tokens/s,
3.082 s mean latency, 594.3 ms TTFT, and 32.2 ms TPOT. Evidence is under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-glm53-channel-fp8-tp8-humaneval32-final`.

DeepSeek V4.1 was explicitly excluded at the requester's direction. The
optional current-branch HY4 DP8/TP1/EP8 rerun subsequently passed 16/16 with
MTP3, E4M3 sparse KV, DeepEP low-latency/DeepGEMM, MRV2 on all eight ranks,
and default FULL_AND_PIECEWISE target/speculator Graph capture. Its artifacts
are under `/tmp/vllm-hcu-evalscope/v0281-gfx938-hy4-dp8-ep8-mtp3`. The
generic two-request prefix probe was not accepted as DP8 evidence because its
requests landed on different ranks; no current-branch DP8 prefix-hit claim is
made.

`/models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3` was then validated on
both the required TP8 route and the optional DP8/TP1/EP8 low-latency route. The
checkpoint is regular GQA rather than MLA, so both profiles used
`FLASH_ATTN`, Model Runner V2, public E4M3 KV, MTP2, prefix caching,
`reasoning_effort=no_think`, and the default FULL_AND_PIECEWISE target and
speculator Graph policy. `VLLM_KV_CACHE_LAYOUT=HND` selected the HND/BHSD-facing
route and resolved to the runtime's physical LBHNC cache layout.

The TP8/AITER run passed raw and independently normalized HumanEval 16/16,
with 18/18 HTTP 200 responses and no ERROR or Traceback. It allocated
4,860,672 KV tokens, reported a final 35.1% prefix-hit rate and 90.0% draft
acceptance, and observed 31.23 output tokens/s, 3.024 s mean latency, 906.9 ms
TTFT, and 21.91 ms TPOT. Evidence is under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-hy3-channel-fp8-mtp2-kvfp8-tp8`.

The DP8/TP1/EP8 profile selected DeepEP low-latency and DeepGEMM, started one
Model Runner V2 engine per device, captured FULL and PIECEWISE target/draft
Graphs, and passed raw and normalized HumanEval 16/16. Its successful request
window contained 25/25 HTTP 200 responses and no runtime ERROR or Traceback;
owned SIGTERM cleanup subsequently emitted one expected cancellation-side
`EngineDeadError`. A two-request probe initially returned zero because the
requests landed on different DP ranks. The harness now supports a configurable
prefix-probe request count, and this profile sends DP-size-plus-one (nine)
identical requests; the repeated rank recorded 2,752 hit tokens. The report
observed 32.73 output tokens/s, 2.368 s mean latency, 273.3 ms TTFT, and
27.55 ms TPOT. Evidence is under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-hy3-channel-fp8-dp8-ep8-mtp2-kvfp8`.

The subsequently added `/models/DeepSeek-R1-Channel-FP8-w8a8` checkpoint was
validated twice on the same eight-card host. Both the plain TP8 route and the
TP8+MTP3 route passed raw and independently normalized HumanEval16 at 16/16,
with all 16 generated programs executed successfully. The plain run recorded
10.33 output tokens/s and 95.18 ms mean TPOT. The MTP3 run recorded 24.30
output tokens/s, 41.42 ms mean TPOT, and live draft acceptance rates of about
52-65%; the two runs emitted different token counts, so this is observational
rather than a controlled performance benchmark. The MTP3 server resolved
`DeepSeekMTPModel`, compiled a separate `eagle_head`, warmed the three-token
rejection sampler, captured default FULL_AND_PIECEWISE graphs, allocated
635,200 KV tokens with LBNHC layout, and loaded AITER channel-shuffle stage1
and stage2 kernels on all eight ranks. Artifacts are under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-deepseek-r1-channel-fp8-{tp8,mtp3-tp8}`.

The first plain TP8 attempt exposed a harness compatibility issue rather than
a model failure: vLLM 0.28.1 returned reasoning-only probe output in the
current `message.reasoning` field while the probe recognized only `content`
and the deprecated `reasoning_content` alias. The probe now checks those three
fields in current-protocol order, with regression coverage. After that narrow
fix, the unchanged service route passed the prefix probe and HumanEval16.

The later `/models/DeepSeek-V3.2-channel-fp8` checkpoint was validated through
both a plain TP8 profile and a TP8+MTP3+E4M3-KV profile. Both passed raw and
independently normalized HumanEval16 at 16/16, with all generated programs
executed successfully. The plain run observed 9.31 output tokens/s and 104.13
ms mean TPOT. The combined feature run observed 22.32 output tokens/s, 38.46
ms mean TPOT, mean MTP acceptance length 3.17, and 72.5% draft-token
acceptance in its final metric window. Generation lengths differed, so the
throughput numbers are route evidence rather than a controlled benchmark.

Both routes used HcuGPUModelRunnerV2, `FLASHMLA_SPARSE`, channel-wise FP8
dense linear, tuned AITER channel-shuffle MoE, prefix caching, native LBNHC KV
layout, and the default FULL_AND_PIECEWISE Graph policy. The combined profile
mapped public `fp8_e4m3` to sparse-MLA `fp8_ds_mla`, allocated 981,376 KV
tokens, compiled the `eagle_head`, warmed the three-token rejection sampler,
and captured both target and speculator PIECEWISE/FULL graphs. Its checkpoint
keeps `model_type=deepseek_v3`, so vLLM reports `DeepSeekMTPModel`; the plugin's
shared HCU MTP implementation identifies V3.2 sparse attention from
`index_topk`. Evidence directories are
`/tmp/vllm-hcu-evalscope/v0281-gfx938-deepseek-v32-channel-fp8-tp8` and
`/tmp/vllm-hcu-evalscope/v0281-gfx938-deepseek-v32-channel-fp8-mtp3-kvfp8-tp8`.

The first plain attempt used a 4,096-token server context together with an
EvalScope `max_tokens=4096` output budget, leaving no room for input tokens and
producing HTTP 400 before inference. The accepted profiles use
`--max-model-len 8192` with the same 4,096-token output budget; this was a
request-contract correction, not a model or kernel fix.

The subsequently added `/models/GLM-5.2-Channel-INT8-w8a8` checkpoint is
704.33 GiB across 282 shards and was validated at TP8. Its combined acceptance
profile used HcuGPUModelRunnerV2, `FLASHMLA_SPARSE`, `--reasoning-parser
glm45`, AITER INT8 MoE, MTP3, public E4M3 KV mapped to `fp8_ds_mla`, prefix
caching, native LBNHC layout, and default FULL_AND_PIECEWISE target/speculator
Graphs. Raw EvalScope and the independent normalizer both passed HumanEval16
at 16/16, with all generated programs executed successfully.

The run selected the official `TritonInt8ScaledMMLinearKernel` for dense
compressed-tensors W8A8 and AITER for MoE; this is an expected separation of
dense and MoE providers, not a fallback of the requested MoE route. All ranks
loaded the gfx938 `E=256,N=256` ordinary and bottom-layer AITER configs plus
the channel-shuffle tuned table. The final metric window reported mean MTP
acceptance length 3.92 and 97.2% draft-token acceptance. The report observed
29.19 output tokens/s, 4.581 s mean latency, 595.19 ms mean TTFT, and 29.81 ms
mean TPOT. The feature route allocated 744,896 KV tokens; evidence is under
`/tmp/vllm-hcu-evalscope/v0281-gfx938-glm52-channel-int8-tp8`.

The later `/models/Kimi-K2.6` checkpoint was validated as a language-only TP8
route with HcuGPUModelRunnerV2, regular FLASHMLA, Triton WNA16 MoE, prefix
caching, and BF16/auto LBNHC KV. The checkpoint has no MTP layers. Two runtime
ownership hypotheses were investigated, but only one reproduced the fault:
the HCU fused-MoE replacement omitted upstream ROCm's guard against
auxiliary-stream shared-expert overlap when routed/shared inputs alias.
WNA16's unquantized routed-input path now serializes shared experts. Restoring
BF16 FlashMLA's advertised full-graph capability passed the high-risk batch
and HumanEval 64/64, disproving the scheduler-metadata hypothesis. The final
command needs no debug environment switch and retains default
FULL_AND_PIECEWISE Graph capture and prefix reuse.

The deterministic Instant-mode run uses temperature zero, `thinking=false`,
batch eight, and a 1,024-token output cap. It produced 64 predictions and
reviews and passed independently normalized HumanEval 64/64. EvalScope's raw
checker undercounted complete fenced or horizontally indented modules, so the
accepted gate uses the repository's syntax-aware normalizer inside the
explicit isolated execution boundary. Earlier 51/64, 52/64, 54/64, and 55/64
controls remain diagnostic history from before the stream-race fix. Full
evidence and exact commands are in
`docs/validation/kimi-k26-gfx938-humaneval64.md`.

## Source closure after runtime fixes

- Full `tests/runtime_patch`: `1679 passed, 14 warnings` after the final Kimi
  force-control fix and review updates.
- Full `tests/models/hy_v4`: `222 passed, 14 warnings`.
- HY4 live TP8 gate: `16/16`, 16 predictions and reviews, final observed
  prefix hit rate about 34.3%, final-window MTP acceptance about 91.3%.
- Hy3 Channel-FP8 live gates: TP8/AITER and DP8/TP1/EP8
  DeepEP-low-latency/DeepGEMM profiles each passed raw and normalized
  HumanEval `16/16` with MTP2, E4M3 KV, prefix caching, and default target and
  draft FULL plus PIECEWISE Graphs. The DP profile used nine identical prefix
  requests so at least one request returned to the same rank.
- DeepSeek-V3.2 live TP8 gates: plain and MTP3+E4M3-KV profiles each `16/16`,
  with 16 predictions, reviews, and successful code executions per profile.
- GLM-5.2 Channel-INT8 live TP8 gate: MTP3+E4M3-KV profile `16/16`, with
  16 predictions, reviews, and successful code executions.
- GLM-5 W8A8 post-shared-expert live TP8 gate: MTP3 profile `16/16`, with
  target/draft FULL plus PIECEWISE Graphs, 18/18 HTTP 200 responses including
  prefix probes, and no ERROR or Traceback.
- GLM-5.3 Channel-FP8 post-shared-expert live TP8 gates: MTP3+E4M3-KV passed
  `16/16`, then the extended accuracy gate passed `32/32`; the final service
  recorded 34/34 HTTP 200 responses and no ERROR or Traceback.
- Kimi-K2.6 live TP8 gate: deterministic Instant mode produced 64
  predictions/reviews and normalized HumanEval `64/64`; service, prefix,
  FULL plus PIECEWISE Graphs, FLASHMLA/LBNHC, and Triton WNA16 routes passed
  without a shared-expert stream environment override. The same service first
  passed the eight high-risk prompts, for 72/72 HTTP 200 responses total and
  no ERROR or Traceback.
- Kimi-K2.6 explicit E4M3-KV smoke: the TP8 route resolved
  `kv_cache_dtype=fp8_e4m3`, captured FULL plus PIECEWISE Graphs, retained
  LBNHC, allocated 1,718,272 KV tokens, and completed the eight historical
  high-risk concurrent prompts with HTTP 200 and normal stops.
- Kimi-K2.6 benchmark-style Thinking diagnostic: 64 predictions/reviews,
  normalized HumanEval `55/64`, with four 16K reasoning-only truncations. This
  predates the shared-expert stream-race fix and is retained only as history.
- Kimi changed-file focused suite includes explicit BF16/FP8 FlashMLA Graph
  capability, shared-expert stream-safety, and force-control regression
  coverage. The final deterministic profile enforces normalized score `1.0`;
  the final focused re-review passed `40` tests with one deselection and both
  static command contracts passed (`2 passed, 1 deselected`).
- Current profile/report focused suite: `69 passed, 1 skipped`.
- Skill validation: `Skill is valid!`; repository/skill PAT-pattern scan:
  clean; `git diff --check`: clean.

HumanEval executes model-generated code. The harness now requires an explicit
EvalScope sandbox or the operator assertion
`VLLM_HCU_HUMANEVAL_ISOLATED=1`; merely detecting a container is not treated
as a security boundary. Sandbox results are not re-executed by the local
normalizer. Diagnostic profiles normally check only artifact counts; an
explicit `record_normalized_score` option may re-run HumanEval checks only at
an already isolated host boundary, recording but not enforcing the score.
The evaluator receives an isolated HOME and an explicit
credential denylist. EvalScope API keys are passed through a protected
environment to a Python launcher and do not appear in OS argv or persisted
command logs.
Inherited outbound proxies are preserved for dataset access while loopback is
always added to `NO_PROXY`; the vLLM server ignores inherited proxies but
honors an explicitly configured server proxy.

## Final reviewed plugin wheel

The reviewed runtime/test change is commit
`cdea830d36ba79a8dc2410253d871954498ab72c`. The clean tracked worktree at
that commit produced:

| Package | Filename | SHA256 |
| --- | --- | --- |
| vLLM HCU plugin | `vllm_hcu-0.28.1rc1.dev491+das.cdea830.dtk26041-cp310-cp310-linux_x86_64.whl` | `73898c9400f4c0476498573a243b7c9feb7494719cbe4aca1a492f26c85f37be` |

It was installed with `--no-deps` under
`/models/.installs/vllm-plugin-v0281-gfx938-cdea830`. With
`PYTHONNOUSERSITE=1` and the pinned vLLM root first in `PYTHONPATH`, both the
parent and a fresh child interpreter resolved:

```text
vllm      /models/.installs/vllm-v0281-gfx938-g77acaf/vllm/__init__.py
vllm_hcu  /models/.installs/vllm-plugin-v0281-gfx938-cdea830/vllm_hcu/__init__.py
hcu_ops   /models/.installs/vllm-plugin-v0281-gfx938-cdea830/vllm_hcu/hcu_ops.cpython-310-x86_64-linux-gnu.so
```

The installed-root compatibility, packaging, and lifecycle gate completed
with `62 passed, 14 warnings in 126.35s`. The exact committed runtime patch
suite completed with `1669 passed, 14 warnings in 426.33s`; the final complete
changed-file suite completed with `581 passed, 3 skipped, 14 warnings in
151.81s`.
