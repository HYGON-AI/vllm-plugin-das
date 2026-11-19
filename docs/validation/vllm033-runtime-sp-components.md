# Runtime SP communication and official attention components

This candidate targets Plugin v0.33.0 at `6fa728d2b913d8e4e46e5f5638c90a48405c3b3f`.
Core owns model algorithms, attention, Linear allocation/loading/GEMM, MoE
factory/backend/manager policy and teardown. Adapter owns platform identity,
the communicator selector and process-local lifecycle. Plugin adapts optional
communication admission and attention token/head layouts through public seams.
No Core, native kernel or FlashAttention library change is included.

## Construction and ownership

The communication extension is explicit. Discovery imports no framework and
the existing custom-SP sidecar remains the policy input. EP admission reaches
the unchanged official constructor before manager construction. Constructor
arguments, including stateless rank metadata and `tcp_store_group`, are
preserved. Equal-split transport requires the borrowed TP device group.

Attention installs concrete QKV/Row public OOT subclasses, with foreign-owner
preflight and serialized publication through Adapter's `ProcessLocalLifecycle`.
Partial publication is sticky; subsequent installation/health checks refuse.
Scoped construction adapts only attention QKV/output prefixes. Ordinary
construction retains official TP. Active instances reject PID/group/geometry
drift before their projection.

Replicated weights are selected before allocation with `disable_tp=True`.
Official split/fused weight and scale loaders, post-load processing and
`update_param_tp_status` remain responsible for loading/refitting. QKV sends
each token shard's correct KV head to every destination, including contiguous
GQA replicas: TP4/KV2 maps ranks0/1 to KV0 and ranks2/3 to KV1. The legacy
modulo-selection/all-gather counterfactual mismatches KV2 on all four ranks;
the candidate matches the independent reference for KV1 and KV2.

There is no automatic attention extension/model registration or service
enablement. Full model entry/exit and early MoE construction policy must be
composed first.

## Recorded validation

Installed attention runtime: `89fb8ce07282404ef486fea69cf363180ca95766`.
Plugin wheel SHA256:
`95d0583d25aad030f950c57f63d93c50db7d4d306e81f76ff94d1c2f142c3e19`.
Core source: `c440c089fe9621c5b03e1618cf86a13e95de8bf1`.
Adapter candidate: `004cce0a83055b2f67acdc1060258773c3275787`.
The latter exposes `HcuPlatform.register_device_communicator_selector` and
`owns_device_communicator_selector`; this is an unpublished dependency, not a
claim about released Adapter main. SHAs identify evidence, not runtime locks.

| Scope | Result and limit |
| --- | --- |
| Attention source CPU | 64 passed; real Core constructors, parameters, split/fused loaders and reloads. GEMM/transport doubles isolate layouts. |
| Independent installed attention CPU | 64 passed, zero failures/errors/skips; only test source-path injection removed; production import resolves inside the private prefix. |
| Current package/CI contracts | Source/package8, independently installed communication36, clean CPU CI98 and skill48 pass. The native prefix and then an unsanitized new venv produced metadata fixture failures; clearing inherited Python search-path variables in the clean venv passes the unchanged98 tests. |
| FP8 CPU allocation/loading | Real Fp8Config/Fp8LinearMethod weight/scale loading with a kernel-initialization double. This does not qualify FP8 execution. |
| Communication CPU | Original source36/installed36, plus Adapter host9, passed in their recorded r186 combination. |
| Native communication | Original r186 TP2 gfx938 baseline reproduces missing EP manager; candidate uses official AgRs. FP32/BF16 independent zero-tolerance eager and changed-input Graph checks pass on both ranks. |
| Native official attention | TP4 gfx938, GLM4/HY3 × BF16/FP16 × KV2/KV8: 8 cases per rank, 32 rank/case combinations. Actual official attention modules, RMS/RoPE and HCU KV writer/vendor reader execute; no attention double. |
| Numerical/Graph | Independent CPU FP32 unsharded reference with dtype roundings; original atol=rtol=0.016. Changed inputs and positions at epochs1/3, stable addresses; maximum absolute error0.00390625. |
| Provenance/cleanup | Source/wheel/installed Python equality and18 complete owner RECORDs checked before/after. Native attention returns0, no controller signal, tracker0, no owned process remains. |

The attention probe uses synthetic 8-token attention, hidden128, Qheads8,
head_dim64, explicit FLASH_ATTN, auto BF16/FP16 KV and LBHNC block64. The32
rank/case combinations are not32 model runs or HumanEval32.

Original failures are retained in the delivery evidence archive
`runtime-sp-attention-layout-r187`: initial hidden-device FP8 fixture failures,
an incomplete legacy counterfactual probe, the first native capture timeout,
and the stale package entry-point expectation. The package test now retains
the complete exact table and includes the new communication entry point.

The first native arm warmed only the default stream, then captured on a fresh
stream. hipBLASLt reported capture errors and the480s controller sent SIGTERM.
Its immediate descendant snapshot was not clean; a later PID/starttick audit
confirmed all owned processes exited naturally, with no supplemental signals.
The second arm additionally warmed the exact capture stream and passed. The
attention math, precision, backend and numerical assertions are unchanged;
between the arms the runtime also gained the separately tested publication
lifecycle guard. This is evidence for the corrected probe, not a causal proof
about the vendor's internal failure mechanism. FlashAttention was not rebuilt.

## Reproduce the bounded native probes

Use an isolated installed environment with the compatible Core/vendor stack,
the paired Adapter selector API and this Plugin. Reserve two GPUs for
`tools/probe_runtime_sp_communication.py` or four for
`tools/probe_runtime_sp_attention.py`; set the device masks for that reservation.
Create an empty output directory. With `HCU_ADAPTER_HOST=vllm`,
`VLLM_PLUGINS=hcu_adapter` and
`HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=runtime-sp-communication,hcu-flash-attention`,
run:

```sh
python -P -B tools/probe_runtime_sp_attention.py /path/to/empty-output candidate
```

For the communication probe, require `runtime-sp-communication` and pass
`baseline` or `candidate` as the second argument. These scripts execute the
recorded probe bodies; use an external bounded controller with exact owned
process tracking and inspect every rank JSON and terminal process result.
They do not manage another client's devices or qualify a service lifecycle.

## Remaining migration scope

Keep Draft. Full model embedding/residual/PP entry/exit, dense/shared experts,
early `FusedMoEFactory(is_sequence_parallel=True)` policy, quantized native
execution, target/draft batching and capture, MTP/PP/DP/PCP/DeepEP modes,
checkpoint/HumanEval32 accuracy, performance and the original15s model
shutdown gate remain open. Component constructor/transport/attention success
does not discharge those requirements. DeepSeek V4.1 checkpoint tests remain
deferred. AI self-review is not independent approval.
