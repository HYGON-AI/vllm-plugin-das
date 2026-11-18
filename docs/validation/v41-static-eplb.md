# DeepSeek V4.1 static EPLB migration

## Sources and ownership

The initial CPU migration used Plugin `89ee352c671d49fd1702029d6388caeb2919a434`
on `HYGON-AI/vllm-plugin-das` `v0.33.0`, paired with Adapter
`4b9ea6d761a67ddc677414d4c50a74b83ba46e01` on `HYGON-AI/adapter-das` `main`.
The old feature input is `74086113a97751920ee6820dfb23cd8ac0c6b430` in
`yueyang123321/vllm-plugin-das`.

| Previous component | Owner and migration decision | Affected calls and checks |
| --- | --- | --- |
| Version-2 map validation | Plugin, retain immutable plan and fingerprint validation | `bind_placement`, map replay |
| Expert checkpoint fan-out | Plugin, retain each logical expert's physical-slot loading | `load_weights`, shared-tail and dummy-load checks |
| Replica selection | Plugin, retain `nearest` and deterministic `locality_fair` | `install_routing`, primary-replica checks |
| Offline calibration | Plugin, retain Core counters and balancing-policy proposal | outer `forward`, padding/startup exclusion and atomic-map checks |
| Owned model constructors | Plugin, compose the current FP4 attention and PCP Engram with static MoE construction | constructor namespace and outer selection checks |
| Decoder SWA replay | Existing PCP backbone, inherit its current gate | accepted replay and existing refusal cases |
| DeepGEMM physical capacity | Plugin, admit validated static counts through the scoped HT method | 384 logical, 416 physical, 52 local at EP8 |
| PCP admission and preparation | Current Adapter, reuse existing public binding | existing PCP/FP4 regression files |
| Dummy-input randomization/profile worker | Current official implementation, retain existing owner | existing PCP worker regression files |
| DSpark compression history and draft | Adapter owns PCP compressor history; Plugin separates replicated draft attention from PCP-owned MoE and sizes the FP4 profile workspace | eager V4.1 PCP DSpark with adaptive verification disabled |

Core continues to own routing kernels, scheduling, allocation, checkpoint
iteration, DeepEP dispatch/combine and the balancing policy. The factory
receives per-layer `enable_eplb=True` to construct Core's routing state and
physical capacity. Global `--enable-eplb` stays disabled, so the online
rearrangement lifecycle is absent.

The current MiniMax31 static implementation has a different model contract,
quantization format and map schema. DeepSeek retains its previous version-2
maps and shared expert handling.

## Configuration

For TP1/DP1/PCP8/EP8 Channel-FP8 with FP4 indexer and decoder SWA replay, use
the existing PCP and FP4 provider settings, `deep_gemm`,
`deepep_high_throughput` and eager MRV2. Add the following sidecar to the
service's existing `--additional-config` JSON.

```json
{
  "hcu_static_eplb": {
    "expert_map_path": "/workspace/eplb_official_20261117/assets/expert_map.json",
    "num_redundant_experts": 32,
    "static_dispatch_policy": "locality_fair"
  }
}
```

The map must match the native `DeepseekV41LLMForCausalLM` key, exact model
path, MoE layer count and expert counts. At 40 layers and 384 logical
experts, 32 redundant experts produce 416 physical slots and 52 slots per
EP8 rank. Both deployed FP8 and FP4 indexer paths share the same static map
loading lifecycle.

Calibration uses identity placement with zero deployed redundancy. The
configured redundant count determines the proposed map. Replace
`expert_map_path` with `expert_map_record_path` and configure a start marker,
for example when recording a single request:

```json
{
  "hcu_static_eplb": {
    "expert_map_record_path": "/shared/run/expert_map.json",
    "record_start_file": "/shared/run/record.start",
    "record_step_interval": 1,
    "num_redundant_experts": 32
  }
}
```

1. Use a new run directory or remove the old start marker before launching the
   calibration service. The marker must be absent while every rank starts up.
2. After `/v1/models` reports the model ready, create the marker at the
   configured path, for example with `touch /shared/run/record.start`. All EP
   ranks must see that file before you send the first calibration request.
3. Send the calibration requests and keep the marker in place until the
   service stops. Verify that the proposal was written, then start a fresh
   service with the resulting file as `expert_map_path`. Omit the recording
   fields from the deployment configuration.

Padding remains excluded. Without `record_start_file`, dummy forwards that
have attention metadata can enter the calibration counts.

For the validated W4A8 + FP4 + SWA + DSpark configuration, use eager V2 with
TP1/DP1/PP1/DCP1, PCP8/EP8 and `deepep_high_throughput`. Use this speculative
configuration with adaptive verification disabled:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true,"moe_backend":"deep_gemm"}'
```

PCP DSpark requires an Adapter build containing the companion PCP compressor
implementation from [Adapter PR #139](https://github.com/HYGON-AI/adapter-das/pull/139).
Existing non-PCP DSpark and non-speculative W4A8 deployments still support the
declared Adapter 0.5.22 minimum.

Runtime EPLB, KV transfer, DBO and unsupported topology combinations
are rejected by the existing scoped admission checks. The default MegaMoE
implementation retains its original identity-layout restrictions.

## CPU verification

The replay script uses an existing Python 3.12/Core environment, hidden GPUs
and `HCU_ADAPTER_HOST=none`. Source overrides are confined to the test process.
`-I -S` prevents installed startup hooks from binding an older Adapter before
the selected source paths.

```bash
VLLM_TEST_PYTHON=/usr/local/bin/python3 \
  bash tests/vllm033/run_v41_static_eplb_cpu.sh /path/to/new-output
```

The pytest integration file `tests/vllm033/test_v41_static_eplb.py` covers
the previous replay contracts, FP4/static constructor selection, inherited
SWA gate, physical counts, backend exclusions and load ordering.
Distributed membership, heavy allocation and native kernels are explicit
CPU doubles. The separate device result below covers full-model execution for
the stated configuration.

## Source-bound results on 2026-11-17

Shenzhen `10.17.151.1`, container `vllm033_official_20261116_leyang`,
Python 3.12, installed Core `5afd482`, Torch 2.11 and the source pair above
were used for CPU execution. GPU visibility was empty and Torch reported
`gpu_initialized=false` after the combined suite.

| Check | Result |
| --- | --- |
| Original static EPLB source replay | 19 passed |
| Static EPLB pytest integration, including the original replay cases | 37 passed |
| Combined eight-file CPU regression | 206 passed, 2 failed |
| Same seven existing files at unmodified Plugin `89ee352` | 169 passed, the same 2 failed |
| Original deployment map | SHA-256 and full 40 × 416 layout validated |
| Shell syntax and Git whitespace check | passed |

The two retained baseline failures are
`test_dsv41_megamoe_cpu.py::test_native_provider_signature`, because the
installed grouped-GEMM package lacks `deepgemm.mega`, and
`test_v41_indexer_review_fixes.py::test_profile_pass_reserves_the_staging_pool`,
whose fixture references the removed `fp4_indexer_lightop.get_current_vllm_config`.
Original failures and reruns remain in separate evidence directories.

Evidence lives under
`/root/leyang/workspace/eplb_official_20261117/validation_runs`, with final
directories `cpu_targeted_final`, `cpu_combined_final` and `cpu_baseline`.
The reused map is outside the repository at
`/root/leyang/workspace/eplb_official_20261117/assets/expert_map.json`, with
SHA-256 `b545eb9e13a66431317f2842eda7e0f89dbb3857bff9a86ada9433cd548e09d0`.

## Device validation on 2026-11-18

Adapter `cbecf5d` and Plugin `c1f6799e` were tested together on eight HCUs
with Channel-INT4 W4A8, PCP8/EP8, FP4 indexer, SWA bounded replay, static EPLB32
and DSpark5. Calibration used one 32,768-token input with one output token
after arming the start marker. The restarted service loaded the proposal,
exported all eight rank traces, and scored 99/100 on HumanEval pass@1.

This run used a 33,792-token model limit, 32 maximum sequences, and a fixed
1.25 GiB KV cache per card. Prefix caching and KV transfer were disabled.
The three unprofiled single-request rounds averaged 1.398 s.
