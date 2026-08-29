# DeepSeek V4 DSpark Mooncake P/D Design

## Goal

Enable Mooncake prefill/decode disaggregation for the DeepSeek-V4-Flash-0731
Channel FP8 and INT8 W8A8 checkpoints while retaining the existing DP+EP,
DeepEP, DeepGEMM, and DSpark behavior from the dependent DSpark change.

The target single-node topology is:

- prefill: HCU 0-3, DP=4, EP enabled;
- decode: HCU 4-7, DP=4, EP enabled;
- KV transfer: `MooncakeConnector`;
- speculative decoding: DSpark through the standard
  `--speculative-config` interface;
- accuracy gate: exactly 32 HumanEval samples per checkpoint through the P/D
  proxy/router endpoint.

## Upstream and Existing Plugin State

vLLM merge commit `0feca7ffa8f626f51f5ea262eb586a3bbe704991`
(PR #46807) added group-aware Mooncake transfer metadata, logical-to-physical
block mapping, and DeepSeek-V4 MLA support. Those core changes are already
present in the plugin's owned Mooncake connector.

Plugin commit `c5e191a0312bef00ece5929e23b3a0a68fb8375f` additionally fixed the HCU
DeepSeek-V4 physical MLA block stride and previously validated Mooncake P/D
traffic with zero failed transfers and GSM8K accuracy of 0.9659. Later commit
`c0cfe2ef9a370ba84fccde544aa5d50046b3fb82` replaced patch-fragment ownership
with an explicit connector implementation and contract tests.

The current incompatibility is therefore not missing Mooncake MLA machinery.
It is the conservative cross-configuration guard introduced by the dependent
DSpark change: every DSpark configuration with a non-null KV transfer config is
rejected before model loading.

The official recipe command synthesizer allows the DSpark feature under
`pd_cluster` and emits its standard speculative configuration into both role
commands. The hand-written Mooncake guide omits speculative decoding, so HCU
runtime validation remains required rather than treating recipe composition as
proof of correctness.

## Scope

In scope:

- DeepSeek-V4 DSpark with `MooncakeConnector` P/D disaggregation;
- FP8 and INT8 Channel W8A8 checkpoints;
- one-node 4-prefill plus 4-decode DP+EP validation;
- standard vLLM CLI configuration, without new P-side/D-side backend flags;
- Mooncake transfer evidence, DSpark draft/acceptance evidence, DeepEP and
  DeepGEMM selection evidence, semantic smoke tests, and HumanEval-32 accuracy;
- launch/cleanup/accuracy documentation and a repeatable integration harness.

Out of scope:

- PCP with P/D disaggregation;
- DSpark with connectors other than Mooncake;
- pipeline-parallel P/D;
- cross-node Mooncake or transport performance tuning;
- changes to the already-uploaded dependent DSpark MR.

## Configuration Contract

Replace the broad DSpark-plus-KV-transfer rejection with a narrow allowlist:

1. If DSpark is disabled, preserve current behavior.
2. If DSpark is enabled and no KV connector is configured, preserve current
   behavior.
3. If DSpark is enabled with `MooncakeConnector`, allow configuration creation.
4. If DSpark is enabled with any other connector, reject before model loading
   with a connector-specific error.
5. Retain the existing PCP-plus-P/D rejection independently; this change must
   not enable PCP+DSpark+P/D.

No quantization-specific branch belongs in this validation because FP8 and
INT8 share the model architecture and KV-cache layout. Checkpoint-specific
runtime tests provide the coverage for both weight paths.

## Runtime Topology

Both endpoints use the normal vLLM arguments:

- `--data-parallel-size 4`
- `--enable-expert-parallel`
- `--kv-cache-dtype fp8`
- the DSpark `--speculative-config` used by the existing deployment;
- `--kv-transfer-config` selecting `MooncakeConnector` and the appropriate
  Mooncake role supported by the selected router/proxy.

The launch harness pins prefill to HCU 0-3 and decode to HCU 4-7, assigns
distinct HTTP, DP RPC, and Mooncake bootstrap ports, waits for both health
endpoints, starts the Mooncake-aware router/proxy, and guarantees cleanup on
failure or interruption.

The initial implementation uses the plugin's already-supported Mooncake
producer/consumer routing contract. `kv_both` may be used only if the installed
router and the HCU connector pass an explicit contract smoke test; command
appearance alone is insufficient.

DSpark remains configured through the official `--speculative-config` path.
The implementation will not add custom `p`/`d`, high-throughput/low-latency,
MoE-backend, DeepEP, or DeepGEMM command-line switches. Existing automatic
selection remains responsible for those backends.

## Validation

Unit and contract tests must first fail against the existing guard, then prove:

- Mooncake is the only KV connector allowed with DSpark;
- unsupported connectors still fail before model loading;
- PCP plus any P/D connector remains rejected;
- unrelated speculative methods and unrelated models retain current behavior;
- Mooncake connector metadata/layout contracts used by DeepSeek-V4 remain
  intact.

For each of the FP8 and INT8 checkpoints, runtime validation must prove:

1. Four prefill and four decode DP ranks become ready on the intended HCUs.
2. Expert parallelism is active in both pools.
3. Mooncake registers the MLA cache and completes at least one real transfer
   with no failed send/receive count.
4. The decode path records DSpark draft and accepted tokens.
5. DeepEP and DeepGEMM resolve through the existing automatic backend logic.
6. A deterministic semantic smoke request succeeds through the P/D endpoint.
7. HumanEval produces and reviews exactly 32 records, with the score reported
   separately for FP8 and INT8 and compared with the non-P/D baseline.

Any startup-only success, direct request to only one endpoint, or evaluation
without confirmed KV transfer is not considered a pass.

## Failure Handling and Compatibility

The harness will capture separate prefill, decode, and router logs. On failure,
it reports the first relevant error and always terminates only the process IDs
it created. It will not use broad process-kill patterns.

The configuration allowlist prevents accidental enablement for unvalidated KV
connectors. The retained PCP guard prevents this work from broadening the
previously excluded PCP+DSpark scope. The Mooncake connector itself will be
changed only if a failing unit or real transfer test demonstrates a concrete
gap; upstream code will not be copied wholesale.

## Delivery

This work is a stacked MR based on commit
`ae6a19481a71241243971d4f6f1162b88eea091e` from the existing DeepSeek-V4
DSpark MR. The new MR contains only P/D-specific code, tests, harness, and
documentation. After the dependency merges, its base can be retargeted to
`v0.25.1` without duplicating the DSpark implementation.
