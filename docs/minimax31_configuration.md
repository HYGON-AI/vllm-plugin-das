# MiniMax31 deployment configuration and environment review

MiniMax follows `adapter-das/docs/env-vars.md`: new features use configuration,
not private environment switches. All 20 new `HCU_MINIMAX_*` variables have been
removed from runtime source. `VLLM_PLUGINS` and cache settings owned by vLLM are
existing upstream interfaces, not new Plugin knobs.

| Removed inputs | Replacement and default |
|---|---|
| `HCU_MINIMAX_CAPTURE_CONFIG`, `HCU_MINIMAX_LOG_ALL_PCP_LAYOUT`, `HCU_MINIMAX_STATIC_ROUTE_CONTROL`, `HCU_MINIMAX_ROUTE_COUNTS_ANNOTATION` | Removed capture hooks, per-batch layout logs, route diagnostic module and annotations. No replacement debug switch. |
| `HCU_MINIMAX_DEEPEP_FIXEDCOUNTS`, `HCU_MINIMAX_SHARED_OVERLAP`, `HCU_MINIMAX_HIPC_PACK_TILED`, `HCU_MINIMAX_ALIGNED_MOE` | Fixed-count DeepEP and guarded shared overlap are defaults for the admitted PCP8/EP8 runner. HIPC uses the aligned executor and tiled movement by default. Short shared batches retain the stock lifecycle. |
| `HCU_MINIMAX_FP8_PCP`, `HCU_MINIMAX_FP8_INDEX` | Follow public `--kv-cache-dtype fp8` and `--attention-config '{"indexer_kv_dtype":"fp8_e4m3"}'`. No second boolean gate. BF16 deployments are not silently converted to FP8. |
| `HCU_MINIMAX_NATIVE_MSA`, `HCU_MINIMAX_MSA_EXTENSION_DIR` | A matching `additional_config.minimax31.msa_extension_dir` automatically enables native MSA. FP8 index requires it; ABI/hash checks remain mandatory. |
| `HCU_MINIMAX_STATIC_EPLB` | A validated `quantization_config.hcu_minimax_static_placement` automatically activates static136. No map means the 128-expert layout; no fabricated map or online EPLB. |
| `HCU_MINIMAX_ROUTED_BACKEND`, `HCU_MINIMAX_W8A8_BACKEND` | Versioned quantization configuration selects backends. Defaults: `deepgemm_hipc_v1` and `hipblaslt_v1`; explicit `bolt` / `triton` comparison configurations remain supported. Fallback defaults to `error`. |
| `HCU_MINIMAX_STATIC_DATASET_MANIFEST`, `HCU_MINIMAX_STATIC_RECEIPTS`, `HCU_MINIMAX_STATIC_READY_MANIFEST`, `HCU_MINIMAX_STATIC_CONSUMER_RECEIPTS` | Artifact paths in public `--additional-config`; required identity/evidence inputs, not switches. |
| `HCU_MINIMAX_STATIC_GENERATION` | Preload supervisor creates one fresh UUID generation for all eight spawn workers. Consumers verify the READY generation; no user environment override. |

## Artifact configuration

Use official vLLM `--additional-config` on producer and server. Only these five
keys under `minimax31` are accepted. Unknown keys, including debug/performance
flags, and non-string/empty paths fail closed. Other plugins' top-level
additional configuration is left untouched.

```json
{
  "minimax31": {
    "msa_extension_dir": "/artifacts/msa/build_lib",
    "dataset_manifest": "/artifacts/static-dataset.json",
    "loading_receipts": "/outputs/run/loading-receipts",
    "ready_manifest": "/outputs/run/producer-ready.json",
    "consumer_receipts": "/outputs/run/consumer-receipts"
  }
}
```

`msa_extension_dir` is necessary for native MSA / FP8 index; `dataset_manifest`
and `loading_receipts` are required for fresh static loading. IPC consumers
additionally need a matching `ready_manifest` and their own `consumer_receipts`.
Receipts are part of the loading/publication contract, not debug logging.
Normal safetensors loading does not require a producer, READY file or IPC.

Defaults remain scoped to the admitted MiniMax model, audited wheel, eager
TP1/PP1/DP1/PCP8/EP8/DCP1 and validated weight/cache contracts. They do not waive
architecture, layout, numeric, thread/stream or IPC identity checks. Changed
implementation/configuration requires a new honest identity and, for IPC, a
fresh producer; do not edit old hashes. Candidate status and incomplete full
checkpoint quality evidence remain unchanged.

The MSA provider's existing `MSA128_FP8_PREFILL_KERNEL` / `MSA128_FP8_PREFILL_SLOTS`
are external binary interfaces; this Plugin neither defines nor reads them.
Their settings must match the supplied provider and build identity.

## Review verification

CPU contracts cover real EngineArgs CLI transport, artifact validation,
per-runner configuration binding, FP8 cache guards, shared scheduling boundaries,
backend defaults, READY generations and cached MSA artifact drift. A source gate
rejects private MiniMax environment reads and capture hooks. Real EP8 default
path evidence is recorded in the MR; it is component numerical/lifecycle
validation, not a complete checkpoint accuracy result.

Long shared prefill overlap is attempted only for the audited HIPC/hipBLASLt MLP with an enabled safe stream. Retained Bolt/Triton configurations, disabled streams and other unsupported overlap conditions use the stock shared lifecycle; an occupied output slot remains an error. Failed static forwards release the host frame but require a new engine, rather than reusing uncertain GPU state.

## Paired Adapter prerequisite and worker startup

Merge/install [Adapter #128](https://github.com/HYGON-AI/adapter-das/pull/128)
before [Plugin #427](https://github.com/HYGON-AI/vllm-plugin-das/pull/427).
The minimum Adapter feature baseline is the `vllm_minimax_prefill` catalog and
startup-veto registration introduced by Adapter commit `1f643926a890749cd80008e94cc56b5ec6a0b265`;
a merged/released build must contain that functionality. An older Adapter fails
at model construction with the paired-catalog diagnostic, rather than silently
ignoring a requested backend. A container tag alone does not establish this floor.

Use the existing vLLM worker-startup interface:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

This is an upstream worker setting, not a new Plugin variable. The policy and
scratch ownership guards reject active state inherited by fork. Configuration
changes are rejected at the next planner/profile boundary and require restart.
The memory probe reserves maximum local K/V/index pack scratch before KV capacity
is computed; inference is restricted to that profiled stream and row budget.
