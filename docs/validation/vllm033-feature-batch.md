# Additional v0.33 feature observations

These observations use the same final production wheels as the initial
[migration record](vllm033-upgrade.md), on gfx936 / DTK 26.10 / Torch 2.11 /
Python 3.12. Core runtime Python sources remain unchanged. Installed modules,
checkpoint hashes, commands, resolved configuration, scores and comparison
details are recorded in the [candidate JSON](../../validation/v0.33.0/feature-batch-gfx936-candidate.json).
This is feature validation, not a new compatibility lock or last-good. Core
DTK-only !73 and Adapter admission #33 have merged; these observations reuse
their frozen production wheels.

## Model pairs

All arms keep prefix caching and the official default FULL_AND_PIECEWISE
Graph policy. Each uses four arithmetic/long-prefix smoke requests, two waves
of four concurrent arithmetic requests, and HumanEval/0–31 with temperature 0,
seed 0 and thinking disabled through the checkpoint's template. The GLM parser
is `glm45`; Qwen3 uses `qwen3`. Generated code is scored only in the existing
isolated evaluator, with uid 65534 and no network/GPU access. Raw and normalized
scorings remain separate. Paired checkpoint hashes and HTTP request bodies are
checked for equality; endpoint URLs and runtime feature switches differ.

| Checkpoint / path | raw | normalized | Observation |
| --- | --- | --- | --- |
| Qwen3-4B-Channel-INT8-w8a8, no Plugin | 0/32 | 29/32 | official compressed-tensors and Triton INT8 linear |
| Same checkpoint, Plugin + LightOp RMSNorm ON | 0/32 | 31/32 | no new failed IDs; newly passes /1,/10 |
| Qwen3-1.7B, TP2 Plugin Flash/BF16 | 1/32 | 28/32 | official RCCL communication |
| Same checkpoint/topology, Flash/E5M2 | 8/32 | 27/32 | newly fails /12; quality remains open |
| GLM-4.7-Flash, TP2+EP2/TRITON_MLA, MTP OFF | 0/32 | 31/32 | official MLA/MoE/collectives |
| Same checkpoint/topology, MTP3 | 0/32 | 31/32 | same normalized pass IDs; nonzero acceptance at all three positions |

Qwen3-1.7B is the 1.7-billion-parameter checkpoint, not Qwen3-17B. The E5M2
new failure stops normally at 199 completion tokens, compared with 177 in the
passing BF16 control; it is not a token-budget exhaustion. No tolerance was
introduced for that loss. The earlier Qwen2 E5M2 difference remains unresolved.

Both GLM arms use Core+Adapter without the Plugin, explicit `TRITON_MLA`,
official Triton BF16 MoE and RCCL allgather/reducescatter. They test upstream
capabilities on HCU; they do not prove automatic MLA priority, AITER MoE/Custom
AR or a new Plugin MTP implementation.

The RMSNorm pair compares a no-Plugin baseline with the Plugin's RMS switch
enabled, both using official TRITON_ATTN and the same INT8 linear provider.
The official read-only worker inspection, enabled compile factor, normal graph
and AOT wrapper calls identify the LightOp path. The changed score is an
observation on this protocol, not a general quality or performance claim.

## Official communication and hybrid kernels

The installed official `PyNcclCommunicator`/RCCL passes 24 cases per rank:
AllReduce, AllGather and ReduceScatter; BF16/FP16; 16, 2,048, 131,072 and
1,048,576 elements. Each case checks eager output and three Graph replays
after changing rank-specific values at the same input address, against exact
independent references. Graphs are released before communicator teardown.
This provides no AITER Custom AR, fused AR+RMS or cross-node evidence.

Fifty-two selected unmodified official test-function cases run against the
installed wheel: causal-conv updates (20), packed GDN decode (14), and sigmoid
gating/non-speculative/speculative updates (18). The source tests retain their
original numerical assertions. Cases cover width 3/4, indexed/padded state,
strided QKV, FP32 beta, null/PAD slots, GVA 4/8 and 4/12, FP16/BF16/FP32 and
one/three speculative tokens. These calls load the original files with runpy;
they are not a full pytest collection, fixture or CI-job result. All selected
functions are fixture-free, and their CUDA-alike platform condition applies
on this ROCm host. They do not validate the Plugin's unported BoltOPs paths.

An additional chunk-prefill probe compares the official implementation with
a direct FP64 CPU recurrence, using nonzero gates/initial states, head size
128, GVA 4/8 and 4/12, and unequal sequence lengths 17+129 and 63+65. It also
checks the caller-owned output buffer alias. All eight BF16/FP16 cases pass
with the upstream CPU GDN tolerance (rtol/atol 1e-2); observed maximum output
error is below 8.4e-5 and state error below 4.3e-4. Results are recorded separately
from decode and from full hybrid-model acceptance.

The same eight prefill contracts also pass three changed-input Graph replays
each (24 total), updating Q/K/V, gates, beta and initial state in place and
checking output/final state against the independent recurrence after every
replay. All input/output addresses remain fixed and caller output aliasing is
preserved. Maximum observed errors are below 1.4e-4 for output and 6.9e-4 for
state. The original launcher discovery conflict (zero kernels) and a separate
420-second cold-compilation timeout (18/24 completed replays) are retained.
The successful final attempt used a 900-second budget with the shared compiler
cache already partly warm; it is not a cold-start latency measurement.

The first MTP3 attempt reached both target/draft weight loading and target
compilation, but expired its original 900-second readiness budget with zero
inference. Its driver, progress and failure are retained; the second attempt
has a separately recorded 1800-second startup budget and the same 3600-second
overall budget. The checkpoint has one trained MTP layer reused for three draft
positions, rather than three distinct trained heads. The successful retry
records 2,144 draft iterations, 6,432 draft tokens and 2,709 accepted tokens
(42.12% accepted/drafted). Per-position accepted counts are 2,097 / 526 / 86;
these are the final observed metrics snapshot, which may lag final requests.
All 32 completions stop normally. Normalized passing IDs match the no-MTP
control exactly (both fail /26); raw outputs are not byte-identical. This is
bounded accuracy/functional evidence, not a throughput improvement claim.

## Performance and remaining scope

Ordinary RMSNorm was measured after warmup, in three paired/interleaved
repetitions of a 64-call Graph replayed 20 times. The 18 cases cover BF16/FP16,
rows 1/16/128 and widths 2,048/2,560/4,096. The native/LightOp median time ratio
ranges from 0.894 to 1.041; the median across cases is 0.981, with LightOp
faster in 8/18 cases. This mixed result does not support enabling the optional
provider globally. Its default remains OFF. No fused-RMS or model-throughput
claim is made.

All six completed model arms exit zero. Resource-tracker warnings remain
recorded: semaphore warnings occur in some Qwen runs, and shared-memory
warnings occur in both GLM arms. These are not claimed as clean shutdowns.

Coverage, numerical accuracy, stability and performance remain separate:
short concurrent waves are not sustained stress or idle-rank DP/EP coverage;
operator GDN success is not full hybrid prefix/resume or model acceptance.
MTP1/2, AITER Custom AR, FlashMLA, sparse MLA, W4A8 lifecycle, broader topology
and model/performance gates remain open. All owned servers have exited and
their ports are closed; all eight cards returned to about 2 MiB. GPU5/6 are
excluded from every launcher. No v0.32 result is relabeled here.
