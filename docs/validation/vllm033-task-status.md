# v0.33 ownership and E5M2 investigation

The current target is official vLLM **v0.33.0**, commit
`4ba787ec9959e566d3b587a796dff07673324da5`, with Core/Plugin release branches
`v0.33.0` and Adapter admission on `main`. The scope remains the three-layer
migration described in [the upgrade record](vllm033-upgrade.md). Historical
v0.32 results do not certify this target. Core runtime Python and all production
wheel bytes remain unchanged in this follow-up.

The [candidate JSON](../../validation/v0.33.0/e5m2-investigation-gfx936-candidate.json)
records installed paths, checkpoint/tool hashes, configuration, isolated
scores, first-token-difference evidence and cleanup. It is not a compatibility
lock; `last_good=false`.

| Requirement | v0.33 owner and current result | Remaining gate |
| --- | --- | --- |
| Explicit BF16/FP16 KV names | Official native dtype mapping; Plugin forwards unchanged | No production normalization patch needed |
| E5M2 cache bytes and attention | Plugin E5M2 writer/descales + vendor attention; expanded numerical/Graph cases pass | Model-quality difference remains OPEN |
| MTP1/2/3 | Official Core+Adapter; previous GLM OFF/3 pair completed | MTP1/2, broader topology and performance |
| AITER MoE/Custom AR, W4A8 | Recheck current registry, config/ABI and lifecycle before migrating | Official RCCL or dense INT8 success does not certify these paths |
| Hybrid prefix/resume | Official manager/allocation first; Plugin only for proven missing capability | Real model/state-page acceptance beyond GDN operator tests |

## Native KV-name takeover

Official `csrc/core/vec_dtypes/dtype_fp8.cuh` maps `auto`, `float16` and
`bfloat16` to `kAuto`, and retains a separate `kFp8E5M2` enum. The installed
Plugin `do_kv_cache_update` passes eight exact-cache checks across both layouts,
BF16/FP16 activations, and explicit/default names. A historical native-writer
rejection does not justify carrying its old normalization patch into v0.33.

Regression tests now cover all three names at the forwarding boundary and
exercise the Plugin update entry before vendor attention. The changed device
suite passes **52 tests**, including native BF16/FP16 writing, E5M2 writing,
non-unit scales and changed-input Graph. The complete portable v0.33 suite
passes **231 tests, 58 skipped** (device cases). These are new full-batch counts;
the earlier 225/42 record remains historical. Ruff F checks and diff whitespace
checks pass.

## E5M2 model-shape and long-context checks

Twenty-four installed-provider cases pass independent FP32 causal attention
on dequantized OCP E5M2, retaining the existing `atol=rtol=0.025` limit:

- Qwen2-0.5B geometry: 14 query heads / 2 KV heads / D64.
- Qwen3-1.7B TP2 geometry: 8 query heads / 4 KV heads / D128.
- Both HND/NHD layouts, padded block strides and randomized physical pages.
- Query-length pairs 1+1, 3+11, 128+17 or 129+65; KV lengths 2057+3075 and
  8193+9231, including the historical long-prefix gather boundary.

Each case checks eager output and three same-address Graph replays with new
Q/K/V and non-unit scales: **72 changed-input replays** in total. Physical K/V
bytes match an independent E5M2 conversion exactly. Maximum observed attention
output error is below **0.00166**. This validates the exercised mechanism,
not the original model-accuracy requirement or full long-context model output.

## Reproduced model difference and captured token IDs

Two diagnostic Qwen2-0.5B-Instruct runs retain default Graph, prefix caching,
checkpoint, model/sampling request parameters, and all 32 original prompts in
order. The additional API fields request token IDs and five raw top-logprobs;
no production function is wrapped or replaced. Generated text is scored only
in the existing isolated evaluator (uid 65534, no network/GPU).

Both arms reproduce **all 32 raw generated texts byte-for-byte** from their
uninstrumented attempt3 runs. Raw scores remain 0/32 for both; normalized scores
remain **16/32 BF16 and 13/32 E5M2**, with the same passing IDs. The unaccepted
quality loss is retained.

The five newly failing tasks (/0,/8,/16,/25,/31) share their first three
generated token IDs and first differ at token **4**. The early choices differ
among `import`, `from` and `def`; those valid Python tokens are an observed
boundary, not themselves an explanation of the later failed unit tests.
For each task, both services receive the identical captured prompt IDs plus
the identical generated prefix. Two new completion requests per arm reproduce
that arm's original next-token choice, with identical repeated raw logprobs.
Prefix caching stays ON. These new requests are not graph-internal layer
snapshots and do not establish the earliest internal numerical divergence.

The checkpoint's `generation_config.json` supplies repetition penalty **1.1**;
resolved logprobs mode is **raw_logprobs**. Reported top-logprobs precede those
sampling penalties, and the API may insert the sampled token first in the
list. Sort by logprob when discussing raw rank, and do not infer the processed
greedy margin or a sampler defect from that list. The existing protocol is
preserved rather than silently replacing its generation defaults.

Observed BF16/E5M2 cache capacities are 1,230,016 / 2,460,416 tokens at the
same memory-utilization setting (about 2x); that capacity benefit does not
waive the accuracy loss. An independent Transformers experiment also uses those frozen token prefixes:
one BF16 eager model and a public `DynamicCache.update` subclass that changes
only K/V by an OCP E5M2 round trip at unit scale. It calls neither vLLM nor the
vendor attention provider. With the checkpoint repetition penalty, **all ten
next-token choices** (five tasks x two arms) match their serving counterparts.
All five reference pairs already differ at layer0 attention output before
`o_proj` (last-token max difference 0.059–0.076, cosine 0.896–0.920). The
reference's layer snapshots, raw/processed logits and choices are preserved.

Thus KV quantization alone is sufficient to reproduce these five first-token
splits in an independent implementation; a vLLM/vendor defect is not needed
to explain those observations. This is not a unique root-cause proof for the
entire later completion, all HumanEval failures or Qwen3. It does not justify
a Core patch, tolerance relaxation, scale tuning to the failed tasks, or
relabeling the original model gate as passed. Quality remediation still needs
representative calibration or a separately validated E5M2 implementation. The
independent eager reference is not a replacement for the original Graph gate.

Both services exit zero with a retained resource-tracker semaphore warning.
Owned ports are closed and all cards return to about 2 MiB. GPU5/6 and the
pre-existing port18127 service are untouched. The Plugin MR remains a draft.
