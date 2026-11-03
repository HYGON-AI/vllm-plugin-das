# Qwen3.5 W8A8 full-model comparison

Qwen3.5-35B-A3B-W8A8, single gfx936, official vLLM v0.33.0 with Adapter,
comparing official Triton MoE against explicit Plugin AITER MoE. Both arms
completed. [Candidate, checkpoint identity and raw-evidence hashes](../../validation/v0.33.0/int8-model-gfx936-candidate.json).
This supersedes the pending experiment in the [admission review](vllm033-int8-admission.md).
AI assistance was used; `last_good=false`.

| HumanEval32 | Official Triton MoE | Plugin AITER MoE |
| --- | --- | --- |
| Raw | 31/32 | 32/32 |
| Normalized | 32/32 | 32/32 |
| Finish reason | 32 stop | 32 stop |
| Completion tokens | 4,872 | 4,896 |

All checkpoint/config/tokenizer hashes match. Each of the 32 request bodies,
wire bodies and task IDs is identical across arms; only the loopback endpoint
differs. Protocol hashes remain distinct because they include that URL and the
corresponding requests-file hash. Prompt tokens total 4,203 in both arms. Scoring
uses the same isolated uid65534 environment with external networking blocked.
The official raw failure is HumanEval/24, invalid syntax; the existing unchanged
normalizer resolves it. No normalized task regresses. Text/token differences
remain possible; equal scores are not bit-identical model output.

The actual worker metadata identifies all 40 routed-expert layers as official
`TritonExperts` or Plugin `HcuAiterInt8Experts`, respectively. Both have INT8
weights `[256,1024,2048]` / `[256,2048,512]`, and float32 channel scales
`[256,1024,1]` / `[256,2048,1]`. Both use official Triton attention and dense
INT8 kernels, BF16 activations and the checkpoint's dynamic-token quantization.
The AITER arm reports native ASM configurations, loads both stage code objects,
and has no reported provider fallback. Decode and Graph shapes M1/M2/M4 and
prefill M1024 enter ASM. This is execution evidence beyond the backend flag.

Both arms retain official FULL_AND_PIECEWISE Graph, pass four smoke requests
and two waves of four concurrent requests, and finish both Graph capture
phases. MTP is off, TP is1, and there is no distributed EP claim. The profile
explicitly sets `combo_kernels=False` in both arms; no default-policy acceptance
or model speedup is claimed.

**Prefix reuse is not accepted by this experiment.** Prefix caching is enabled,
but both report 6,239 queried tokens and zero hit tokens. The real manager page
is1056 and these requests do not establish positive reuse. Preserve this as a
separate missing test rather than calling the enabled switch a passing prefix
test. The earlier0.8B model's positive fine-grained hits do not transfer here.

The full-model arm uses the frozen ASM mathematics wheel from production
`529855a5`, SHA256 `e905ce20ef0a9025b32d478806af6c015da0863f94965f3455793297e7450925`.
The later `ec01e209` admission-only wheel separately passes262 portable and11
native tests; it rejects expert biases and parameterized activations that this
checkpoint does not use. This evidence does not certify an untested build,
arbitrary model/layout, unchanged vendor-Triton pipeline, E5M2, or distributed EP.
Core runtime and Adapter production remain unchanged.

Both services exited normally, owned ports18159/18160 are closed, and device
memory returned to about2MiB per card. GPU5/6 and unrelated services were excluded;
no device reset. **PR330 remains on hold** for the earlier E5M2/hybrid quality
differences and unchanged vendor-Triton strict numerical gate.
