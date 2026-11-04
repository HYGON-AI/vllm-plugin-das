# Stage 2 optimization evidence

Skill repository: BBuf/AI-Infra-Auto-Driven-SKILLS, commit
371dee04ee30ed918b240823b2a12c139d1a67fa.

Read:
- skills/llm-torch-profiler-analysis/SKILL.md and numerical/dispatch heuristics.
- model-pr-optimization-history/SKILL.md.
- model-pr-optimization-history/vllm/hunyuan4/README.zh.md.
- docs/upstream-source-contracts.md, vLLM profiler section.
- skills/llm-serving-auto-benchmark/references/paired-validation.md.

History identifies upstream vLLM #57526 (ROCm backbone compilation) and #57811
(indexer completion event for graph capture). These generated PR inventories
are leads, not manually audited diffs or evidence of HCU compatibility. No code
from those PRs is copied. The installed HCU path already uses a coarse eager
break around indexer/MLA; its source must remain authoritative.

Initial experiment: enable the installed breakable CUDA graph implementation
for decode only, capture sizes 1/2, keeping TP8, Triton channel FP8, native
operators, image resolution, batch limits and all sampling settings unchanged.
No speculative decoding, lower precision or image token reduction.

The existing text-only architecture automatically selects breakable graphs;
the VL architecture does not. Use an explicit environment/CLI experiment,
without changing default behavior until validation completes.

Temporary localhost-only diagnostic worker can force dispatcher mode NONE or
restore graph dispatch on the same loaded weights. This provides an eager
runtime control; source revision and runtime mode must be reported separately.
Counters and profiler traces must prove actual decode replay, not just capture.

Precision acceptance: no HTTP/empty/truncated results; full original 765-item
RealWorldQA paired evaluation, same question/scorer/hashes. A fixed-concurrency-1
pair controls batch formation; require identical raw predictions for a strict
prediction-invariance claim. Concurrency-2 differences remain separately visible.
Task-score parity alone is insufficient proof of numerical equality.

Target: another 50% request-throughput improvement relative to stage-1 optimized
serving, at the same c=1/c=2 workload; report latency separately. A changed batch
or cache policy would be deployment tuning and must not be called a code-only gain.

Additional source screen: installed AiterExperts forwards scales but does not
forward quant_config.gemm1_clamp_limit to fused_moe; the HCU FP8 adapter also
rejects a non-default swiglu_limit ABI argument. This checkpoint requests a
positive swiglu_limit. Therefore an AITER backend switch is excluded from the
initial precision-preserving candidate set, absent an explicit clamp audit.

Profiler follow-up: eight real rank traces were analyzed with the provided
script, including its kernel, overlap and fusion tables. Actual installed source
and runtime warning confirm missing BW1101 E256/N256 Triton tuning. The profiler
exhibited duplicate ROCTracer flows and a VMFault after stopping; preserve this
failure and avoid using the instrumented window as a steady performance claim.
The subsequent paired test uses a fresh profiler-free server.
