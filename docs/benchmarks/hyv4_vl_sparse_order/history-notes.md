# Source and history decisions

Reused AI-Infra-Auto-Driven-SKILLS pinned at
371dee04ee30ed918b240823b2a12c139d1a67fa. Read the Hunyuan4 vLLM dossier
and upstream source contracts. Historical #57811 identifies an indexer event
fix for full graph capture; it cannot alone explain the current eager reproducer.
Current HCU source, rather than that NVIDIA path, is inspected before patching.

Actual source path:
HYV4 Indexer -> SparseAttnIndexer.forward_hip ->
rocm_aiter_sparse_attn_indexer_native -> native LightOp decode Top-K ->
HYV4 sink-capable BF16 FlashMLA sparse attention (auto/BF16 KV).
No torch.profiler is configured, because its previously recorded graph-stop
VMFault remains unresolved and profiling is not needed for this causal probe.

The newly measured canonical intervention is version 2: every RPC invocation
replaces its closure, and all eight ranks return version/selector metadata.
This is distinct from the invalid source-only edit in the prior report.
Retain native score computation and Top-K membership. Canonicalize active
logical-index rows in HYV4-VL after the producer, before slot conversion and
shared consumers. Do not replace native selection with sorted-score PyTorch:
that is a separate numerical reference and is not needed by this fix.
