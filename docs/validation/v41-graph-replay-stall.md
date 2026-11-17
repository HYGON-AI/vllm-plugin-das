# V41 graph replay stall: failure contract and validation plan

Recorded before implementation on 2026-11-06. Plugin HEAD: 7ec3d795c8ca07bc7218170565aeac6db7b1c862. Existing C-group MoE and A-group graph changes are present; Core and Adapter remain read-only. No unit tests will be added.

## Established evidence and open causality

- Run `vllm_hcu/workspace/ifb/acc_d/20261106_194950` reaches PIECEWISE breakable graph capture and API startup. Streaming HumanEval stalls after 15 responses; non-streaming also stalls after 8 responses.
- The recorded rank-0 stack is in production `v41_indexer.py` at `decode.decode_lens.tolist()` while executing a breakable replay eager segment. This locates a synchronization wait; it alone does not prove a dependency cycle or exclude another rank/collective fault.
- Core breakable capture ends the capture segment before an eager callback and replays graph/eager callables in order. Host synchronization is not categorically forbidden by that API.
- All-rank stacks and fresh reproduction are required to establish the distributed wait mechanism. GPU utilization snapshots, an out-of-order DP stats warning, and HTTP 200 headers alone do not prove successful completion or the root cause.
- `/proc/<pid>/environ` does not reliably reflect Python updates after process startup; missing plugin variables there cannot establish that multiprocessing stripped them.

## Failure modes to cover before changing code

1. A GPU-to-host copy (`decode_lens.tolist`, per-request context `item`, or block validity `item`) waits behind cross-rank work during replay; fixing only one site relocates the wait.
2. Rank-local uneven/zero decode or mixed prefill leads to inconsistent graph/eager collective ordering or stale captured metadata.
3. A cached eager closure reads capture-time metadata instead of the current forward context; same descriptor reused with new contexts/lengths must produce current results.
4. Ratio-1/2 cache values use 16x16 SHUFFLE with a separate FP32 scale tail; interpreting NORMAL storage silently changes logits/top-k. Block size 1 is NORMAL and must still work.
5. FP8 dtype, FP32 weights/scales, reduction order, ReLU-before-head-reduction, negative weights, and causal masks drift from the independent logical-key reference.
6. Ragged/zero-query request lengths map compact Q/output rows to the wrong request, block table, effective context column, or padded candidate-bank row.
7. Candidate writer and consumer disagree on padded request-major rows; padded decode plus prefill overlap must retain explicit refusal; output/candidate tail rows must not be clobbered.
8. Causal lengths exceed the model/table capacity or physical page IDs are invalid. The GPU path must fail closed without illegal loads; masking invalid storage alone is not a correctness policy.
9. Max sequence length, page boundaries, 16-token shuffle tiles, nonmultiple tail, empty batch, all padding, single token/page, ratio-2 effective 2D lengths and changing same-bucket batches produce stale or uninitialized logits.
10. Runtime allocates unbounded temporary tensors for full cache/head dimensions, introduces host sync via GPU indexing/dynamic shape output, or changes the caller-owned output address.
11. Prefill gather/logits contains another synchronization point or the zero-token profile/dummy route diverges from the native contract.
12. Service startup succeeds while reference ownership or graph replay is unverified. Raw request/response JSON, source hashes, capture settings, bounded request deadlines, all-rank stalls, and final HumanEval reports must be preserved.
13. Evaluator timeout values are seconds; a value of 600000 is not 600000 ms. No indefinite evaluation retry or an aborted tool leaving duplicate evaluators should be mistaken for progress.

## Validation stages

- Preserve unchanged baseline source hashes and reproduce with the existing installed 0.33 service on nmz18; collect every engine/worker main-thread stack on a bounded request stall. Do not stop a healthy run merely because a log marker is absent.
- Before implementation, prepare repeatable real-HCU source integration for packed-cache decode/candidates (200 deterministic seeds plus listed edges) with independent logical keys and CPU reference. Assert no CUDA host-sync operations inside the real production callback using PyTorch sync-debug mode; save JSON/JSONL artifacts.
- Re-run the existing CPU source integration as a regression, without adding unit cases.
- Validate actual capture/replay with changed current metadata and same static buffers; compare all valid logits and top-k membership for ties, candidate-bank placement, output addresses, dtype/shape and cache immutability.
- Use the unchanged full-model graph topology: W4A8 TP1/DP8/EP8, PCP1/DCP1, DeepEP LL+DeepGEMM, PIECEWISE breakable, capture sizes 1/2/4/8/16/32, 256-token scheduler budget, Engram host route off. Save bounded staggered multi-request results and real HumanEval with max_tokens=16384 and thinking off. Graph remains the desired default. No package installation, Core/Adapter changes, commit or push.
