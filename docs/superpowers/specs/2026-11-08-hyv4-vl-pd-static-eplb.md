# HYV4-VL image PD and static EPLB design

The user selected multimodal PD + offline EPLB adaptation on 2026-11-08. Continue the prior P PCP16/EP16/TP1 HT and D DP16/EP16/TP1 LL deployment with the new `/data_hy4_w4a8/Hy4-testing-Channel-INT4-w4a8` checkpoint. Source is v0.25.1 `21065784` + MR338 `c758f415`, preserved in local benchmark snapshot `e13e5cf3`.

## Intended behavior

Images, text, multiple images and chunked/cached image windows retain the native processor/encoder/cache/M-RoPE semantics through P prefill and D decoding. Real routing remains enabled. P loads a newly generated VL-only static R0 or opt-in R16 JSON; D uses the native LL layout and does not load a static map. MTP stays disabled. Preserve existing text serving and reject unrelated multimodal PCP variants.

## Runtime design

1. Reuse native MRV2 `DefaultModelState`, encoder/cache lifecycle, request admission/removal and RopeState through the public model `get_model_state_cls()` extension. Do not fork those implementations or change the frozen vLLM core.
2. For real PCP steps, run native image merge and model-position preparation on the saved global batch. Gather `[G,H]` embeddings on dimension0 and `[3,G]` rotary positions on dimension1 using the existing rank token permutation into distinct persistent local buffers. Mask `local_batch.is_padding`, including collective tails, but preserve non-owner replicated decode lanes.
3. Keep `InputBatch.positions` one-dimensional and unchanged. All KV slots, causal/sparse lengths and Mooncake transfer continue using canonical token ordinals. Rotary positions may differ from those ordinals; decode must retain the native image delta.
4. Carry explicit scoped dummy status from HCU `execute_model`. Profiling and later dummy calls must never read a stale saved global batch or live request RoPE. Preserve persistent-buffer addresses used by native PCP1 graph preparation. Initial P PCP16 image execution is eager.
5. Initially execute/cache the native vision tower on each PCP rank (TP1), then select language inputs. A rank0 vision/broadcast optimization is separate from correctness bring-up.
6. Native MRV2 currently omits real vision/cache profiling. Add an image-scoped HCU profile context which exercises registry-generated worst-case image input and accounts for the configured encoder cache budget through decoder profiling. Include full per-rank vision peak (not divided by PCP), persistent global/local embedding buffers and real encoder outputs. Clean only profiling-owned allocations. Derive limits/budgets from existing native utilities, not a fixed arbitrary KV cap.
7. Narrowly extend VL fixed HT/LL selection and opt-in static HT scratch ownership. Do not change the communicator ABI exact-type guard or widen DeepEP auto routing unnecessarily. Generic static loader/parser inheritance already works and should remain generic.

## Supported initial P scope

Exact `HYV4VLForConditionalGeneration`, image inputs, MRV2, PP1/TP1/PCP16/DP1/DCP1/EP16, fixed DeepEP HT with DeepGEMM, compressed-tensors W4A8, eager, existing Mooncake producer, static R0 or experimental R16. Preserve the existing rejections for other model families/topologies, MTP, LoRA, KV offload, video, encoder-only, precomputed embeddings, pruning and alternate projector contracts.

The compressed-tensors method label alone does not prove W4A8. Reject ordinary non-packed metadata early; before allocating the vision tower, validate every resolved routed MoE method against the existing compressed-W4A8 marker/static capability. This uses the native resolved scheme instead of duplicating compressed-tensors regex target matching. The VL fixed LL addition is confined to the intended D PP1/TP1/PCP1/DP16/EP16 lane; existing text predicates remain unchanged.

## Calibration and measurement

Use a VL-only identity R0 bootstrap map with 77 MoE rows and256 logical experts. Collect image-workload routing counts with an opt-in external recorder, generate a fresh R16 map keyed `HYV4VLForConditionalGeneration`, and omit the unused MTP map. Check loaded map bindings, inverse replica sets and sampled weight/scale rows; for R16, physical histograms have272 entries and must scatter-add back to256 logical counts. Report native load max/mean and coefficient of variation, separating calibration and held-out verification. No routing simulation or counters in formal performance.

Validation ladder: CPU contracts → existing text regressions → W4A8 VL reference smoke → MRV2 image/position tests → P PCP16 image requests → real PD transport → R0/R16 precision → performance with a documented cache protocol. Compare image-sensitive responses, full original RealWorldQA scoring and raw answers; do not infer numerical equivalence from equal scores or sampled weights. Avoid KV priming in formal MM performance. Release owned GPU sessions on failure/completion.

## Main failure classes

- A second PCP segment reads the first segment's image slice or RoPE coordinates.
- Image rotary values accidentally change logical KV offsets or sparse causal lengths.
- Padding, decode replicas, request-slot reuse or dummy calls read stale/cross-request data.
- D skips required suffix image encoding or loses image-grid metadata during transfer.
- KV sizing omits real vision/cache peak and causes post-startup OOM.
