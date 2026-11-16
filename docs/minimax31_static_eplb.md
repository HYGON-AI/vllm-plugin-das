# MiniMax M3.1 static physical placement

This opt-in path separates **128 logical experts** from **136 physical slots**, with17 slots on each of8 PCP/EP ranks. It targets the audited vLLM0.33 wheel, TP1/PP1/DP1/PCP8/EP8/DCP1, eager text inference, HIPC routed GEMMs, hipBLASLt W8 linears and strict zero-copy IPC. Online runtime EPLB updates remain disabled; the immutable physical map provides static expert load balancing.

## Construction and loading

The scoped MiniMax factory keeps `num_experts=128` and `num_redundant_experts=0` at the stock factory boundary. The stock redundancy assertion is unchanged. The public `moe_config_cls` and `routed_experts_cls` seams select `StaticMoEConfig` and `StaticRoutedExperts`; their explicit physical-capacity contract sets MoE storage/layout counts to136/17 while preserving `num_logical_experts=128`. The router, gate, correction bias and HF expert count remain128. Physical rank ownership is linear in the136-slot address space; a separate immutable table maps those slots to logical checkpoint experts.

The outer conditional-generation wrapper scopes the validated contract across the wheel's nested MiniMax text-model construction. It allows inheritance only by `MiniMaxM3SparseForCausalLM` and resets the context on exit. It does not alter the wheel's HF objects or enable online EPLB.

The static checkpoint loader reads logical IDs0..127 and fills every local slot assigned to that logical owner, including replicas. All17 slots require both W13 shards, W2 and all corresponding FP32 scales. Every copy is checked in full. Before packing, an independent safetensors read checks bounded source samples and their logical/shard ownership. The existing HIPC packer transforms persistent weights once; scale semantics and the internal factor16 remain unchanged. Publication includes loading receipts and packed samples for every slot. IPC consumers reject checkpoint loading, skip packing, and validate mapped samples against that same publication.

## Routing identity and execution

The frozen selector is `splitmix64-request256-position-layer-expert-v1`, seed `0x4d33315043503831`, unsigned64-bit wraparound, little-endian request-key words and ascending physical replica IDs. Each logical Top4 route selects exactly one physical replica before DeepEP dispatch. Received IDs are local0..16 or padding -1. Weights, routed scaling and shared expert addition retain their existing owners.

The worker binds each new request to the SHA256 of its complete initial prompt as uint32 little-endian token IDs. If that SHA is in the frozen dataset manifest, it verifies and uses the manifest's group/slot/content request key. Otherwise the same key schema uses `group_id="runtime"`, slot0 and the prompt SHA. HTTP IDs and the wheel's random internal suffix do not enter selection. Decode/resumption retain the original prompt identity. PCP absolute positions come from the actual partitioned batch; bounded diagnostics independently compare them with CPU segment positions.

The implementation permits one model forward in flight per worker. A frame lock covers PCP identity preparation through the last MoE layer; layer order and caller thread are checked. Failed forwards require an engine restart. DBO, speculation, non-eager execution and online expert-map updates are rejected. Static routing can compose with the opt-in shared auxiliary stream: the DeepEP call lock covers shared launch, selection, dispatch, the shared join and static-layer completion. Selection/frame state stays on the caller thread; the auxiliary TP1 shared MLP cannot issue collectives. The shared launcher retains kernel/type/stream checks, input/output record_stream ownership and exception joins. Multiple consumers have separate frames, tables, KV and scratch.

Both HIPC pack paths accept explicit16 or17-slot capacities. The tiled upper-bound search masks out-of-range trials for17 slots, including repeated boundaries from empty experts. Counts must match weight capacity; invalid IDs, count disagreement and inverse restoration remain checked.

## Configuration and publication

Use the task configuration builder's `--static-placement <placement.json>` argument. It embeds the full57-layer table, selector, map SHA, capture SHA, dataset manifest SHA, precision decision SHA and `minimax31-static136-hipc-loader-v1` ABI in `quantization_config.hcu_minimax_static_placement`. The weight identity binds the physical layout and implementation/ABI dependencies. A different map or physical partition requires a new producer, socket and generation.

Required producer/consumer opt-ins are `HCU_MINIMAX_STATIC_EPLB=1`, `HCU_MINIMAX_DEEPEP_FIXEDCOUNTS=1`, `HCU_MINIMAX_HIPC_PACK_TILED=1` with serial shared execution by default. Set `HCU_MINIMAX_SHARED_OVERLAP=1` to use the guarded auxiliary stream for local prefill batches of at least2048 rows. Supply `HCU_MINIMAX_STATIC_DATASET_MANIFEST` with the frozen manifest. Producers additionally require a fresh32-hex `HCU_MINIMAX_STATIC_GENERATION` and a new `HCU_MINIMAX_STATIC_RECEIPTS` directory. Consumers require the matching `HCU_MINIMAX_STATIC_READY_MANIFEST` and a new `HCU_MINIMAX_STATIC_CONSUMER_RECEIPTS` directory.

The supplemental static partition and expected publication are checked before export. The unchanged native daemon method independently checks the full native `WeightCacheKey`. The static envelope then binds generation, rank/UUID, full-map SHA, weight contract, per-tensor names/shapes/strides/dtypes, aliases, slot-loading proofs and the actual returned IPC reduction metadata/handles in the same response. The consumer checks that envelope before rebuilding storage, then checks mapped metadata and samples. READY files and hashes are identity evidence, not signatures or hardware write protection. No in-place reload/repack or daemon release command is exposed by this path.

`HCU_MINIMAX_STATIC_ROUTE_CONTROL` optionally enables bounded target-append diagnostics. Actual selected physical counts, DeepEP receive counts and sampled selector inputs/outputs are recorded separately. Keep this variable absent in formal performance runs.

## Validation status

The task artifacts are under `task_dir/support_minmax_31_pcp8_ep8_perf/static_eplb_multistream_fp8index/`.

- A_fit48 groups were captured on the selected FP8 index path, with8-rank nonoverlapping coverage and262144 logical routes per group/layer. Exact executed calibration sources and raw-record provenance are preserved.
- Frozen map SHA: `c23ec67feec9dd96d0a766e76f08af2ae396217a3d8500abfbb0d5412d83f3a5`.
- GPU selector matches the independent CPU planner exactly, including9 replicas and padding.
- Real17-slot HIPC matches the independent INT64 small-case oracle; the33280-padded-row two-chunk test is bit-exact against16-slot storage with the same4 expert weights.
- Partial4-layer/8-rank loading, same-generation zero-copy mapping, no consumer repack,1/2048-token requests, and24 live wrong-map/generation/native-key rejections passed.
- On nmz22, complete60-layer/8-rank loading and publication passed:46512 full-copy checks and46512 independent checkpoint sample checks per generation,17 owners per layer/rank,pack once; all8 consumers mapped all57 MoE layers with no consumer pack.24 live wrong-map/generation/native-key cases were rejected before handles.
- Full60 fixed4 and concurrent8 functional assertions passed. Engine-only restart retained identical generation/UUID/map/publication/sample proofs on all8 ranks.
- A_dev16 passed actual selected-vs-received counts and exact PCP coverage for all912 group-layers,262144 routes each;233472 bounded selector routes matched independent CPU replay.
- Five developer trials per mode did not prove stable additional static speedup: E median3.520067s vs linear-F3.594079s;74.011ms gain was below247.778ms maximum observed range. All valid points were retained. Static remains an experimental opt-in and is not the selected performance layout.
- The original A_fit placement was calibrated on FP8 index; BF16-index static calibration/performance is not established by these runs.
- The frozen experimental F candidate has a measured original-workload client speed benefit, but FP8/BF16 heldout first-token agreement is42/64 and independent-history agreement is3/4. The independent numeric oracles and fixed functional assertions passed; these unlabeled first-token differences neither establish quality loss nor precision equivalence. Precision acceptance remains unresolved; B0 (linear,serial shared,BF16 index) remains the recommended configuration. No parameters are retuned using B.
- See the task's final decision/report for final trace and performance scope. No controlled SGLang superiority is established by these checks.


## RSC alignment validation (2026-11-15, nmz20)

The reference SGLang physical table was imported without refitting, with a fresh configuration, producer generation and strict IPC publication. Its map SHA is `07b326c398430e27c1b4a85f1104d36bc610c70c506f3a2f07ed3712402ccef9`. The selector remains this implementation's splitmix policy; the reference uses locality_fair. No previous A_fit map or producer identity was relabeled.

The registered MiniMax runner scopes the initial no-attention memory probe to a conservative PCP-local row budget: `min(N,roundup256(ceil(N/8)+2*min(max_requests,N)))`. Each DualChunkSwap prefill contributes at most two ceil-sized chunks; single-token decode ownership fits the request slack. DCP1, decode ownership and no speculation are required. Other dummy runs and the real scheduler token budget are unchanged. At global65536/max_requests80 this profiles8448 local rows instead of65536 per rank. On nmz20/mem0.86 the previous probe left8.16GiB KV and could not serve147457 tokens; the corrected probe exposes35.92GiB and557343 logical KV tokens. No upstream wheel files are edited.

- The new lifecycle/profile checks and affected existing CPU regressions passed (27 tests).
- All8 full60 consumers map the same reference publication, validate weight samples and perform no permanent repack. All57 shared MLPs on all8 ranks match serial versus auxiliary execution bit-exactly at2048/4096/6144/8192 rows (1824 checks).
- The complete53-case RSC matrix uses the reference frozen tokens and rates, with1347 successful147456-input/1-output requests. Each reports exactly131072 device-cached tokens. Native mean/P95/throughput formulas and per-case input SHAs reconcile with raw records. Performance runs have no active profiler.
- A separate BS4 trace contains forwards1+1+2,65536 new tokens and262144 actual received routes per MoE layer, with17 local receive slots. Whole-window layer balance averages95.83%; the step/layer average is88.32%. Balance here is100*mean(rank route count)/max(rank route count), not a GEMM timing metric.
- Every traced MoE uses the shared auxiliary stream. Per-rank shared kernels overlap other kernels for about108-111ms in total, including about13-18ms of dispatch, and0ms of routed GEMM. Interval overlap is execution evidence, not an isolated end-to-end speedup claim.
- Within the measured matrix and mean-TTFT<2s rule, peak input throughput including cache is179278tok/s versus the reference238276tok/s. The baseline is historical nmz18, not a same-node A/B. KV capacity differs (557343 versus1202176), and BS>=32 does not meet the reference client's full-batch-residency capacity precheck; those completed runs are validated by actual per-request cache hits. Do not claim all configuration constraints are identical.

Evidence lives under `task_dir/support_minmax_31_pcp8_ep8_perf/sglang_prefill_alignment_20261115/`. FP8 precision acceptance remains unresolved; this benchmark does not replace the original quality decision or establish a production recommendation.
