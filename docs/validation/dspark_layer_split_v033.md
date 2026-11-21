# PCP Layer Split with DSpark

The candidate supports eager DeepSeek-V4.1 TP1/DP1/PCP8/EP8, fixed
DSpark with five speculative tokens, DeepEP high throughput and the existing
FP4 Indexer. SWA bounded replay additionally checks the draft's attention
window against the target's retained window.

The Adapter owns the speculative compressor used by the target CR2 cache.
Its metadata builder gathers a native ring capacity of tail rows per PCP chunk and
keeps the newest row for every ring slot. Rejected verification suffixes can
therefore resume from any accepted prefix. Native compressor math and ring
storage execute with that metadata.

The Plugin retains the drafter's native global query batch and PCP1 attention.
Its MoE layers receive a private construction config describing the target's
actual PCP/EP expert partition. The draft uses native expert placement. The
calibrated static EPLB map applies to the target. Draft SWA cache specs retain their native
`is_draft` annotation and have separate replicated pools. Target main KV and
SWA retain Layer Split placement. Every pool participates in native admission,
sliding-window retention and request freeing.

The fixed DSpark policy is validated before model allocation. The compressor
consumer and draft constructor sources are pinned to the installed v0.33 Core
contract. Relevant CPU tests cover ring wraparound, every rejected suffix,
cross-rank predecessor rows, physical expert configuration and draft cache
isolation.

The passing W4A8 Layer Split baseline uses twelve GiB of KV memory per GPU,
FP4 Indexer and SWA bounded replay, with DSpark disabled. Its results are in
`acceptance_runs/nmz18_w4a8_wait_keep_20261117_retry3`. The DSpark candidate is
prepared in an independent source copy under `dspark_candidate`.

## NMZ18 GPU validation on 2026-11-18

The W4A8 candidate completed the paired acceptance run `dspark_pcp_06` with
FP4 Indexer, SWA bounded replay and fixed DSpark(5). The configuration used
eager TP1/DP1/PCP8/EP8, DeepEP HT, batch tokens 32768, model length 65536,
128 request slots and block size 64.

Both arms reserved eight GiB of KV memory per rank. Actual scheduler capacity
increased from 272866 to 1310720 tokens, a ratio of 4.80353. Inputs of 128,
4096, 12288 and 32768 tokens each completed two warmups and one recorded
generation of sixteen tokens. OFF and ON token IDs and text agreed in all
four cases. The maximum reported selected-token logprob delta was 0.01590046.

The ON service completed HumanEval/0 through HumanEval/19 with 20/20 passes,
zero HTTP errors and zero truncated outputs, using temperature zero,
concurrency sixteen and a 2048-token output limit. Final native metrics recorded
514 draft steps, 2570 drafted tokens and 2332 accepted tokens. Rejection was
exercised. Its four fixed-input outputs also agreed with the earlier DSpark-off
baseline, with a maximum selected-token logprob delta of 0.00354208.

The Adapter bounds large fallback mHC Post calls to 1024 independent rows per
call, retaining the injected native Torch arithmetic. Device comparisons were
bit-exact for BF16, FP16 and FP32 at 1025 rows and BF16 at 32768 rows. Each
case additionally passed three changed-input graph replays. Twenty-one CPU
tests cover chunk boundaries, strided inputs, dtype and broadcast behavior.

The final profile uses
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:256`.
Twelve GiB KV runs exhausted memory during draft context insertion. An
eight-GiB run with the default split policy exhausted memory on its second
32K input while 2.44 GiB remained in fragmented reserved allocations.
An earlier expandable-segments attempt produced a VMFault. Its kernel fault
log remains in `dspark_pcp_02`, and that fault's underlying cause remains open.

The controller port probe uses `SO_REUSEADDR` to allow sequential service
restarts after TCP TIME_WAIT. Its CPU regressions also verify refusal of a live
listener. The final sixty-test DSpark/controller regression passed.

The OFF service's process-group SIGTERM generated SystemExit error traces
after all twelve successful requests. Its API exited zero and all eight cards
were released before ON admission. The retained ON service remained healthy
with zero runtime error traces. This shutdown behavior is recorded separately
in `completion_audit.json`.

The historical `dspark_pcp_06` run contains its exact
source snapshot, launch parameters, raw responses, metrics and completion audit.
The ON API listened on loopback port 6546 during the run and has since exited.
The source bases for this historical GPU run are Adapter
`b933986d64acc3a3ee69f842ec6208a993a34be5` and Plugin
`487e592ad4d1010f2fa081be11c208228a8a8350`, with candidate changes in the
`codex/layer-split-dspark-20261117` working branch.

## Official integration on 2026-11-19

The branch now uses official Adapter `8ba62c9` and Plugin `0cb6fb2`. Both
FP4 and FP8 target selectors compose Layer Split with the official speculative
compressor. The official draft constructor and static expert placement remain
active in their respective models. The ON-only acceptance profile loads the
calibrated map through `--expert-map` and uses automatic KV memory sizing.
Native FlashMLA, AITER mHC, PCP Q RoPE and inverse RoPE are configured for that
run. CPU preflight validated the calibrated W4A8 map (40 layers, 416 physical
experts) with devices hidden. The 372 selected CPU cases passed across the
main run and focused rerun, and all 30 Linux controller tests passed. GPU
qualification of this combination is pending. The historical NMZ18
measurements above apply to the earlier source pair.

## Layer Split CLI and default RoPE GPU validation on 2026-11-20

The nmz30 W4A8 run used the new `--enable-layer-split` parser path with PCP8,
FP4 Indexer, SWA bounded replay, dSpark(5), and static EPLB32. The loader
keeps the Layer Split option on the target and drops it from the private
config used to build the PCP1 draft before Core validates it.

Two 32K warmups and one profiled 32K request completed. The recorded output
token IDs, text and logprobs matched the previous SWA-prefill baseline.
The capacity was 2,818,048 logical tokens with a logged 16.97 GiB available
KV budget. Rank 0 recorded 40 GPU launches for each of the default HCU
query and inverse RoPE paths. The service was stopped and all eight cards
returned to 2.05 MiB with zero KFD clients.

These device results apply to the pre-rebase candidate. The 2026-11-20
run covered single-node inference and operator selection.
