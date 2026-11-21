# DeepSeek V4.1 layer split on vLLM 0.33

The current candidate targets eager standalone PCP with TP/PP/DP/DCP set to
one. Use `--enable-layer-split` with `vllm serve` and the companion Adapter.
The switch writes `"hcu_layer_split":{"enabled":true}` to the engine's
`additional_config`. The default is disabled.

The implementation follows the owner-plus-scratch design from HYGON-AI
vllm-plugin-das PR #370 at `279aeee` and PR #243 at `8c827d9`. DeepSeek V4.1
has four main-KV source layers, so ownership is assigned over the source list
`[2, 8, 14, 20]`. PCP8 ranks 4–7 have no persistent source cache and still
participate in every read collective. Indexer-K ownership follows its KV source.

## Placement and integration

Every rank reports every native source cache spec. The platform's public
`get_kv_cache_config_builder_cls` hook selects `DsaLayerSplitCacheBuilder`.
The builder preserves all native groups on every rank. Splitting SWA/ring
layers into independent groups increases the scheduler's block demand per
request and can erase the placement savings. Only physical offsets change.
Cache specs retain their native lifetime and semantics. Block-stride alignment
uses the native `get_block_stride_alignment()` accessor, including `MultipleOf`
constraints expressed in kernel tokens.

The engine plans a common worst-rank byte budget. The Adapter's existing
Worker calls a registered pre-allocation transform, which materializes the
correct rank-local offsets before entering the official worker allocator.
Group IDs, block counts and Core-owned configuration metadata are preserved.
The integration uses a generic callback registry in
`hcu_adapter.bridges.vllm_kv_cache`. The feature implementation lives in Plugin.

Within each group, remote caches share one scratch slot after that group's
owned/persistent pages. Groups overlay the pool by block ID using a common
block stride: the native pool assigns a live block ID to only one group.
This layout is safe ONLY with length-masked, live-ID transfer; prefix broadcasts
and unmasked stale table tails would overwrite other groups. A trailing wire
region reserves 256 bytes per block ID for the union bitmap, int64 ID staging
and dense transfer chunks. The builder includes all three in the same backing
allocation and the one-block sizing probe. It has no lazy page-sized receive
allocation outside the planner's budget.

Only source owners commit main-KV and indexer-K writes. The two write guards
are pinned to reviewed native function hashes. Native PCP gathers precede the
guards and retain their original order.

## Read collectives

Indexer-K broadcasts occur after the native indexer preparation, including
the short-context shortcut. Main-KV broadcasts occur before the attention
zero-query branch. This gives empty local-query ranks the same collective
sequence as their peers.

A MAX collective unions a bitmap of live block IDs from all PCP ranks, including
ranks with no local queries. Only these physical IDs are packed/unpacked; their
coordinates do not change. A block-table entry is live only within the actual
request count and ceil(uncompressed_seq_len / planner_block_size). Null block
zero is not transferred. Neither prefix holes, stale columns nor padded rows
are copied. The null bitmap slot carries a collective metadata-error flag so
all ranks reject invalid live IDs or missing schema before payload transfer.

The worker registers each source's block size in uncompressed tokens. Cache
tensor dimensions cannot supply it for ratio-2 sources. Indexer root seq_lens
come directly from common request metadata; compressed decode lengths and
PCP-padded prefill chunks are not used. Main-MLA metadata lacks seq_lens, so its
paired indexer's root lengths are applied to the main-MLA's own block table.
Missing or incompatible metadata fails closed: there is no full-pool fallback.

Dense chunks pack into the budgeted wire region and unpack into the receiver's
bound cache view, including packed record bytes. Repeated reads always
participate because another cache may have overwritten the shared receive slot.
Host copies of the root table/lengths and union bitmap are intentional in this
eager candidate; no graph support or device-performance improvement is claimed.

## Validation and limits

The current CPU suite covers widths 2/4/8/16, four native view layouts,
native allocation, persistent/scratch separation, padding, multiple groups,
hybrid SWA/ring grouping and eight-worker Core planning/convergence. A real
two-process Gloo test covers repeated owner broadcasts, gapped block IDs and
owners with no local queries. Platform and Worker entry-point tests exercise
the registered callbacks. Existing PCP source replays remain part of validation.

Hardware validation is tracked in the nmz22 section below. CPU tests establish
placement and integration contracts. Model-level output comparisons and
measured capacity use separate device evidence. Model services wait while the
eight cards are occupied.

Native Mooncake remains guarded out for layer split. The opt-in Plugin PD candidate and its CPU-only evidence are documented in the final section; GPU PD is not validated.
Speculative decoding remains outside this PCP adapter's contract. SWA replay
composition requires its own device validation.

## CPU commands

Run inside the existing Python 3.12 container with source paths selected.

```bash
cd /workspace/deepseekv4.1-flash/vllm033_adapter
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ VLLM_TARGET_DEVICE=cpu
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export PYTHONSAFEPATH=1
export PYTHONPATH="$PWD/adapter-das:$PWD/vllm-plugin-das/adapter_extension/src"
/tmp/vllm033_validation_20261110/.venv/bin/python -m pytest --noconftest -q \
  vllm-plugin-das/tests/vllm033/test_dsa_cache_layout.py \
  adapter-das/tests/adapter/test_vllm_kv_cache.py
# Core imports resolve to the installed vLLM wheel, not the Core source tree.
```

The earlier implementation is preserved in
`project_info/backups/layer_split_before_rewrite_20261110.tgz` at the project
root. It omitted non-owner specs and failed at native cache allocation. The
current placement replaces that approach and the PCP-width guard it required.

## Earlier nmz22 device validation on 2026-11-11 (global receive region)

The FP8 baseline completed two warmups and one measured batch containing
128, 4096 and 12288 input tokens, with 16 output tokens each. Its measured
batch duration was 9.052949517 seconds and its reported capacity was
1,706,205 tokens. The baseline directory is
`accuracy/runs/fp8_layer_split_nmz22_20261110_230735_c6srie03`.

The first enabled attempt exposed an integer-only alignment calculation at
native `MultipleOf` constraints. Six new alignment cases reproduced the
failure. The fix uses the native byte-alignment accessor. The next attempt
generated all requested tokens with identical greedy IDs to the baseline,
but its collector rejected missing ownership logs. Plugin INFO records had
inherited the Python root WARNING level. Ownership logging now uses the
`vllm.hcu_dsv41_layer_split` logger and its own regression covers this route.

That enabled attempt reported only 527,180 tokens of capacity. A CPU probe
using the native group-aware capacity calculation reproduced a reduction
when SWA/ring groups were split. Keeping native groups reduced scheduler block
demand, but a following repeated device comparison exposed cross-group data
corruption. The previous per-group receive slots overlapped other groups'
persistent data when prefix broadcasts copied holes in the source block table.
Three new CPU cases reproduced overwritten SWA data before the fix. The
rank-wide receive region then followed the largest persistent group on that rank.
The complete layout suite passed all 42 cases, including cross-group holes,
unchanged groups, native byte views and scheduler capacity accounting.

The safe receive region is fully included in the pool budget. Hybrid-cache
capacity can be lower than native capacity. Source-only savings remain covered
separately. The earlier group-only capacity improvement excluded the required
cross-group isolation and establishes no safe device memory benefit.

Device revalidation completed in
`accuracy/runs/fp8_layer_split_nmz22_20261111_011843_jb7apdf5`.
All three cases produced 16 tokens in both warmups and the measured round.
Every greedy token ID matched the original disabled baseline. Token logprobs
and the returned top-5 values were finite. The maximum selected-token logprob
delta across those rounds was 0.011328521. All eight rank ownership records
were present. The driver exited with code zero and the service released its
GPU allocations. Machine occupancy had returned to the original 290–322 MiB
baseline with zero utilization before launch. Two pre-existing import/build
contexts were identified by their exact commands and retained.

The final measured batch took 12.306132602 seconds. Rank0 reported 43.63 GiB
available for KV, with scheduler capacity of 1,282,167 tokens. The disabled
baseline reported 1,706,205 tokens and took 9.052949517 seconds. This safe
reference layout reduces capacity by 24.85% in this hybrid configuration.
Sparse transfer and receive-space optimization remain necessary for a memory
benefit. Full model accuracy, broader inputs and feature combinations retain
their separate validation requirements.

The comparison, per-round deltas and placement evidence are recorded in the
enabled run's `comparison_to_off_baseline.json`. Additional checks passed
13 PCP source replays under the tested stock FP4-indexer policy, plus four
Adapter KV-registry cases. An earlier replay with the default LightOp policy
failed the stock-only publication expectation. Its log is retained alongside
the passing explicitly configured run.

Attempt logs, backups and CPU regressions are retained under
`logs/nmz22_fp8_layer_split_on_20261111`. The one-minute idle watcher uses
`accuracy/watch_layer_split.py` with `LAYER_SPLIT_TEST_MODE=on` and the FP8
entry point. It requires two consecutive idle samples and runs once.

## Capacity candidate on 2026-11-11 (CPU only; awaiting GPU reservation)

The user authorized code changes and explicitly deferred GPU tests until a
reservation. No model server or idle watcher was started for this revision.
Changes are confined to Plugin layout/transport, their tests and this document;
the existing Adapter registry and installed Core wheel are unchanged.

PR #370's owned-plus-scratch and selected-block transfer informed this revision.
Its permissive stale-ID handling cannot be copied into v0.33's shared hybrid
block pool. The new path masks by real request lengths before union/transfer.
Native SWA/ring groups and scheduler accounting remain unchanged.

Three negative-control cases failed against the prior implementation: two
same-budget hybrid-capacity cases and one sparse-transfer case. After the
change, 52 layout/transport/integration tests passed. They cover all previous
alignment and ownership cases plus root-length selection, compressed block
units, stale/padded/null entries, invalid metadata, tiny transport buffers and
two real Gloo processes. Gloo covers changing active sets, different local
sets, an owner with no queries and preservation of other groups' data. The
Gloo workers assert that CUDA was not initialized.

The same-budget CPU probe used 10 GiB, 22 unchanged native groups, four
source-KV/indexer pairs, 40 SWA layers and three circular-state layers. This
is a synthetic planner fixture, not specs captured from a new model service:

| Planner | Bytes per pool block | Pool blocks | Capacity in tokens |
| --- | ---: | ---: | ---: |
| Native, split off | 105408 | 101865 | 580102 |
| Live-ID split candidate | 78976 | 135957 | 774250 |

That fixture gains 33.47% token capacity. Its transport reserve is included
in the 78976 bytes. It does not predict the exact FP8 device gain or throughput.

The Adapter registry suite passed 4/4. The PCP standalone replay initially
passed 12/13 because its direct test calls bypassed pytest's new block-size
registration fixture. After explicitly providing that worker setup in the
replay, it passed 13/13 with `cuda_initialized=false`; the original failure
log is retained. The replay report now hashes the layout/transport modules
as well as the existing PCP modules. No production guard was relaxed.

Evidence and original file backups are under the project directory
`logs/layer_split_capacity_20261111/`. CPU capacity results are synthetic native
planner calculations, not a new measured model result. The earlier GPU output
agreement does not validate this revision. After reservation, rerun identical
FP8 off/on settings and inputs, check both warmups and the measured outputs,
Token/logprob agreement, all rank ownership and `GPU KV cache size`. Success
requires more token capacity at the same budget and correct outputs; record
runtime/communication overhead separately. Do not silently enable PD, DSpark,
SWA replay or change the memory fraction to manufacture a capacity gain.


## 2026-11-11：owner-aware Mooncake PD 候选

新增 `hcu_vllm_das.mooncake_layer_split.HcuLayerSplitMooncakeConnector`，通过 Core
公开的 `kv_connector_module_path` 入口加载，不修改 Core wheel 或 Adapter 源码。
它继承已安装 0.33 Mooncake 的收发协议、统计、错误和完成处理，使用独立的 PCP
bootstrap 描述真实 PCP rank；没有将 PCP 伪装为 TP 或 PP。

- 源层 main KV 与 indexer-K 由源层 owner 发送。未开启分层时由 rank0 发送全部缓存。
- 复制保存的 SWA/压缩状态只由 rank0 发送。rank4–7 仍完成请求握手与 ACK，参与 P
  端 worker 释放计数；D 在所有 PCP rank 完成后才报告接收完成。
- 注册共享 backing 可以包含整个存储，RDMA **传输描述符**仅包含实际页视图；
  不传非 owner 的 scratch、wire 区、整行 padding 或其他组的空间。
- 保留 native group 的 block ID 和地址步长，P/D stride/offset 可以不同。
  bootstrap 验证每个真实传输层恰好一个发送者，并比较 P/D 的组、页形状、dtype、
  block size、压缩比和 SWA语义。不匹配先拒绝，不能默默漏层。
- V41 compressor 在 PCP 下仅保留未闭合的 ratio-2 行。D 会重算最后一个输入 Token，
  因此扩展 scheduler 将 Core 现有的 recurrent-state 截断流程用于 CircularBufferSpec：
  P 计算 N-1 Token，D 获取对应前缀后计算最后一 Token。与用户请求输出 Token 数无关。
  保留 Core 对重复调用、prompt logprobs、失败和释放的处理。

当前范围：V41 MLA、TP/PP/DCP=1、P DP1 eager/PCP，D PCP1 可 DP8；两端不启用草稿模型。
D DSpark 在 40–42 层有独立上下文 KV，P 当前不产生它们；任一端启用 speculation 都拒绝，
不能通过丢弃这些层或填零来宣称精度兼容。默认 N=0 和 DSpark5 的 compressor ring
均为8槽，但这不解决草稿层上下文缺失。

CPU 回归位于 `tests/vllm033/test_mooncake_layer_split.py`，覆盖原生规划器/allocator
产生的 packed views、跨组稀疏 block IDs、完整字节复制与未触及字节保护、owner
重复/缺失、真实本机 ZMQ 编码/收发/ACK、模拟复制引擎失败以及延迟的第八个 ACK。
还检查 public factory/HMA入口、单 bootstrap 选举和 P/D 最后一个 Token 的状态交接。
CPU 模拟复制替代 RDMA 引擎，测试不初始化 GPU。这些证据不证明 GPU 起服、RDMA
注册/传输、PIECEWISE、吞吐或整模型精度已通过。

PD入口与请求客户端见工程 `accuracy/PD_LAYER_SPLIT.md`；证据在工程
`logs/pd_owner_20261111/`。部署 GPU 前需确认节点和显卡预约。真实验收须包含：
P Layer Split off/on 相同固定输入输出；日志中每 rank owner/注册页；实际 RDMA bytes
及失败计数；生成 Token/logprobs 对照；重新记录可用 KV 容量。


## 2026-11-11 nmz18 FP8 standalone GPU validation

The initial device run exposed a native zero-query PCP metadata path omitted
from the synthetic metadata tests. Its indexer metadata has zero local decode
and prefill tokens, with block_table=None. The source now contributes no local
IDs in exactly that case, then still performs the collective and receives pages
requested by other ranks. The added regression uses the actual Core metadata
class, including the main-MLA placeholder with a nominal request row.

The real FP8 grouping also has different overhead from the earlier synthetic
fixture. Seven replicated SWA groups dominate the common stride. A 4096-byte
per-block communication reserve erased the source-cache savings. The reserve
is now 256 bytes per physical block. Bitmap and aligned int64 IDs need at most
9*N+7 bytes, leaving positive bounded payload capacity even at N=1. Existing
row and column chunking handles this capacity. Native group identity and all
cache lifetimes remain intact. The older CPU capacity table above records the
previous 4096-byte reserve and its synthetic fixture.

The real native group schemas captured from the failed device attempt are in
`tests/vllm033/fixtures/dsv41_fp8_native_cache_groups.json`. The new test checks
same-budget capacity including transport and two concurrent serving batches.
CPU regressions passed 55 layout cases, 24 owner-transfer cases, and 13 source
PCP replays. The following full off/on device run passed after both fixes.

| Mode | KV capacity tokens | Formal request duration seconds |
| --- | --- | --- |
| off | 1706052 | 8.205278248991817 |
| on | 1745215 | 10.577076940098777 |

Capacity increased 2.2955%. Each mode used the same three prompts of 128, 4096
and 12288 tokens, generating 16 tokens each, after two warmup rounds. All 48
formal output token IDs match. Selected-token logprobs are finite, with maximum
matched-token delta 0.00209554610773921. Durations include generation and are
one formal sample per mode. The device comparison covers these fixed inputs.

The run is `accuracy/runs/fp8_layer_split_nmz18_20261111_184512_8a38mzt_`.
Console, initial failure logs, backups, CPU results and final byte/source checks
are retained in `logs/p_only_nmz18_20261111`. The final record is
`final_verification.json`. Both model services shut down after collection and
GPU memory returned to approximately 2 MiB per device. PD GPU validation remains
a separate pending task.

## 2026-11-11 PD bootstrap router compatibility

The first real P18/D16 run reached healthy services, but the installed Rust
router rejected the custom /query response with
`Invalid dp_rank 'protocol': invalid digit found in string`.
No inference requests were sent in that failed deployment.

The Plugin now keeps /query compatible with Core's DP-to-EngineEntry schema,
advertising the actual producer DP=TP=PP=1 topology and its rank-zero leader.
PCP ranks are not advertised as TP ranks. The independent
/hcu/pcp-owners/v1/query endpoint returns all owner manifests; the Plugin
consumer explicitly uses this endpoint and still requires all eight ACKs.
Both endpoints return 503 until all owners are registered.

Real HTTP CPU tests compare /query against Core's actual bootstrap server
response and exercise the real consumer discovery method for split off/on.
81 Mooncake/layout regressions passed with GPUs hidden. Runtime validation
is recorded separately; CPU success alone is not a PD accuracy result.
Evidence and pre-change backups: logs/pd_bootstrap_fix_20261111.

The existing Rust router binary also needs libssl.so.1.1 and libcrypto.so.1.1.
Run-local symlinks to the existing /opt/hyhal/hbm/Miniconda3/lib copies,
used only in the router process LD_LIBRARY_PATH, resolve this environment
issue without replacing system libraries or model packages.


### 2026-11-11 truncated-prefill EOS handoff

The first live HumanEval run exposed a second PD seam: all 16 initial chat
requests finished on P with `stop`, and D waited for remote KV indefinitely.
P's transferred-request counter remained at the nine earlier fixed requests.
The fixed-input capture had `ignore_eos=true`, so it did not cover this case.

`OwnerScheduler` truncates P to N-1 input tokens to transfer the recurrent
compressor state before D recomputes the last input token. The token sampled
from this shortened prompt is internal and discarded, not the user's first
completion token. Core's producer `request_finished` only retained pages for
`FINISHED_LENGTH_CAPPED`; EOS/stop-token completion freed the handoff instead.

The Plugin now also retains/send-enqueues pages for `FINISHED_STOPPED` only
when this is a compressor-state, explicitly truncated, remote-decode handoff
with a transfer ID. It preserves the request status, sampling parameters,
native group clipping and delayed-free lifecycle. Abort/error/untruncated and
consumer paths remain delegated to Core. No EOS suppression, zero-filled KV,
Core source edit, or precision threshold change was used.

Regression uses real Core `Request`, `SamplingParams` and `check_stop`:
EOS and explicit stop-token cases fail before the fix; after it, they and
normal length completion retain the clipped handoff pages. Abort/error and
untruncated stopped requests still take native cleanup. Combined Mooncake and
layout CPU suite: **87 passed**, GPU uninitialized.
Evidence: `logs/pd_truncated_eos_fix_20261111/{before,after}.log`.

The interrupted HumanEval attempt is invalid and has no accuracy score.
The next live run is
`accuracy/runs/pd_layer_split_nmz18_16_20261111_eos_fix`.
D is retained from the preceding run after all pending requests drained; this
fix changes only the P-side stopped handoff branch, not the consumer.
Live off/on HumanEval results are pending.

#### Live off result and requested shutdown

Run `pd_layer_split_nmz18_16_20261111_eos_fix` completed the fixed-input off
capture and all 164 HumanEval samples: **159/164, Pass@1 96.9512%**, zero HTTP
errors and zero length-truncated outputs. Failed tasks: 32, 83, 103, 132, 145.

Transport audit: 629,213,056 bytes in 157 batch-write calls; all 164 D requests
completed. D prompt accounting is 26,745 tokens, with 26,581 external KV tokens,
exactly sum(N-1); no transfer/receive/expiry error increments. The initial
audit incorrectly compared batch-write count against request count; its failed
artifact is preserved. Core batches ready requests into one write, so the
corrected audit uses completed requests and exact external-token accounting.

The user requested releasing GPUs after this result. Layer Split **on was not
started**, so no PD on/off precision comparison or on HumanEval claim is made.
The P/D/router services are stopped; both nodes show 8 idle GPUs, 2 MiB each, no KFD processes. Final release evidence is recorded
in the run directory. The D log is shared with the preceding bootstrap-fix run
and includes historical cancelled-request failures; new-run deltas are separate.
