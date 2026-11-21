# v0.33 Layer Split 独立缓存池

状态：候选实现；GPU 容量、模型精度和 PD 端到端验证待预约显卡。

## 改动目的

旧实现所有组共用一个 BlockPool，物理块按最大组预留。这是合法布局，
但不同组的页面大小不同时存在填充开销。FP8 实测 schema 中，三个压缩器
状态页面每请求只需要一个块，却把公共块推至 96 KiB。

新实现在 Plugin 内使用官方 builder 和 custom scheduler 接口；不修改官方
Core 文件，也不替换算子。Adapter 的现有公开接口可以直接承载这次改动。

## 分配与调度

- 只合并完整 spec 和角色相同的 SWA 组；真实 FP8 schema 的七个 SWA 组
  合并为一个组，主 KV 和压缩器状态各为一组。不会仅凭字节数混合不同语义。
- 三个组有各自的 BlockPool、页面大小、块数和 null 页，仍放在官方 allocator
  的同一个 backing allocation 内。不同组可以使用同一数字块 ID，物理地址隔离。
- 主 KV 保留源层 owner；SWA 40 层按 rank 循环分配；每组额外保留一个可复用
  接收槽。压缩器计算和状态仍复制，只改变它的分配池，不假称已做状态分层。
- 每组容量按一个最大长度请求的原生需求计算。可完整容纳的请求数取整数；
  Core 的 num_blocks 表示等价准入单位，具体 tensor 页数由 num_blocks_of 返回。
  因而日志中的容量公式与实际各池准入一致，不能把 num_blocks 直接当任意组的页数。
- 每次分配前逐池检查需求；某池不足时整个分配拒绝，避免分配一半后才失败。
  保留原生滑窗回收、前缀命中、CircularBuffer 生命周期、同步/异步调度。
- 延迟释放通过块对象的 pool 路由。Core 错误失效接口只传裸 ID，故对各池同号
  块保守失效（不释放活跃块）；可能额外减少前缀命中，不造成跨池释放。
- 使用率按各池实际页面字节加权。通信位图、ID 列表和 payload 使用显式尾部
  区间，全部计入预算，不再用某个 tensor 的页数推算公共通信起点。
- 各池内部存储 dtype 一致，块不会在 FP8 KV 与 FP32 压缩器状态之间流转。
  主 KV 与 indexer 的量化标记不同，但每层持久区的偏移和格式固定，scratch
  在每次读取前整页接收。因此按实际复用关系判断 zeroing 需求；不会
  调用假定全局块 ID 的 KVBlockZeroer。池内混合精度及 Mamba 状态明确拒绝。
- 继续限定现有 eager 场景。官方 eager 配置关闭 JIT kernel warmup；若启用
  fault tolerance 等使 JIT warmup 保持开启，规划明确拒绝，不能把原生公共
  块 ID 预热送进独立池。测试脚本的 HTTP 预热请求仍通过真实调度器正常执行。

## SWA Replay 与 P/D

保留 SWA bounded_replay、窗口、block_size、tokens_per_state 等原生语义。
CPU 原生 replay 流程和调度测试不替代 GPU replay 数值验证。

Mooncake owner 协议升级为 v3，两端均须使用新版 Plugin。按层名核对页面语义，
允许 P 的三组与 D 的九组编号不同。发送时将每层各自的块表投影到临时虚拟组，
复用官方传输描述符生成；不修改共享 worker 状态。只发送 owner 的实际页面，
不发送 scratch、padding 和通信区。旧 v2 握手被拒绝。

## 验证与后续实机门禁

CPU 使用隐藏 GPU 的独立进程和安装的 vLLM wheel；测试代码中的 HIP 算子替身
只验证接线与元数据，不证明硬件算子精度。

本轮 CPU pytest：142 项通过；另执行 PCP 源码回放（13项）。结果保存在项目
`logs/independent_kv_pools_20261112/`，同目录 `before/` 保存修改前文件。

测试覆盖真实 schema、八 rank allocator 视图、独立池耗尽、原生分配/回收、
SWA 循环回收、前缀命中、同步/异步调度入口、跨 rank 容量收敛、显式块数覆盖、
pickle/deepcopy、Gloo 广播跨池哨兵，以及 P 三组/D 九组的原生字节传输。

容量脚本 `tests/vllm033/report_dsa_cache_pools.py` 使用相同的 10 GiB 预算、
max_model_len=16384、max_in_flight_tokens=16384，输出 CPU 规划结果。
用户提出的 4 倍目标必须以同口径 GPU 实测确认，不能用 CPU 规划冒充达成。

当前 CPU 规划（10 GiB）：关闭组 390,989 tokens，上一版主 KV+SWA 公共池
463,852 tokens，新独立池 2,310,144 tokens，为关闭组约 5.91 倍。
新池实际 backing 10,640,785,408 bytes（含 4 KiB 对齐），没有超出预算；SWA/主 KV/状态池的
页面大小分别为 114,048 / 65,088 / 98,304 bytes，页面数量分别为
72,334 / 36,097 / 142（各含一个 null 页）。不能把这些 CPU 数字写成实机结果。

预约显卡后先跑单 P 开启组，保持旧容量测试的输入128/4096/12288、输出16、
预热2轮、PCP8、mem0.9、Replay OFF；记录每个 Rank 的实际可用字节数和 Token
容量，检查 owner/pool 日志、输出 token/logprob、错误及清理释放。再分别验证
Replay ON 与 PD/HumanEval；不得用启动成功代替精度验收。


## 2026-11-13 PD validation checkpoint

This is a local development checkpoint, not a claim of accuracy equivalence.
The installed Core wheel is `0.33.0+das.dtk2610.gfx936gfx938.g5afd482`;
Core source was not modified. Adapter and Plugin source overrides were used.
The Plugin requires the companion Adapter KV-cache extension seam commit.

Single-P tests on nmz30 used FP8 PCP8, memory utilization 0.9,
inputs 128/4096/12288, output 16, and two warmup rounds:

- Replay OFF: 10,125,312 KV tokens, 43.64 GiB available KV memory.
- Replay ON: 10,059,776 KV tokens, 43.32 GiB available KV memory.
- The 48 formal output token IDs matched; maximum logprob delta was 0.00312206.
- These short requests are not a full-capacity stress test.

PD validation used P on nmz18 with Layer Split and Decoder SWA Bounded Replay
ON, and D on nmz16 with both OFF. Prefix caching and dSpark were OFF.
P reported 10,059,776 KV tokens and 43.32 GiB available KV memory;
eight-rank independent pools and SWA ownership were verified from startup logs.
All nine fixed-input requests (two warmups and one formal round) matched the
historical OFF-group token IDs. Maximum logprob delta was 0.00502075;
no numeric logprob tolerance was enforced by this comparison.

HumanEval used 164 tasks, temperature 0, max output 2048, concurrency 16.
The ON group scored **157/164 (95.73%)** versus historical OFF/OFF **159/164
(96.95%)**. Regressions: 67, 130, 163; improvement: 83. All requests succeeded,
all stop reasons were `stop`, and the longest output was 789 tokens.
Both switches and the historical implementation differ, so attribution remains
unresolved; this result must not be presented as no accuracy regression.

Mooncake verification observed 164 D requests, 26,745 prompt tokens and exactly
26,581 external KV tokens (sum of N-1). Transfer bytes were 629,213,056;
1,256 transfer calls are batches, not request counts. Transfer/receive/expiry
error increments were zero. Own P/D/router services were cleaned up and both
nodes returned to 2 MiB per GPU with no KFD processes.

Evidence is in the project-level run directory
`accuracy/runs/pd_layer_split_replay_nmz18_16_20261113_manual/`:
`accuracy_audit.json`, `comparison.json`, `result.json`,
`humaneval_on_transport.json`, and `post_test_gpu.json`.
CPU evidence remains 142 pytest cases plus 13 PCP source-replay cases.
Future upstream rebases require CPU regression and GPU PD/accuracy revalidation;
these measurements do not qualify a different Core version or configuration.
