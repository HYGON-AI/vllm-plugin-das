# MiniMax31：独立 MoE 模块与 SGLang 对齐

已实现独立 `AlignedMoE`，完成同节点 EP8 对照。六种目标 shape/路由分布、八个 rank，共 48 组输出与 SGLang **逐位一致**；INT8 输入、scale、接收行数及逐专家 padded counts 全部一致。新模块比旧模块降低时延约 20.4%–29.2%，在本微基准中比 SGLang 参考低约 6.4%–10.5%。

本轮未启动推理服务，未接入已有服务默认路径。运行结束八卡均恢复到约 2 MiB。

## 本轮范围

测试的是 **给定物理 Top4 的 routed-expert＋EP 通信模块**：输入量化 → dispatch → 专家重排 → GEMM1 → SwiGLU/量化 → GEMM2 → 加权恢复 → combine。Gate/TopK、静态副本选择器、输入 norm 和真实 shared MLP 不计入下表时延；固定 Top4 是为了严格控制工作量。模块支持 shared callback 与外层 routed scale，其跨流 join 已在全部八个 rank 检查逐位正确，但这不是完整 shared MLP 性能评测。

输入/权重为受控合成 fixture，使用目标维度 H=6144、I=3072、每卡 17 物理专家、Top4、EP8。未读取完整 checkpoint、未验证完整模型质量，因此这些结果不能直接换算为服务 TTFT 或整模型收益。

## 已实现的优化

- 在 dispatch 前使用与历史 SGLang 配置对应的 native per-token INT8 量化，传输 INT8＋FP32 scale。
- DeepEP `expert_alignment=256`，以相同 aligned counts 驱动原生 `ep_scatter`/`ep_gather`，移除接收后的全局 argsort 与 BF16 专家缓冲物化。
- 复用 INT8 输入、scale、M 索引、inverse index、GEMM1/GEMM2 输出及最终接收输出 scratch；pinned counts H2D 为异步拷贝，并用完成 event 保护 host 缓冲复用。
- 当前接收批次连续执行两次 HIPC GEMM，移除旧路径 32768 行分块循环；覆盖最大 264192 padded routes、65528 received rows 的 rank0 极端负载。
- 与 SGLang 一致，route weight 在 GEMM2 的 BF16 store **之后**由 native gather 应用；这与旧模块的 weight-before-store 数值边界不同。
- DeepEP normal dispatch/combine 显式支持已测试的 24/64-block 配置，默认 64；不修改进程全局 Buffer 默认值。
- 对外部物理 Top4 的范围、有限权重、padding 权重做融合 GPU 检查，保留异步断言；保留设备、shape、dtype、counts 对齐、stream 绑定和非重入约束。
- 只读使用 HIPC 权重，独占并复用 activation scratch；支持显式 `close()` 销毁通信 buffer。

## 相同工作量的性能结果

环境：BW1100，单节点 EP8。每个 rank 4 次 warmup，15 次无 profiler 测量；每次先取八个 rank wall time 的最大值，再对 15 次取中位数。单位 ms。权重一次性 pack、fixture 加载/哈希、数值检查和 profiler 均不计入这些时延。

| 每 rank 输入行数 | 路由分布 | 旧路径24 | 新路径24 | 新路径64 | SGLang64 | 相对旧路径时延降低 | 相对SGLang时延变化 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2048 | 均衡 | 6.045 | 4.837 | 4.280 | 4.574 | 29.2% | -6.4% |
| 2048 | 全部路由到 rank0 | 24.494 | 19.649 | 19.236 | 21.256 | 21.5% | -9.5% |
| 6144 | 均衡 | 14.670 | 12.387 | 10.834 | 11.606 | 26.1% | -6.7% |
| 6144 | 全部路由到 rank0 | 70.139 | 57.129 | 55.823 | 62.275 | 20.4% | -10.4% |
| 8192 | 均衡 | 19.032 | 16.172 | 14.060 | 15.131 | 26.1% | -7.1% |
| 8192 | 全部路由到 rank0 | 93.070 | 75.816 | 74.082 | 82.767 | 20.4% | -10.5% |

2048/6144/8192 个本地输入行分别对应 CP8 下全局 16384/49152/65536 个新 token。偏斜场景将全部有效 Top4 送往 rank0，其他 rank 的专家接收为空；它用于检验长尾、空接收与大地址范围，不代表真实 RSC 的路由分布。

这些测量按 variant 顺序串行运行，未做交错随机顺序的统计实验；百分比说明本次受控微基准结果，不是通用硬件加速保证。

## SGLang 参考的身份

参考直接调用实际安装的 `DeepEPMoE.forward_deepgemm_w4a8_marlin_contiguous`，仅以轻量对象提供其权重、OAI 配置和无 offloader 状态，没有实例化模型或 scheduler。两侧使用同一独立 EP 驱动与对外 Top4 校验，使对照边界一致；因此它不是 SGLang 服务完整 MoE wrapper 的原样时延。

SGLang 使用 Torch 2.11 / Python3.10，插件使用 Torch 2.13 / Python3.12，分别在各自容器内调用本地原生扩展并独立 pack，未跨 ABI 搬运 SO 或 packed tensor。参考复用了历史启动脚本指定的 LightOP native 量化与 DeepGEMM 路径：

- `/home/sglang-ws/1020_minimax/minimax-sh/merged-runtime/lightop-int8-per-token-20261105-v2/prefill`
- `/home/sglang-ws/1020_minimax/minimax-sh/merged-runtime/prefill/deepgemm-megamoe-3f535bf`

实际 routed 核依旧分别为 WASP 与 Marlin。虽然内部权重布局不要求字节相同，但本轮外部输入量化字节、接收计数及最终输出全部对齐。

## 数值与安全检查

- **48/48** 完整 EP 输出与 SGLang 逐位一致：相对 L2=0，cosine=1；覆盖六种负载与全部八卡。
- 同一负载，24-block 与 64-block 输出逐位一致；两侧每 rank 的原始 fixture SHA256 相同。
- INT8 输入与 FP32 scale 的 SHA256 全部相同；每专家 aligned counts、接收行数全部相同。
- 独立 INT64/FP64 点积与显式 BF16 舍入参考：两侧小样例相对 L2 均为 `5.849291e-10`，max abs `7.450581e-9`；沿用既有严格 HIPC 检查门槛，没有放宽输出误差。
- 小样例覆盖 slot16、未使用专家、重复专家、无效 -1 路由、有效专家零权重、零输入；新模块还覆盖 0/1/9 行全空路由，以及 33792 padded rows 的 workspace 扩容、缩小、再次复用。
- 融合路由 guard 与独立 CPU 谓词对照：正常输入接受；-2、越界136、大 INT64 ID、NaN/Inf 权重及非零 padding 权重均拒绝；输入未被改写。
- shared 辅助流 callback＋外层 scale 的 join：8/8 rank 逐位一致。该检查使用受控 callback，不宣称验证了真实 shared W8 MLP。

输入量化的额外诊断保留在原始日志：CPU FP32 除法可能把数学上的半整数移到相邻浮点数。独立量化参考使用 FP64 理想值，允许的不同仅限一个 INT8 单位且位于半整数边界；专家点积 oracle 则对两侧输入相同的规范 INT8 字节。完整目标 shape 的两侧量化已经用 SHA256 证明逐字节相同，不依赖上述半整数例外。

## 独立 trace 的路径证据

下面是每 rank 8192 行均衡场景的 **一次 rank0 profile**，只说明命中和结构，不用于替代上面的 15 次无 profiler 统计。单位 ms，pack/scatter 列按各自实际实现归类。kernel 累计与 span 不可混加。

| 路径 | kernel数 | GPU span | 无kernel区间 | dispatch | combine | GEMM1+2 | pack/scatter |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧路径 | 54 | 23.539 | 1.864 | 3.409 | 3.277 | 7.237 | 1.280 |
| 新路径24 | 17 | 17.004 | 0.331 | 2.827 | 3.252 | 7.214 | 0.716 |
| 新路径64 | 17 | 14.133 | 0.356 | 1.445 | 2.577 | 7.212 | 0.718 |
| SGLang | 17 | 15.499 | 0.327 | 1.384 | 2.524 | 8.308 | 0.716 |

全部八卡的新路径 profile 都只有两次 routed GEMM、无 rocPRIM 排序；dispatch/combine grid 与 24/64 配置一致。主要收益来自传输、准备与调度流程收敛，GEMM1+2 在旧路径与新路径之间并没有同等幅度的下降。

## 使用与复现

源码：`triton_tmp_dir/minimax31_moe_sglang_v1.py`。本轮没有修改旧 `minimax31_deepep_runner.py` 或已有默认后端选择。

```python
from triton_tmp_dir.minimax31_moe_sglang_v1 import PackedExperts, AlignedMoE

# Raw checkpoint bytes must be CUDA INT8, with split gate/up rows and FP32
# checkpoint scale/16. Pack once using this process's own DeepGEMM build.
weights = PackedExperts.from_checkpoint_bytes(w13, w2, s13, s2)
moe = AlignedMoE(ep_group, weights, num_sms=64,
                 routed_scale=1.0, shared_fn=None)
try:
    output = moe(hidden_bf16, physical_top4_int64, route_weights_fp32)
finally:
    moe.close()
```

外部 Top4 必须已经完成逻辑→物理副本选择；-1 padding 对应权重必须为0。每个实例绑定一张卡、一条主 stream 和一个非重入 DeepEP buffer；shared callback 必须只读输入。`ReceivedExperts` 接口仅接收已验证 DeepEP local-minus-one ABI 的输出；其结果是下次调用前有效的 owned scratch view。

若权重已在 producer 做过 HIPC pack，应构造 `PackedExperts(existing_w13, existing_w2, s13, s2)`，不要重复 pack 或改变 IPC 权重。后续接入服务时还需对新的加权舍入语义和完整模型质量做验证，本轮模块结果不替代该验证。

从 plugin 根目录运行 standalone 工具（`RUN` 指向可写实验目录）：

```bash
export PYTHONPATH="$PWD/triton_tmp_dir:$PWD:$PWD/adapter_extension/src"
python benchmarks/kernels/minimax31_moe_module_alignment.py --phase oracle --root "$RUN"
python benchmarks/kernels/minimax31_moe_module_alignment.py --phase guard --root "$RUN"
python benchmarks/kernels/minimax31_moe_module_alignment.py --phase fixture --root "$RUN"
python -m torch.distributed.run --standalone --nproc-per-node=8   benchmarks/kernels/minimax31_moe_module_alignment.py --phase bench --root "$RUN" --profile
```

`--reference` 调用 SGLang 实际方法；需要 SGLang 的独立解释器、上述 PYTHONPATH、`SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1`、原生 scatter/gather 和 OAI INT8 开关。SGLang 容器 `/public` 只读，本次在 `/tmp/minimax31_moe_align_20261115` 创建输入 fixture 软链接并保存结果，再复制至任务目录。禁止同时运行两个 GPU benchmark。

## 交付与原始产物

归档制品目录：`moe_module_alignment_20261115/`（独立实验归档）。

- `module_alignment_summary.json`：全部时延数组、48组数值结果、输入/计数身份及32份 profile 摘要。
- `test_module.py`、`audit_results.py`：fixture、oracle、guard、八卡基准和审计脚本；对应代码也放入 plugin 的 `benchmarks/kernels/`。
- `fixture_rank*.pt`：八个相同输入来源，原始 SHA256 保留在 summary。
- `bench_ours_rank*.json`、`sglang_reference_module/bench_sglang_rank*.json`：逐 rank 原始时延。
- 各 variant 的 `*_rank*.trace.json`、`*_balanced_rank*.pt`、`*_skew_rank*.pt`：独立 trace 和完整输出。
- `oracle_optimized.json`、`oracle_sglang.json`、`guard_validation.json`、`audit_results.log`：已通过的检查。
- `preliminary_segments/` 为首轮探索记录，未使用最终 native 量化配置；最终数字只取顶层及 `sglang_reference_module/`。

基线插件提交为 `4ce0e7de3f4a5a0a24efb28cb836740abe5b70bb`。本次只新增独立模块、独立 benchmark/audit 工具和说明，不注册或切换服务后端。

## Lifecycle review (2026-11-16)

An `AlignedMoE` instance now admits its bound CUDA stream before quantization,
shared work or DeepEP communication. `close()` takes the same non-reentrant lock
as calls, so an active call cannot race buffer destruction. Shared inputs are
recorded on the auxiliary stream before invoking the callback; both callback and
routed-path exceptions enqueue the shared-stream join before releasing the lock.
Normal expert math, weight layouts and numerical thresholds are unchanged.

CPU contract tests are in
`tests/vllm033/test_minimax31_aligned_lifecycle_cpu.py`. On nmz20 in the original
bench container, an eight-rank synthetic EP8 probe passed the independent integer
expert reference including BF16 rank partials, normal-output bit equality against
the pristine branch, shared output/scale composition, injected routed failure and
recovery, foreign-stream refusal, and inflight-close refusal on all ranks. The
existing single-GPU oracle and 33792-padded-row workspace reuse also passed.
These component checks do not certify full-checkpoint model accuracy.
