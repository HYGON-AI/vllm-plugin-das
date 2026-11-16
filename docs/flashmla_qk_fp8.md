# Sparse MLA prefill QK-FP8

可选的 gfx938 sparse prefill 路径，沿用 `sparse_prefill_qk_fp8` 接口：BF16 Q/KV 输入，选中 FP8 时内部 QK 使用 FP8，PV 使用 BF16，输出 BF16；矩阵累加为 FP32。它不改变模型权重、KV cache 格式或 decode 实现。

原生 sink 接口使用以下 FlashMLA 版本：

```bash
pip install flash-mla==1.2.0+dtk2604.torch2110.2611161646.gf58f93 -i https://pypi.sourcefind.cn/nightly/dtk/
```

安装包含本适配的插件后，启动进程前设置：

```bash
export FLASH_MLA_DSA_FP8_QK=1
```

不开启时保留原公共 API。该路径适用于 eager prefill；不支持 CUDA Graph 内的数据检查。图捕获、短 top-k、无效索引或不支持的形状均使用原路径，不修改模型的 mask 语义。

## 接口与条件

```python
from vllm_hcu.v1.attention.ops.flashmla_qk_fp8 import sparse_prefill_qk_fp8

out, lse = sparse_prefill_qk_fp8(
    q, kv, indices, sm_scale, d_v=512, attn_sink=sink
)
```

- `q`：BF16，`[s_q, 64/128, 576]`；`kv`：BF16，`[s_kv, 1, 576]`。
- `indices`：INT32，`[s_q, 1, 2048]`；每个索引必须在 `[0, s_kv)`，不能使用 `-1` 或越界 padding。
- 三者位于同一 gfx938 设备，末维连续，入口要求的行/head stride 为正。
- `h_q=64`：`s_kv>=2048`；若小于 24576，需要 resident prefill 开启且 KV 行 stride 小于 8192。
- `h_q=128`：`s_kv>=24576`。
- 新 frontend 还须通过其 `_dsa_qk_fp8_shape_ok` 检查；当前原生 sink 版本要求 `s_kv>=24576`，包括 h64。上面的 resident 小 KV 条件仅适用于旧包。
- 不得设置 `FLASH_MLA_DISABLE_DSA_MLS_PREFILL`，包括将其设为 `0`。

直接入口对不支持的输入抛出 `ValueError`。插件原有 `flash_mla_sparse_fwd` 入口负责回退；vLLM 传入 `topk_length` 时，只有每一行均为 2048 才省略原生参数。检查每次调用的实际索引和计数，不缓存资格判断；Triton 合并范围/计数检查，随后有一次 GPU→CPU 同步。

兼容入口先根据工作量筛选候选。记 `W = s_q * h_q`：

- 不带 sink：`W < 32768` 且 `s_kv >= 2W` 时使用 BF16。
- 带 sink：只有 `W > 32768` 且 `8W > s_kv + 131072` 时进入候选，其余沿用 BF16。

候选调用再用实际输入校准成本：每个设备、stream、Q shape、4K KV 长度区间和布局组合对比3组 BF16/完整 FP8 调用，交替执行顺序。后续调用用 CUDA event 的非阻塞 `query()` 读取已完成的计时；只有 FP8 中位耗时低于 BF16 的90%才选 FP8。计时未完成时先走 BF16，不额外等待。最多缓存64组性能选择，每1024次选定路径调用重新校准，适应索引分布变化。

校准期返回 BF16 结果，会临时增加一次 attention 计算和一个输出的显存；不适用于把首次调用作为稳定态性能的比较。缓存只保存耗时事件与性能选择，不保存 tensor 或内容合法性：选中 FP8 后仍逐次检查当前 indices/counts；计数、索引变化仍按原语义回退。图捕获和开关关闭时不校准。已有非连续正 stride sink 在 BF16 对照和 FP8 两条路径都按需转连续，普通连续模型参数不拷贝。

原生 FP8 prefill 会量化整个 KV workspace。旧包的 sink 校正另需一次输出缩放，新包由原生 kernel 处理 sink；两者都需要框架索引校验和同步。单靠 Q/H/KV shape 不能预测稀疏索引重用，随机索引的算子收益可能不能复现到真实 indexer 输入。BF16 分流避免新增 FP8 专用的 indices 校验、CPU 同步和 Q/K 再量化。直接调用 `sparse_prefill_qk_fp8` 仍明确请求 FP8，不受性能校准影响。

### 真实输入回放与分派校准

单 P、PCP16、离线 R16、workspace dedup 的正常5步采样主要是 Q `[378,64,576]` 和 `[756,64,576]`。两步 profile 另捕获 Q1134。Q378 原本已经回退 BF16；Q756 会通过旧阈值，而 rank0 一部分调用校验失败后仍退回 BF16。实际 Q756 的 FP8 主 attention kernel 约1.544ms，BF16约1.745ms；新增 amax 和量化合计约0.675ms，尚未计每层内容校验和同步就已超过节省时间。

保存 rank1 一次真实 Q756/KV90694 的完整输入后回放。原样本前16行中，相邻 top-k 集合交集平均占80.12%，均匀随机对照约2.34%。只替换 indices，就会改变两条路径的相对成本。下面是校准后3组 wall time 中位数，每组10次；FP8 列包含原来的内容检查和同步。

| Q shape / KV 长度 | indices | BF16（ms） | 强制 FP8（ms） | 校准后（ms） | 选择 |
|---|---|---:|---:|---:|---|
| [756,64,576] / 90694 | 真实输入 | 1.856 | 2.762 | 1.833 | BF16 |
| [6168,64,576] / 177280 | 原真实索引行循环扩展 | 13.401 | 15.485 | 13.447 | BF16 |
| [6168,64,576] / 177280 | 均匀随机 | 30.627 | 18.257 | 18.293 | QK-FP8 |

Q6168 的真实索引行扩展是几何对照，不是真实捕获的大 batch。它说明 shape 相同也不能沿用随机索引结论。校准后保留随机大 Q 约40%的完整调用收益，同时让该真实 Q756 回到 BF16 耗时附近。

Q6168 的单卡校准调用峰值额外分配约1093MiB，稳态 FP8 约704MiB、BF16约389MiB；校准期较 FP8 多约389MiB。负载局部性变化后，最多1024次调用内会重新校准；同一 key 的不同层仍共享性能选择，并不保证每层都取到各自最优。

## Attention sink

HY4 的 `learnable_sink=True` 会向 attention 传入每 head 一个 FP32 sink。新 frontend 提供 QK eligibility 检查时，插件将 sink 直接传给原生 binding，不再追加缩放 kernel。只有每行计数均为2048且索引全部合法时，才省略原生计数参数以启用 FP8。

旧包没有该 frontend 检查时，保留原实现：先计算无 sink 的 FP8 output/LSE，再用一个 Triton kernel 原地缩放：

```text
output_sink = output * sigmoid(LSE - sink)
```

sink 是 value 为零的虚拟 token，仅增加 softmax 分母，因此该缩放保留 sink 语义。权重用 FP32 计算，输出仍为 BF16；`+inf` sink 强制输出零，`-inf` 保持原输出。返回的 LSE 按原公共接口约定不包含 sink，HY4 后端需要跨 rank 合并时再折入 sink。短计数、非法索引及图捕获继续调用原生 BF16 sink 路径，不重复应用 sink。

sink 要求同设备、FP32、形状 `[h_q]` 且 stride 为正。原生 binding 要求连续 sink，适配仅为非连续的正 stride sink 做一次小拷贝；连续模型参数不拷贝。其他布局交给原公共接口处理。

旧包的原生 QK 路径未初始化 `max_logits`。适配继续保持原有返回约定：直接入口仅返回 `out, lse`，兼容入口返回 `out, None, lse`；现有 MLA 调用方只使用 output/LSE。

## 算子验证与性能

### 实参校准版服务复测（2026-11-17）

与下节静态版本使用相同 FlashMLA 包和服务/客户端参数：单16卡 BW1100、PCP16、离线 R16、workspace dedup、输入85K/输出1、到达率2.55。新起服务，先146预热，再2920正式轮，组内保留前缀 KV。正式轮没有尺寸探针或 profiler。

| 2920 请求 | BF16 对照 | 旧静态 FP8 | 实参校准 |
|---|---:|---:|---:|
| 成功 / 失败 | 2920 / 0 | 2920 / 0 | 2920 / 0 |
| 客户端平均 TTFT（ms） | 1564.66 | 2441.53 | 1553.92 |
| 中位 TTFT（ms） | 1520.48 | 2422.33 | 1500.18 |
| P95 TTFT（ms） | 1903.26 | 3028.84 | 1887.93 |
| P99 TTFT（ms） | 2169.16 | 3392.11 | 2152.43 |
| 完成请求/s | 2.54648 | 2.54455 | 2.54687 |
| 引擎平均 TTFT（ms） | 1467.150 | 2288.533 | 1456.561 |
| 平均排队（ms） | 0.015 | 0.057 | 0.012 |
| 平均 prefill（ms） | 741.014 | 1221.559 | 738.657 |
| 前缀命中率（%） | 92.91294 | 92.91294 | 92.91294 |

客户端平均 TTFT 较旧静态 FP8 降低36.35%，与 BF16 对照接近（低0.69%，不单独声称这点差异是稳定加速）。收益来自修复错误分派，该负载中选择 BF16，不是 FP8 主算子单项超过 BF16。固定到达率的完成 RPS 不代表峰值容量。146预热平均TTFT为3751.40ms，包含首次长前缀计算和校准。

本版通过3137项 CPU contract（35 skipped）及35项独立 HCU测试。增加缓存 FP8 选择后原地修改 indices/counts 的回退验证；选定 FP8 的数值用例固定性能选择，仍运行真实检查和 kernel。不同数据局部性的实际回放另行验证性能选择。独立复审发现的问题均已修复，完整分布式 CI 和整模型精度仍不在这些测试的证明范围内。

正式服务8次显存采样中，各卡最少空闲9147MiB；无抢占。服务正常退出后16卡均回到2MiB。三组结果都有相同25条响应文本异常，见文末审计说明；排除同一并集后，平均TTFT为 BF16 1563.90ms、旧静态FP8 2440.42ms、校准版1553.04ms，正式表仍保留全部请求。

### 原生 sink 版本（2026-11-16，gf58f93）

本节保留加入实际输入校准前的静态分派结果；后续定位和校准后的算子回放见上节。

同一 gfx938、同一新包，对比 BF16 与原生 sink FP8 完整适配调用，包含量化、索引校验和同步。每组预热5次、循环15次，3组取 wall time 中位数：

| Q shape | KV 长度 | BF16＋sink（ms） | 强制 FP8（ms） | 自动分流（ms） | 自动路径 |
|---|---:|---:|---:|---:|---|
| `[16,64,576]` | 32768 | 0.358 | 0.673 | 0.363 | BF16 |
| `[378,64,576]` | 85000 | 1.892 | 2.178 | 1.922 | BF16 |
| `[6168,64,576]` | 177280 | 30.584 | 18.313 | 18.311 | QK-FP8 |
| `[378,128,576]` | 85000 | 3.711 | 3.306 | 3.305 | QK-FP8 |

大 Q 场景耗时降低约40.1%，h128场景降低约10.9%；小 Q 强制 FP8 仍更慢，继续使用 BF16。Profiler 观察到 FP8 QK kernel，未观察到旧框架 sink 缩放 kernel。

该版本通过102项检查：69项 CPU 合约和33项 HCU 测试，包括原生 sink 只应用一次、独立 FP32 参考、无穷边界、真实 HY4 后端、计数/索引回退、非连续 sink 和满足 FP8 形状条件时的 graph 回退。随机输入相对 BF16 的输出 L2 差异约3.2%，不能代替整模型精度验证。

复现单 P 的 FP8 实验组时，参考[部署命令](https://github.com/HYGON-AI/inference-cookbook-das/blob/aa1813de3ccceb3e63c61d2d52cb8119dce075dd/docs/model-deployment/vllm/hy4.md)，在服务启动前加上：

```bash
export FLASH_MLA_DSA_FP8_FLAGS=1
export FLASH_MLA_DSA_FP8_QK=1
export VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP=1
```

BF16 对照只将 `FLASH_MLA_DSA_FP8_QK` 改为 `0`；两组均保留 workspace 去重。每组新起服务，先146、再2920，组内保持同一服务和前缀缓存。使用离线 R16 EPLB、PCP16，输入85K、输出1，单 P 不设置固定 MTP 接受率。

| 2920 条正式测试 | BF16＋去重 | 原生 sink FP8＋去重 |
|---|---:|---:|
| 成功 / 失败 | 2920 / 0 | 2920 / 0 |
| 客户端平均 TTFT（ms） | 1564.66 | 2441.53 |
| 中位 TTFT（ms） | 1520.48 | 2422.33 |
| P95 TTFT（ms） | 1903.26 | 3028.84 |
| P99 TTFT（ms） | 2169.16 | 3392.11 |
| 完成请求/s | 2.54648 | 2.54455 |
| 引擎平均 TTFT（ms） | 1467.15 | 2288.53 |
| 平均 prefill（ms） | 741.01 | 1221.56 |
| 前缀命中率（%） | 92.91294 | 92.91294 |

该静态分派版本的客户端平均 TTFT 高56.04%。后续5步尺寸采样、两步 profile 和完整真实输入回放确认，Q756 的预处理/检查成本和真实索引重用是关键，见上节。此表不是校准版服务结果；未完成整模型精度验证。

### 旧包测量与通用测试命令

以下表格为 2026-11-14 旧包 g2f4bd3 的结果，sink 使用框架缩放。原生 sink 新包的测试和性能单独记录，不能混用版本归因。

不启动模型或 HTTP 服务，使用一张空闲卡：

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest tests/runtime_patch/test_flashmla_qk_fp8_accuracy.py -q
HIP_VISIBLE_DEVICES=0 python benchmarks/bench_flashmla_qk_fp8.py --output result.json
HIP_VISIBLE_DEVICES=0 python benchmarks/bench_flashmla_qk_fp8.py --sink --output sink-result.json
HIP_VISIBLE_DEVICES=0 python benchmarks/bench_flashmla_qk_fp8.py --sweep --warmup 3 --repeats 5 --output sweep.json
```

基线是同一 wheel 的 BF16 路径，显式提供完整 `topk_length`；FP8 使用直接 binding。脚本同时记录原生入口和包含检查/同步的适配入口。计时预热 5 次，每组 15 次、3 组取中位数，表格使用每次调用的 wall time。输入由固定随机种子生成，indices 允许重复；以下形状来自此前 HY4 算子输入记录，不是保存的完整 Q/KV 激活回放。

| Q shape | KV shape | 节点 A BF16 / 适配 FP8（ms） | 节点 B BF16 / 适配 FP8（ms） |
|---|---|---:|---:|
| `[16,64,576]` | `[2048,1,576]` | 0.356 / 0.381 | 0.338 / 0.370 |
| `[378,64,576]` | `[85000,1,576]` | 1.906 / 2.206 | 1.899 / 2.124 |
| `[6168,64,576]` | `[177280,1,576]` | 30.809 / 18.217 | 30.594 / 18.089 |
| `[378,128,576]` | `[85000,1,576]` | 3.715 / 3.240 | 3.693 / 3.055 |

上表是加入工作量分流前的结果：大批次约 1.69–1.70 倍，小批次会被检查和同步开销抵消。更新后的 benchmark 同时记录强制 FP8 与自动分流耗时及选中的精度路径。

单张 gfx938 上追加的不带 sink 对照如下，包含检查和同步，计时口径同上：

| Q shape | KV 长度 | 强制检查后 FP8（ms） | 自动分流（ms） | 实际路径 |
|---|---:|---:|---:|---|
| `[16,64,576]` | 2048 | 0.386 | 0.349 | BF16 |
| `[378,64,576]` | 85000 | 2.187 | 1.896 | BF16 |
| `[6168,64,576]` | 177280 | 18.131 | 18.094 | QK-FP8 |
| `[378,128,576]` | 85000 | 3.091 | 3.093 | QK-FP8 |

另扫描 Q 长度 16/64/128/256/512/1024、head 数 64/128、KV 长度 24576/85000/177280，共 36 组。分流改变其中 19 组，测得约 1.38–3.41 倍加速；其余组仍使用相同的 FP8 实现。阈值是当前 gfx938/wheel 的经验分流条件，不保证其他硬件或版本有相同收益。

当前 native prefill binding 只接收 BF16 KV；保留 FP8 的大 Q 路径仍需要内部 K 再量化。

带 sink 的对照基线是原公共 BF16 路径。以下耗时包含 FP8 安全检查、同步和 sink 缩放：

| Q shape | KV 长度 | BF16＋sink（ms） | 自动分流＋sink（ms） | 实际路径 |
|---|---:|---:|---:|---|
| `[16,64,576]` | 2048 | 0.342 | 0.345 | BF16 |
| `[378,64,576]` | 85000 | 1.888 | 1.888 | BF16 |
| `[6168,64,576]` | 177280 | 30.770 | 18.556 | QK-FP8＋sink 缩放 |
| `[378,128,576]` | 85000 | 3.703 | 3.121 | QK-FP8＋sink 缩放 |

追加扫描了多请求工作区，KV 最长 442880，避免仅在单请求形状上选择路径。HY4 后端单函数调用、sink 的正负无穷边界、原始 LSE 及 graph 回退均经过原生 HCU 测试。整模型精度尚未验证。

该次验证共 **92 项通过**：65 项 CPU 合约、27 项 HCU 测试。Sink 参考通过显式拼接虚拟 token logits 后计算 FP32 softmax，不复用实现的 sigmoid 缩放公式；另覆盖短计数、非法索引回退时只应用一次 sink。

## PD 实测（2026-11-14，旧包 g2f4bd3）

沿用 [HY4 部署命令](https://github.com/HYGON-AI/inference-cookbook-das/blob/926d8f20c58734aa2f8cf090870bf1f189db32cf/docs/model-deployment/vllm/hy4.md)，P 叠加本适配并开启上述开关，D 不开启。两端各16卡，离线 R16、自然 MTP3，输入85K、输出1200、RPS2.55；新起服务，146预热后直接2920，不清KV。

| 2920 请求指标 | 原 sink/BF16 基线 | Sink＋自适应 FP8 |
|---|---:|---:|
| 成功 / 失败 | 2920 / 0 | 2920 / 0 |
| 客户端平均 TTFT（ms） | 2867.03 | 2865.55 |
| TTFT P95（ms） | 3486.73 | 3482.68 |
| TTFT P99（ms） | 3779.69 | 4163.52 |
| 完成 RPS | 2.45980 | 2.47747 |
| 输出吞吐（token/s） | 2951.76 | 2972.96 |
| 系统输出 TPM | 177105.42 | 178377.52 |
| 平均 TPOT（ms） | 30.34 | 30.32 |

平均 TTFT 基本持平，P99 单轮结果变高；不能据此声称有端到端 TTFT 或稳态容量提升。正式轮 P 前缀命中约92.92%，小 Q 仍使用 BF16；未统计正式轮 FP8 命中比例。独立诊断请求的 rank0/rank1 栈中观察到 sink 校正，确认真实模型可进入该路径；诊断请求不计入表格。保留 Draft，待尾延迟及整模型精度进一步验证。

基线虽设置了 FP8 开关，但原适配对非空 sink 直接调用 BF16，不能把旧开关开启的 PD 结果当成实际 FP8 证据。

两台 gfx938 各通过 42 项 CPU 合约测试和 15 项原生测试；另有 24 项既有 sparse MLA 回归通过（55 项无关用例未选择）。原生测试覆盖 h64/h128、3 个随机种子、短计数、负索引、越界索引、计数后的有效索引、strided validation 和 graph fallback。数值对照使用独立 FP32 attention，单测阈值为 max abs <0.05、relative L2 <0.05、LSE max abs <0.05。性能脚本对大形状抽查前 4 行 FP32 参考：FP8 相对 L2 约 3.18%–3.32%，BF16 约 0.2%；这是算子误差口径，不代表整模型精度。

Profiler 已观察到 `qk8_quant_kernel` 与 `fast_fp8qk_kernel`；该内核反汇编包含 `v_mmac_f32_16x16x32_fp8_fp8` 和 `v_mmac_f32_16x16x16_bf16`，对应 QK-FP8 与 PV-BF16。以上是算子口径，不能据此推断整模型精度或端到端收益。
