# Sparse MLA prefill QK-FP8

可选的 gfx938 sparse prefill 路径，沿用 `sparse_prefill_qk_fp8` 接口：BF16 Q/KV 输入，选中 FP8 时内部 QK 使用 FP8，PV 使用 BF16，输出 BF16；矩阵累加为 FP32。它不改变模型权重、KV cache 格式或 decode 实现。

本次验证的 FlashMLA 版本：

```bash
pip install flash-mla==1.2.0+dtk2604.torch2110.2611131851.g2f4bd3 -i https://pypi.sourcefind.cn/nightly/dtk/
```

安装包含本适配的插件后，启动进程前设置：

```bash
export FLASH_MLA_DSA_FP8_QK=1
```

不开启时保留原公共 API。该路径适用于 eager prefill；不支持 CUDA Graph 内的数据检查。图捕获、短 top-k、无效索引或不支持的形状均使用原路径，不修改模型的 mask 语义。

## 接口与条件

```python
from vllm_hcu.v1.attention.ops.flashmla_qk_fp8 import sparse_prefill_qk_fp8

out, lse = sparse_prefill_qk_fp8(q, kv, indices, sm_scale, d_v=512)
```

- `q`：BF16，`[s_q, 64/128, 576]`；`kv`：BF16，`[s_kv, 1, 576]`。
- `indices`：INT32，`[s_q, 1, 2048]`；每个索引必须在 `[0, s_kv)`，不能使用 `-1` 或越界 padding。
- 三者位于同一 gfx938 设备，末维连续，入口要求的行/head stride 为正。
- `h_q=64`：`s_kv>=2048`；若小于 24576，需要 resident prefill 开启且 KV 行 stride 小于 8192。
- `h_q=128`：`s_kv>=24576`。
- 不得设置 `FLASH_MLA_DISABLE_DSA_MLS_PREFILL`，包括将其设为 `0`。

直接入口对不支持的输入抛出 `ValueError`。插件原有 `flash_mla_sparse_fwd` 入口负责回退；vLLM 传入 `topk_length` 时，只有每一行均为 2048 才省略原生参数。检查每次调用的实际索引和计数，不缓存资格判断；Triton 合并范围/计数检查，随后有一次 GPU→CPU 同步。

兼容入口还根据工作量分流。记 `W = s_q * h_q`：

- 不带 sink：`W < 32768` 且 `s_kv >= 2W` 时使用 BF16。
- 带 sink：只有 `W > 32768` 且 `8W > s_kv + 131072` 时尝试 FP8，其余沿用 BF16。

原生 FP8 prefill 会量化整个 KV workspace；sink 校正还需一次输出缩放。小 Q、长 KV 下这些成本容易超过收益。阈值来自 gfx938 的工作量扫描，只读取形状元数据。BF16 分流避免新增 FP8 专用的 indices 校验、CPU 同步和 Q/K 再量化。直接调用 `sparse_prefill_qk_fp8` 仍明确请求 FP8，不受性能分流影响。

## Attention sink

HY4 的 `learnable_sink=True` 会向 attention 传入每 head 一个 FP32 sink。原生 QK-FP8 binding 不接收 sink，插件兼容入口先计算无 sink 的 FP8 output/LSE，再用一个 Triton kernel 原地缩放：

```text
output_sink = output * sigmoid(LSE - sink)
```

sink 是 value 为零的虚拟 token，仅增加 softmax 分母，因此该缩放保留 sink 语义。权重用 FP32 计算，输出仍为 BF16；`+inf` sink 强制输出零，`-inf` 保持原输出。返回的 LSE 按原公共接口约定不包含 sink，HY4 后端需要跨 rank 合并时再折入 sink。短计数、非法索引及图捕获继续调用原生 BF16 sink 路径，不重复应用 sink。

sink 要求同设备、FP32、形状 `[h_q]` 且 stride 为正；其他布局交给原公共接口处理。模型调用方不需要新增启动参数。

原生 QK 路径没有初始化 `max_logits`：直接入口仅返回 `out, lse`；插件兼容入口返回 `out, None, lse`，现有 MLA 调用方只使用 output/LSE。

## 算子验证与性能

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

当前版本共 **92 项通过**：65 项 CPU 合约、27 项 HCU 测试。Sink 参考通过显式拼接虚拟 token logits 后计算 FP32 softmax，不复用实现的 sigmoid 缩放公式；另覆盖短计数、非法索引回退时只应用一次 sink。

## PD 实测

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
