# Sparse MLA prefill QK-FP8

可选的 gfx938 sparse prefill 路径，沿用 `sparse_prefill_qk_fp8` 接口：BF16 Q/KV 输入，内部 QK 使用 FP8，PV 使用 BF16，输出 BF16；矩阵累加为 FP32。它不改变模型权重、KV cache 格式或 decode 实现。

本次验证的 FlashMLA 版本：

```bash
pip install flash-mla==1.2.0+dtk2604.torch2110.2611131851.g2f4bd3 -i https://pypi.sourcefind.cn/nightly/dtk/
```

安装包含本适配的插件后，启动进程前设置：

```bash
export FLASH_MLA_DSA_FP8_QK=1
```

不开启时保留原公共 API。该路径适用于 eager prefill；不支持 CUDA Graph 内的数据检查。图捕获、attention sink、短 top-k、无效索引或不支持的形状均使用原路径，不修改模型的 mask 语义。

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

原生 QK 路径没有初始化 `max_logits`：直接入口仅返回 `out, lse`；插件兼容入口返回 `out, None, lse`，现有 MLA 调用方只使用 output/LSE。

## 算子验证与性能

不启动模型或 HTTP 服务，使用一张空闲卡：

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest tests/runtime_patch/test_flashmla_qk_fp8_accuracy.py -q
HIP_VISIBLE_DEVICES=0 python benchmarks/bench_flashmla_qk_fp8.py --output result.json
```

基线是同一 wheel 的 BF16 路径，显式提供完整 `topk_length`；FP8 使用直接 binding。脚本同时记录原生入口和包含检查/同步的适配入口。计时预热 5 次，每组 15 次、3 组取中位数，表格使用每次调用的 wall time。输入由固定随机种子生成，indices 允许重复；以下形状来自此前 HY4 算子输入记录，不是保存的完整 Q/KV 激活回放。

| Q shape | KV shape | 节点 A BF16 / 适配 FP8（ms） | 节点 B BF16 / 适配 FP8（ms） |
|---|---|---:|---:|
| `[16,64,576]` | `[2048,1,576]` | 0.356 / 0.381 | 0.338 / 0.370 |
| `[378,64,576]` | `[85000,1,576]` | 1.906 / 2.206 | 1.899 / 2.124 |
| `[6168,64,576]` | `[177280,1,576]` | 30.809 / 18.217 | 30.594 / 18.089 |
| `[378,128,576]` | `[85000,1,576]` | 3.715 / 3.240 | 3.693 / 3.055 |

大批次约 1.69–1.70 倍，小批次会被检查和同步开销抵消，不保证每种形状都加速。保留显式开关，不添加针对本次测试形状的特判。

两台 gfx938 各通过 42 项 CPU 合约测试和 15 项原生测试；另有 24 项既有 sparse MLA 回归通过（55 项无关用例未选择）。原生测试覆盖 h64/h128、3 个随机种子、短计数、负索引、越界索引、计数后的有效索引、strided validation 和 graph fallback。数值对照使用独立 FP32 attention，单测阈值为 max abs <0.05、relative L2 <0.05、LSE max abs <0.05。性能脚本对大形状抽查前 4 行 FP32 参考：FP8 相对 L2 约 3.18%–3.32%，BF16 约 0.2%；这是算子误差口径，不代表整模型精度。

Profiler 已观察到 `qk8_quant_kernel` 与 `fast_fp8qk_kernel`；该内核反汇编包含 `v_mmac_f32_16x16x32_fp8_fp8` 和 `v_mmac_f32_16x16x16_bf16`，对应 QK-FP8 与 PV-BF16。未进行端到端性能或模型精度测试。
