# HCU mHC Pre/Post 接入记录（2026-10-13）

本次修改只在 vllm-plugin-das 中完成。vLLM 源码、启动脚本及 AITER/TileLang/TVM-FFI 安装未修改。

## 对照 SGLang 的依据

- [a0840f4](https://github.com/HYGON-AI/sglang-das/commit/a0840f493c5012862925011a80a13f0d7dd68579)：HCU Post 调用 `aiter.ops.tilelang.mhc_post_fwd`；AITER Pre 不融合 decoder RMSNorm，调用者仍需执行该 Norm。
- [0956953](https://github.com/HYGON-AI/sglang-das/commit/0956953)：调整 mHC TileLang 初始化兼容处理；这是初始化修复，不是新增 mHC 算子。不修改安装包中的旧 Allocate 补丁。
- 对照分支快照 `b2a77d5c483a7112764ea5fb0dd6984c49a071f2`：`mhc.py` 向 `pre_big_fuse_tilelang` 传入紧凑的 24 列投影、3D residual、2D post/comb 输出；V4.1 的上一子层 Pre 路径使用 `_hc_mix_stats`、`hc_split_sinkhorn` 和 `_hc_combine`。
- SGLang 的跨层融合有平台门控，HCU 不应直接套用 gfx95 融合路径。插件默认使用独立 Post 与 Pre。

## 接入范围

| 接口 | 插件路由 | 当前环境验证 |
|---|---|---|
| `aiter.ops.mhc.mhc_pre` | 普通 MHCPreOp | GPU 数值/真实类调用通过 |
| `aiter.ops.mhc.mhc_post` | 主模型、delayed Pre 内部 Post、DSpark 末尾 Post | 保留 v3，GPU/Profiler 通过 |
| `aiter.ops.tilelang.pre_big_fuse_tilelang` | 显式开启 TileLang 后，用于普通/分阶段 delayed Pre | 参数和张量布局测试通过；真实执行等待用户修复依赖 |
| `aiter.ops.tilelang.mhc_post_fwd` | 显式开启 TileLang 后，BF16 Post 路由 | 接口布局测试通过；真实执行等待依赖修复 |
| `aiter.ops.mhc.mhc_fused_post_pre` | 普通 MHCFusedPostPreOp 的可选路径 | 当前包缺失；输出顺序适配和独立 Post+Pre 回退已测试 |

## V4.1 delayed Pre

冷替换的 HCU `rocm_aiter_ops` 原来没有 `mhc_pre_delayed`，因此 Pre 一直回退到 Torch。本次补齐适配方法和 `MHCPreDelayedOp.forward_hip` 路由。

实际路径：

1. 如果有上一子层输出，通过 MHCPostOp 应用 Post（默认 v3）。
2. AITER `mhc_pre_gemm_sqrsum` 生成投影与平方和。
3. AITER `mhc_pre_big_fuse` 或 opt-in 的 `pre_big_fuse_tilelang` 生成 post、comb。
4. 现有 `mhc_pre_mix_triton` 从同一投影恢复本次 pre，供下一子层使用。
5. 用传入的上一子层 pre 调用 `hc_collapse_triton`；没有 pre 时选择 residual 的第 0 路。
6. 如果调用者传入 decoder Norm 权重，单独执行一次 Norm。

第一层 embedding 的窄投影也适配：重复 embedding 保持均方值，仅第 0 路填入投影权重。它与直接用窄 `x` 计算统计等价，已做 GPU 数值对照。

这是前一子层系数语义，不能直接用普通 `mhc_pre` 的 layer_input 替换。当前 HCU staged 投影使用保守的 split-K=1，不依赖当前 AITER 包中不存在的 gfx95 调优/融合辅助接口；此配置未完成整模型性能调优。

## 开关

```bash
export VLLM_HCU_USE_AITER_MHC=1                 # 默认开，mHC 总开关
export VLLM_HCU_USE_AITER_MHC_PRE=1             # 默认开
export VLLM_HCU_USE_AITER_MHC_POST=1            # 默认开
export VLLM_HCU_USE_AITER_TILELANG_MHC=0        # 默认关，与 SGLang HCU opt-in 一致
export VLLM_HCU_USE_AITER_MHC_FUSED=0           # 默认关
```

这些开关还受 `VLLM_HCU_USE_CUSTOM_OPS` 控制。`VLLM_ROCM_USE_AITER=0` / `VLLM_ROCM_USE_AITER_MOE=0` 不关闭 HCU mHC 专用路径。关闭 Post 后，即使 Pre 已可用，也明确使用原生 Post，避免上游 `HAS_AITER_MHC` 重新开启它。

修复 TileLang 依赖后，设置 `VLLM_HCU_USE_AITER_TILELANG_MHC=1` 并重启服务，即可选择 SGLang 使用的两个 TileLang 接口。导入失败会记录原因并回退到 HIP；不会改动安装包。普通融合接口需要安装提供该接口的包，并显式开启 FUSED。V4.1 delayed 路径仍保留上一子层 pre 的独立实现。

GPU 快速路径保守限定为 4 路、连续 BF16（Post 也支持已有 FP16 路径）、hidden 为正且可被 1024 整除、系数和投影权重 FP32。不支持的形状回退到原生实现，空输入不启动 GPU 算子。

## 验证与限制

证据目录：`/workspace/deepseekv4.1-flash/serve_start/run/20261013_mhc_full/`。

- delayed Pre：hidden 4096/5120/7168 × tokens 1/8/128，共 9 组；第一层 embedding 3 组。与 Torch 对照通过，系数最大误差约 7e-6。
- 真实 MHCPreOp、MHCPreDelayedOp、MHCFusedPostPreOp 调用及 decoder Norm 对照通过。
- `torch.compile(backend="eager", fullgraph=True)` 通过；GPU 图捕获和 replay 通过。
- Profiler 确认实际执行 `mhc_pre_gemm_sqrsum`、`mhc_pre_big_fuse`、`mhc_post_kernel_v3`。
- 回归测试结果见 `regression.log`；覆盖上一子层 pre、embedding、空输入、TileLang 布局、依赖缺失、融合输出顺序和开关隔离。
- 当前 AITER 没有 `mhc_fused_post_pre`；TileLang 被 `tvm_ffi.dataclasses.Enum` 导入错误阻断。上述分支只验证适配契约，未验证真实算子执行。
- 整模型 HumanEval 已于 2026-10-13 16:27 在 nmz16 完成（本次 Pre+Post 接入版本）：100 个样本，Accuracy / Pass@1 为 98%；DSpark 接受率 87.4%（7140/8165），平均接受长度含 bonus 为 5.37。评测耗时约 3 分 26 秒，不包含模型加载与预热。
- 本次结果比此前仅 Post 版本的 99% 低 1 个百分点；两次运行不足以判断是否由 Pre 接入引起，尚未做同环境关闭 Pre 的对照评测。
- 结果目录：`/workspace/deepseekv4.1-flash/serve_start/runs/20260921_dsv41_flash_vllm_sota/accuracy/20261013_155933/20261013_162337/`；外层目录保存服务日志和 `prometheus_spec_decode.txt`。脚本已自动停止服务；16:32 检查本次脚本、API、EngineCore 和 8 个 Worker PID 均已退出。
