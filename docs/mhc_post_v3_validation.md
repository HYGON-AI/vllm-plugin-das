# AITER mHC Post v3 插件接入与验证

日期：2026-10-12。目标：在用户指定的 vLLM / vllm-plugin-das 源码组合中，使用 AITER 的 `mhc_post_kernel_v3`。

## 实现

- `vllm_hcu/patch/worker/core_fix/patch_mhc_backend.py`：单独探测 Post 接口，修改 `MHCPostOp.forward_hip` 的路由。即使旧 AITER 缺少 delayed Pre、导致 `HAS_AITER_MHC=False`，Post 仍可以使用现有 `torch.ops.vllm.mhc_post_aiter`。
- `vllm_hcu/patch/worker/op_opt/patch_mhc_post_dspark.py`：替换 DeepSeek-V4.1 AMD DSpark 模块中的末尾 Post 函数别名，覆盖该处直接调用 Torch 实现的路径。
- `vllm_hcu/patch/worker/__init__.py`：将 DSpark 回调注册到现有 Worker 生命周期。
- `vllm_hcu/platforms/envs.py`：新增默认开启的 `VLLM_HCU_USE_AITER_MHC_POST`。同时遵守 `VLLM_HCU_USE_AITER_MHC` 和 `VLLM_HCU_USE_CUSTOM_OPS`。
- 扩展现有 `tests/runtime_patch/test_deepseek_v4_dspark_patches.py`，覆盖 Post 独立启用、开关回退、坏形状防护、签名契约、DSpark 路由及幂等性。

默认在模型构造与 warmup 前设置 `AITER_MHC_POST_KERNEL=v3`；用户已显式设置 `legacy`、`auto` 等值时保留其选择。没有在每次推理或图捕获期间修改环境变量。继续使用 vLLM 已注册的自定义算子及 fake 实现，不导入 AITER TileLang 路径。

## 形状与数值边界

快路径要求：GPU 输入在同一设备、连续布局；4 路残差；hidden size 为正的 1024 倍数；x/residual 同为 BF16 或 FP16；混合系数为 FP32。输出保持原残差的形状和 dtype，输入不被修改。

当前容器直接探针发现，v3 对 256、512、768、1280、2560 等 hidden size 的输出异常，属于 b32/b64 变体。接入因此只启用已验证的 b128 路径，其他形状、空 batch、非连续布局等走原生数值实现。4096、7168 属于快路径。

GPU 数值验证不是逐位一致：BF16 使用 `atol=0.01, rtol=0.01`，最大绝对差 0.03125；FP16 使用 `atol=0.002, rtol=0.002`，最大绝对差 0.001953125。整模型 token 输出、HumanEval 精度及端到端性能尚未验收。

## 本次验证

环境：nmz16，容器 `vllm028_0915_leyang`，实测 GPU `gfx938:sramecc+:xnack-`，PyTorch 2.11.0 / HIP 6.3.26113，AITER `0.1.6+dtk2604.torch2110.2609111704.g40137b`。

源码基线：vLLM `4c349cf6340fac8f1643476af9c2fb5e4b3252d6`；插件 `leyang/pd` / `336f4ad14fe71f29078edbaff1f643a6989a2c95`。

| 验证 | 结果 |
|---|---|
| 更新后的 DeepSeek/DSpark runtime-patch 测试 | 58 passed |
| 真实 `MHCPostOp` GPU 数值 | 18 组通过：BF16/FP16 × hidden 1024/4096/7168 × token 1/8/32 |
| 空 batch、512/1280/2560、3 路残差、非连续输入 | 回退对照通过 |
| 多维 outer batch、输入不变、输出不别名 | 通过 |
| 子开关和 custom-op master 的 A/B | 通过 |
| `torch.compile(..., fullgraph=True, backend="eager")` | 通过 |
| CUDA/HIP 图捕获与输入变化后的回放 | token 1、32 各 3 次通过 |
| DSpark 实际模块末尾 Post | 数值、开关回退及 v3 kernel trace 通过 |
| `apply_worker_patches()` 真实自动回调 | 两个入口均自动应用，并实际执行 v3 |
| Ruff 检查、git diff --check、production boundary | 通过 |

真实 trace 中的 kernel：

```text
void aiter::mhc_post_kernel_v3<unsigned short, 128, 4, 8>(...)
```

扩大检查时发现两项既有 GLM 测试在导入未改动的旧 TileLang/boltops 后端时失败：`tvm_ffi.dataclasses.Enum` 不存在。这不阻塞本次非 TileLang Post 路径。旧 `test_worker_dispatcher.py` 还硬编码要求插件目标为 0.28.1，而用户当前源码声明的目标版本为 `0.1.dev21510+g4c349cf63.das.4c349cf.dtk2604`，因此该套件拒绝收集；没有篡改版本或跳过断言来制造通过结果，改用真实 Worker 自动回调完成验证。

## 使用

在容器内使用本次修改的源码启动：

```bash
export PYTHONPATH="/workspace/deepseekv4.1-flash/serve_start/vllm-plugin-das:/workspace/deepseekv4.1-flash/serve_start/vllm:${PYTHONPATH:-}"
export VLLM_PLUGINS=hcu,hcu_model,hcu_ops
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_AITER_MHC=1
export VLLM_HCU_USE_AITER_MHC_POST=1
export AITER_MHC_POST_KERNEL=v3
```

然后用原有启动脚本启动服务。已经运行的 Python Worker 需要重启才能加载新代码。关闭新 Post 路由作对照：`VLLM_HCU_USE_AITER_MHC_POST=0`。

注意：关闭新开关恢复原始路由；如果未来完整 AITER mHC 能力可用，原始路由本身也可能选择 AITER，不保证始终是 Torch。当前实测 `HAS_AITER_MHC=False`，关闭开关时为 Torch 回退。

## 产物位置

宿主机：`/public/home/leyang/workspace/deepseekv4.1-flash/serve_start/run/20261012_mhc_post_v3/`。

容器：`/workspace/deepseekv4.1-flash/serve_start/run/20261012_mhc_post_v3/`。

- `runtime_regression_final.log`：58 项回归结果。
- `gpu_validation.log`、`validation.json`：真实 vLLM Post 数值、编译、图回放验证。
- `vllm_mhc_post_v3_trace.json`：真实 MHCPostOp trace。
- `dspark_validation.json`、`dspark_mhc_post_v3_trace.json`：DSpark 末尾 Post 验证。
- `worker_auto_callback.json`、`worker_auto_callback_trace.json`：自动回调与实际 kernel 证据。
- `runtime_regression.log`、`worker_registration.log`：扩大检查中发现的既有限制。
- `changes.patch`：完整改动，含新增文件。

验证在隔离环境 `.venv` 中复用容器依赖，未重新安装 PyTorch/AITER。

## 2026-10-13 Pre 接入更新

本记录的测试对应此前的 Post 接入版本。新增的普通/延迟 Pre、可选 TileLang 路由及验证见 [mhc_full_integration.md](mhc_full_integration.md)。最新版关闭 Post 开关时明确使用原生 Post，即使 Pre 已可用也不会被上游能力标志重新开启。
