# DeepSeek-V4 sparse FlashMLA 迁移至 v0.28.1

## 来源与范围

来源：[HYGON-AI/vllm-plugin-das PR #144](https://github.com/HYGON-AI/vllm-plugin-das/pull/144)，合并提交 `f892c75d591540d8ca5f3fc5a54294b5f3fc822b`。
目标分支：`v0.28.1_dpskv4_sMLA`，迁移前提交 `0bd777db`。

保留原 PR 的 SWA-only、C4A、C128A sparse decode、sparse prefill、dense metadata、独立开关与 GPU parity 脚本。补丁仅注册在 `vllm.models.deepseek_v4.amd.rocm`，不改通用 MLA、HY V4、其他模型或已存在的 BF16 cache kernel。

## v0.28.1 适配

- compressed metadata 基类改为 `DeepseekV4SparseMLAMetadataBuilder`。
- decode 接口兼容新增的 `adaptive_splits` 参数，回退时传递原值。
- SWA builder 兼容有/无 `replay_start` 的两种已核对接口；新版调用保留该参数，包括原生回退路径。
- 仅 `cache_dtype=fp8_ds_mla` 且 local query heads 受支持时，移除 AITER ragged buffers 并生成 FlashMLA scheduler metadata。
- BF16/auto 等 cache 保留原生 builder 与 decode，继续使用目标分支的 BF16 cache 修复。
- prefill 对非 512 维、非 BF16 或不支持的 heads 回退；缺少 dense indices、lengths 或 sink 时也调用原实现。

本次遵循原 PR 的 kernel 支持范围：decode local heads 为 1–16、64、128；prefill 为 64、128。32-head decode 使用原生路径。未包含原分支后续两个 head-padding 优化提交。

## 使用与回退

默认开启两个独立开关。FlashMLA decode 需要 584-byte packed FP8 cache 和 OCP FP8；prefill guard 限定已验证的 gfx936/gfx938。需要安装导出 `sparse_decode_fwd` / `sparse_prefill_fwd` 的 FlashMLA 扩展，以及兼容的 LightOP。

在进程启动前设置开关；修改后重启服务，以重新分配对应 metadata buffers：

```bash
export VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE=False
export VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL=False
```

## 验证

当前安装的目标包：`vllm 0.28.1+das.77acaf6.dtk2604`。CPU 测试通过设置 `VLLM_TARGET_DEVICE=cpu` 避免 ROCm 平台初始化；这些测试不证明 GPU kernel 的精度或性能。

```bash
VLLM_TARGET_DEVICE=cpu python -m pytest \
  tests/runtime_patch/test_rocm_flashmla_sparse_adapter.py \
  tests/runtime_patch/test_deepseek_v4_dspark_patches.py \
  tests/patch/test_worker_dispatcher.py \
  tests/patch/test_runtime_callbacks.py \
  tests/runtime_patch/test_mla_target_ownership.py \
  -q -k 'not bf16_compressor_writes and not binds_every_constructed'
```

结果：**119 passed, 2 deselected**；另外，较新源码的接口契约检查为 **1 passed**。语法检查与 `git diff --check` 通过。

新增测试覆盖三个 compression ratios、batch 1/8、独立开关、幂等性、支持/不支持的 local heads、BF16 cache 分流、query 布局回退、`replay_start`、`adaptive_splits`，并从实际目标包源码提取签名检查补丁兼容性。也核对 `/home/zhoujie6/vllm-official` 的较新接口。

完整相关回归中有两项环境失败，均已在迁移前 HEAD 的独立快照复现：

- `test_bf16_compressor_writes_finite_576_dim_row_at_last_rope_position`：`No CUDA GPUs are available`。
- `test_dspark_target_binds_every_constructed_decoder_mhc_instance`：DeepGEMM 无法读取设备 CU 信息，得到 `None`。

当前环境无可用 GPU，尚未执行以下硬件精度测试：

```bash
VLLM_PLUGINS=__disabled__ python tests/accuracy/check_dsv4_flashmla_sparse_gpu.py
```

该脚本比较 FlashMLA/AITER eager decode、修改长度和索引后的 CUDA Graph replay，以及 sparse prefill。正式性能验收还需在 BW1000 上使用相同模型、TP、KV cache 配置，对两个开关开启/关闭的精度与吞吐进行对照；原 PR 的性能数字不代表本次迁移结果。
