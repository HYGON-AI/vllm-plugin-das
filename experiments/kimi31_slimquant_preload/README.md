# Kimi K3.1 Pro W4A8 TP16 preload 进展

## 目标与环境

- 模型：Kimi-K3.1-Pro Channel INT4 W4A8；环境：双节点 TP16。实际模型目录、节点地址、Python 环境和网卡由执行者提供。
- 参考 `task_dir/support_kimi31_pro/full93/gpu_rdma_candidate` 的双节点 TP16、fastsafetensors GPU RDMA preload 方案。vLLM 版本 `9d6ac73f3d`，adapter 分支 `adapter-kimi31-preload-opt`。
- 专家中间维度 3072，TP16 每 rank 192，**不做 padding**。实验配置将 `min_moe_intermediate_per_partition` 设为 192，避开原构造器默认的 256。

## 已实现

- BoltOPs 新增 K3.1 DSA 参考组件和量化接口，包括 indexer、top-k、逻辑到物理页映射、稀疏 MLA 包装、W8A8 linear 及 W4A8 expert 权重准备；容器内已 editable 安装。
- plugin 新增 checkpoint 元数据检查、量化策略、无 padding 的 TP 切分、W8A8 linear 和 W4A8 expert 的仅加载方法，以及混合精度 KDA 分离加载。KDA 的 q/k/v 为 W8，g/f_a/b 为 BF16；DSA 的 q_a/kv_a 为 W8，indexer 为 BF16。
- 新增实验性模型和扩展注册，使用独立架构名及测试专用入口，未加入正式生产入口。当前实现只用于权重加载，forward 会拒绝执行。
- 已生成四层、两个专家的独立诊断模型，包含 342 个 tensor；6 个 gate tensor 截取前两个专家，其余所选 tensor 按原字节复制并校验 SHA-256。路径：`task_dir/support_kimi31_pro/quant_w4a8_tp16/fixture_4layer_2expert_v1`。它不是完整模型。

## 已验证

- 原始 checkpoint 的 96 个 shard、498079 个 tensor 的索引及元数据检查无问题，包括 247296 个专家 packed I8 tensor 和 739 个其他 I8 权重及其 scale；此项不验证全部权重数据。
- 一个真实专家的权重按 16 个 rank 无 padding 切分，逐字节校验通过；真实专家的 vLLM loader 及第 0 层 KDA 的 16 rank loader 检查通过。
- GPU 单专家 W4A8 宽度 192 与量化参考结果一致，不代表完整模型推理通过。
- 完整模型 meta 构造通过：93 个文本层、92 个 MoE 层、27 个视觉层；未加载完整权重。
- 四层、两个专家的 CPU 局部真实权重加载中，27 个原始 tensor 的 KDA、expert、DSA W8 和 BF16 indexer 结果逐字节校验通过。两节点的扩展注册及 spawn 子进程检查通过。
- plugin 量化相关测试 25 项通过。证据和诊断脚本位于 `task_dir/support_kimi31_pro/quant_w4a8_tp16`。

## 未完成与后续步骤

- 尚未完成目标模型的完整 TP16 preload，也未验证 GPU RDMA 权重传输、严格权重跟踪、IPC consumer、重复连接、完整权重值或端到端推理性能。
- 两节点 GPU 正由其他任务使用，已停止 GPU 相关操作。
- 空卡后检查两节点显存、端口和网卡，再使用 `quant_w4a8_tp16/run_preload_fixture.sh` 测试双节点诊断模型；检查每 rank 的量化权重和 scale，并补充 IPC 审计及重复 consumer 验证。
- 诊断通过后评估完整模型显存容量，使用独立 socket 和端口运行完整 checkpoint preload。原 full93 审计针对 MXFP4，不能作为该量化模型的验证依据。

## 本地提交

- BoltOPs：`80c89aa`，实验性 K3.1 量化和 DSA 辅助代码。
- plugin：`f15c252`，实验性 K3.1 W4A8 TP16 preload 代码和诊断脚本。

## Adapter dependency

The optional activation candidate requires the companion Adapter module
`hcu_adapter.bridges.vllm_kimi31_preload` with `selected_mode()` and
`require_healthy()`. The MiniMax squash Adapter branch does not ship that bridge;
activation there is explicitly refused before registration. A registry unit test
with a bridge double does not establish real preload support on that branch.

## 可移植的诊断入口

源 checkpoint 和输出目录必须由执行者指定，不提供机器专属默认路径：

```bash
python experiments/kimi31_slimquant_preload/plan_small_fixture.py \
  --source-model "$MODEL" --output "$RUN_DIR/fixture-plan.json"
python experiments/kimi31_slimquant_preload/materialize_small_fixture.py \
  --source-model "$MODEL" --output-dir "$RUN_DIR/fixture"
```

其余历史探针读取已有的 `KIMI_K31_CHECKPOINT` 输入，未设置时明确失败。
`run_extension_probe_node.sh OUTPUT_JSON [PYTHON]` 直接运行仓内探针。
`run_preload_fixture.sh NODE_RANK MASTER_ADDR MODEL_DIR ADAPTER_ROOT PYTHON SOCKET_DIR`
接受 0/1 节点 rank、地址、模型、配套 Adapter、解释器和 socket 目录；NIC/HCA、
RDMA 库和其他集群设置由调用环境提供，不覆盖为某台机器的值。以上诊断不代表
完整模型精度、完整双节点 preload 或性能验收通过。
