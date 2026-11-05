# Kimi-K3 TP8/PP2 + DSpark

AMD Kimi-K3 原 decoder 未接入辅助隐藏状态的 PP relay，末级也未创建 DSpark 共享的 target embedding，导致 PP + DSpark 无法完整初始化。本次补齐这两条路径，使完整模型可在 TP8/PP2 下使用普通 checkpoint 加载和 DSpark 推理；不引入 preload 功能。使用原生 EagleModelMixin 的辅助状态编号与 PP relay 协议；边界层只由发送 stage 捕获一次，末级按层序合并远端与本地状态。attn-res 捕获 prefix sum，保留 target 输出的 pre-norm 语义。

外部 causal_conv1d decode 后端需要物理 stride 为 1 的 state indices。block table 的列视图通过 clone 转换，包括被 PyTorch 判定为 contiguous 的单元素视图；原生 speculative verification 路径保持不变。

## 启动步骤

两节点各 8 张 HCU。安装兼容的 vLLM 0.28.1、当前分支插件和算子，编译插件扩展；确认两端实际导入当前工作区。若通过 PYTHONPATH 选择工作区，需在两端设置相同路径。

```bash
export PYTHONPATH=/path/to/vllm-plugin-das
export MODEL=/path/to/Kimi-K3-Channel-W4A8-INT4-W8A8-INT8
export DRAFT_MODEL=/path/to/Kimi-K3-DSpark
export RUN_DIR=/path/to/pp-dspark-run
export MASTER_ADDR=<node0-ip>
export MASTER_PORT=29562
export VLLM_HOST_IP=<this-node-ip>
export NCCL_SOCKET_IFNAME=<network-interface>
export GLOO_SOCKET_IFNAME=<network-interface>
export NCCL_IB_HCA=<rdma-devices>
export API_PORT=19442

# node 0
bash tools/run_kimi_k3_pp_dspark_server.sh 0
# node 1（另一个节点）
bash tools/run_kimi_k3_pp_dspark_server.sh 1
```

脚本使用普通 safetensors 加载 target/draft，7 个 speculative tokens、probabilistic draft sampling、block rejection sampling、eager、关闭 async scheduling、max-num-seqs=1、max-num-batched-tokens=1024、context=8192、FP8 E4M3 KV、BF16 Mamba cache 和 prefix caching。KDA 使用 legacy prefill 与 align cache；extra_buffer checkpoint 不支持 speculative decoding。

当前 DSpark loader 仅在末级加载草稿，而 fastsafetensors iterator 使用 WORLD 集合通信，因此该启动方式使用 safetensors。这里不增加 preload、IPC 权重缓存或新的 loader 逻辑。

## HumanEval 命令与精度

本轮验收使用 EvalScope 1.9.1 的完整 164 题 HumanEval、每题一个候选、temperature=0、max_tokens=4096、关闭 thinking、并发 1。准备 EvalScope 本地数据缓存和新的输出目录；`DATASET_DIR` 下包含 `datasets/opencompass_humaneval-77ca0cdd29b87a18e04a1a11686e6e1f/`，其中 Arrow 数据 SHA256 为 `f52943a6aa89b8d973a477910be07b0d41e31f6c3df276a61db6164910cb223d`。

```bash
export DATASET_DIR=/path/to/evalscope-cache
evalscope eval \
  --model "$MODEL" --model-id kimi-k3-pp2tp8-dspark-nonthinking \
  --eval-type openai_api --api-url http://127.0.0.1:19442/v1 --api-key EMPTY \
  --datasets humaneval \
  --dataset-args '{"humaneval":{"subset":"openai_humaneval"}}' \
  --dataset-dir "$DATASET_DIR" --eval-batch-size 1 --repeats 1 \
  --model-args '{"timeout":1800}' \
  --generation-config '{"temperature":0,"max_tokens":4096,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --work-dir /path/to/new-humaneval-output
```

DSpark 接受率使用评测前后 `/metrics` 中 `spec_decode_num_accepted_tokens_total` 与 `spec_decode_num_draft_tokens_total` 的差值之比，统计窗口内只运行该评测。平均接受长度为 `1 + accepted_tokens / num_drafts`；核对每个位置的接受数之和与 accepted_tokens 一致。

## 本轮完整模型实测结果

2026-11-04，基于 `v0.28.1-dev` 的 `807c0821`，实现提交 `812a7b59820a18579afb9872bbd163d27e9b00c5`。两节点共 16 张 gfx938 HCU，vLLM `0.28.1+das.77acaf6.dtk2604`，完整 target 93 层、draft 5 层；普通 safetensors 加载，采用上面的 TP8/PP2 配置。

| 指标 | 本轮结果 |
|---|---:|
| HumanEval pass@1（关闭 thinking） | **159/164 = 96.9512%** |
| API 错误 / reasoning 输出 / 长度截断 | 0 / 0 / 0 |
| DSpark draft 轮次 | 4307 |
| DSpark drafted tokens | 30149 |
| DSpark accepted tokens | 26320 |
| DSpark 接受率 | **87.2997%** |
| 平均接受长度（含 target token） | 7.1110 |

失败题：HumanEval/83, HumanEval/101, HumanEval/130, HumanEval/132, HumanEval/145。完整 164 题各生成一次，未筛选题目、未对失败题补采样。DSpark 分位置接受数之和等于总接受数，接受数随位置递减，drafted tokens 等于 draft 轮次 × 7，计数校验通过。

“介绍下故宫”实测返回连贯中文，正常 stop，532 completion tokens，reasoning tokens 为 0；该中文请求的接受率为 15.4622%，与 HumanEval 的统计窗口分开记录。

本轮结果目录：`task_dir/roof_line/support_tp8pp2_dspark/runs/dev93_humaneval_nonthinking_01/`；启动、定向与完整 contract 日志、源码快照、基线复测和 code review 记录位于相邻的 `dev_migration_01/`。结果来自本轮普通加载服务。

## 验证与限制

- PP relay review 修复：decoder 的两个非末级返回分支均保留全部入站辅助槽位，再合并本级捕获的槽位，沿用全局编号和 tensor 引用；中间级没有辅助层时也继续转发。PP2/PP3/PP4 回归覆盖普通 residual、attn-res、边界 tap、连续两个无本地 tap 的中间级，并验证与原生 PPHandler relay 重复转发兼容。新增 PP3/PP4 用例在修复前 4 项失败；修复后定向 CPU 测试 88 项全部通过。本节补充不代表对新修复重跑完整模型或验证 PP3+ 完整模型精度，上面的 HumanEval 成绩对应明确标注的实现提交。
- Patch coverage、变更 Python 语法、HCU 测试登记、nightly/quarantine matrix 与环境 lock 六项 CI 静态检查全部通过。
- 定向 CPU 测试 84 项通过，覆盖 PP 边界捕获、93 层辅助状态顺序、普通 residual/attn-res、末级 embedding、patch 幂等、非 PP 分派、Conv1D stride，以及已有 checkpoint/eager-break 和 worker dispatcher。
- 完整 CPU contract 首轮为 4724 passed、67 skipped、5 failed。worktree 容器路径导致的 setup provenance 失败在修正路径后独立复测通过。其余 4 项 BoltOps 公共 API 测试在未修改的基线 `807c0821` 上同样失败：容器 BoltOps `0.1.0+dtk2604.torch2110.2610081830.gc0f3c9` 缺少 paged-MQA 导出，且 W4A8 接口签名不同于基线要求；本次未修改这些测试或放宽接口校验。完整 Kimi 服务日志确认使用 AITER int8_w4a8 MoE 路径。
- 本轮启动配置限定 eager、单请求调度和同步 scheduling；不由该结果推断 graph、异步调度、多请求或 PP3+ 的完整模型精度。
