# MiniMax 3.1 DP8 decode 与独立 DSpark preload（实验能力）

本能力以 adapter `9cc746522af756a36fb46764cf57cb9f0bc37602`、plugin `b3637997ed2405f93a5d16da41ba33a81da6958d` 为基线，面向 vLLM 0.33 / Torch 2.13 / gfx938。它通过 adapter extension `minimax31_decode` 显式启用。现有 PCP8 prefill 能力仍使用自己的分区、入口和缓存。

## 当前实现

- Target：TP1 / PP1 / DP8 / PCP1 / DCP1 / EP8，线性 128 专家，每 rank 16 个专家。复用既有 mixed SlimQuant W4A8/W8A8 checkpoint、HIPC pack 与 dense hipBLASLt 路径。
- MoE：Core MoERunner/router/shared-expert 接口 → DeepEP low-latency INT8 dispatch → GPU live counts → 两次 masked HIPC GEMM / SwiGLU-OAI INT8 → weighted combine。一个进程内的串行层共用有明确 stream 所有权的 transport/workspace。空 DP rank 仍参加 EP 通信。
- Draft：独立 5 层 MiniMax DSpark，不发布 checkpoint 不拥有的 embedding/lm_head 占位存储，Core 在 consumer 中绑定真实 target 模块。复用 Core DFlash/DSpark proposal、Markov 和 verifier 接口；保留 Gemma norm、SwiGLU-OAI、部分 RoPE、4096 sliding window、5 个 target auxiliary layer features、BF16 fc/Markov exclusions 和 channel INT8 linear。
- Preload：复用 Core daemon、native cache key、IPC handles/meta restoration/watch。增加 DP8 role/rank/GPU/weight identity 与 generation 检查；strict zero-copy，无磁盘 fallback，无原地 reload。
- 支持一次加载 target 后单独加载/重试 draft；消费者只重启自己。合并 ready 清单保存两组各自 generation。复用 target 时检查同容器 PID 出生身份，并在 draft 父进程运行期间监控 target owners。
- target 的 router alias、TopK/MSA scratch 与 completion counters 在 consumer 本地恢复；draft INT8 processed weights 不再次转置。外部权重预算显式计入两组 owners。

Core 的 DSpark 配置处理目前有两处封闭架构判断，会把未列出的 MiniMax draft 改写为 MTP/DeepSeek。实验能力通过 `minimax31_dspark_admission.py` 在内存中仅增加这个显式 draft 的配置准入；两处原函数的源码 SHA256 必须完全匹配，保留函数对象和已有别名，其他架构继续执行原逻辑。官方源码和 site-packages 不写回。升级 Core 时需要重新审查，不能直接更新摘要放行。

正式性能默认值、回退和回放边界见 [MiniMax 配置](minimax31_configuration.md#minimax-performance-defaults)。

## 启用与预载

正式解释器需先激活公共 general plugin，再创建 EngineArgs/parser，避免平台健康检查早于 owner 激活。独立预载入口已经完成这个顺序。

```sh
export HCU_ADAPTER_HOST=vllm
export VLLM_PLUGINS=hcu_adapter
export VLLM_USE_V2_MODEL_RUNNER=1
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=minimax31_slimquant,minimax31_decode
python -m hcu_vllm_das.parallel.minimax31_decode_preload \
  --model TARGET_CONFIG --quantization slimquant_w4a8 --trust-remote-code \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --prefill-context-parallel-size 1 --decode-context-parallel-size 1 \
  --enable-expert-parallel --expert-placement-strategy linear \
  --load-format safetensors --weight-cache-socket-dir /tmp/m31d20 \
  --decode-ready-manifest NEW_TARGET_READY_JSON [其他已验证 EngineArgs]
```

Target config 顶层需要 `hcu_minimax_decode` 标记及 `hcu_minimax_decode_weight_identity`。身份记录 native IPC 方法源码、Torch storage ABI、权重处理源字节、精度/布局和分区。示例生成器保存在任务目录，不能把未知配置自动当作兼容身份。

添加 DSpark 时，`--speculative-config` 指向独立 draft，包含 `method=dspark`、`num_speculative_tokens=7`、`draft_tensor_parallel_size=1`、`quantization=minimax31_dspark_int8`、`attention_backend=TRITON_ATTN` 及显式 `draft_load_config`。draft_load_config 的 load_format 必须为 `minimax31_decode_ipc`，extra config 包含 `is_draft=true`、`mode=zero_copy`、`fallback=false`、短 socket_dir、合并 ready 清单路径。

已有 target owners 时，再加：

```sh
--decode-reuse-target-manifest EXISTING_TARGET_READY_JSON \
--decode-ready-manifest NEW_DUAL_READY_JSON
```

这次只创建 8 个 draft owners，不再读盘/pack target。没有复用参数时可一次创建两组 owners。新清单必须尚不存在；长 Unix socket 路径、远程 seed、错误角色或拓扑会被拒绝。

消费者 target 也使用 `--load-format minimax31_decode_ipc`，extra config 为 `{"socket_dir":"/tmp/m31d20","mode":"zero_copy","fallback":false,"decode_ready_manifest":"..."}`。重启消费者要保留 owners；不能在同一 consumer 上调用 reload_weights。日志 `MINIMAX_DECODE_IPC_HIT` 记录各 role/rank 的真实命中和 generation。

## 验证与边界

2026-11-17 的实验记录位于 node20：
`/public/home/xiesl/inference/verify_torch213_image/task_dir/support_minmax_31_pcp8_ep8_perf/decode_implementation_20261117`。

已完成的检查：

- CPU 分区合同、错 role/rank/GPU/generation/PCP 拒绝、双 role 外部内存预算；新增源码准入、owner 存活、无 KV 量化和共享绑定回归共 11 个测试通过（最终隔离 wheel 环境）。
- 单卡 masked HIPC 独立数值参考和变输入/变 live counts graph。
- 八卡 DeepEP 数值参考：不同全局专家具有不同函数、非均匀路由权重、变输入/变路由 graph、一个 DP rank 为空。
- DSpark 配置准入的真实 Core 函数/缓存别名、非 draft 的原 MTP 语义、源码漂移拒绝。
- 完整 60 层 target DP8 ordinary producer、strict IPC consumer、保留 owners 的 consumer 重启和真实中文请求。
- 实际 target/draft owner socket 上错 generation、role、GPU 和原地 reload 共 8 次均在导出 handles 前拒绝。
- 8 target + 8 draft ready：每 rank target 发布 41,664,331,712 bytes；draft v2 发布 2,384,916,052 bytes。移除不拥有的 embedding/lm_head 占位存储后，draft 比 v1 的 7,301,688,916 bytes 每 rank 减少 4,916,772,864 bytes（8 rank 共 39,334,182,912 bytes）。该数值包含模型导出的独立 storage，不等于 KV、所有运行 workspace 或两组间物理去重后的预算。

完整 DSpark 三条真实请求已通过：中文介绍、17×19=323、2+2=4，累计 24 次 drafting、168 个 draft tokens、11 个 accepted tokens。该小样本只证明真实 proposal/verify 链路运行，不是吞吐 benchmark，不能与 Excel 的模拟接受长度 3.6 对比。

已知精度前置项：同一北京 prompt 的非推测回答为“融合了三千余年”，推测回答为“融合三千余年”。首处分叉为第 8 个输出 token；非推测所选“了” logprob=-0.934597，推测所选“三千”=-0.842591。固定相同前缀再做完整 prefill 的单 token 请求同样选择“三千”（logprob=-0.834088，“了”=-1.584088）。这使差异更指向 target 的 incremental q1 与 block/prefill 执行路径；仍不能据此确定具体 kernel 或豁免精度门禁。正式 q1/q8 分布参考、普通加载与 IPC 对拍、模型级 graph、完整精度和 BS128 匹配性能尚未完成。

组件 graph 通过不能表述为整模型 graph 通过；startup ready 不能表述为请求或精度通过。

模型级 graph 尚未验收，当前显式要求 `--enforce-eager`。当前 max_dispatch_tokens 默认 256；超过容量明确拒绝。静态 136 expert placement、LoRA、DBO、跨节点和原地权重更新尚未接入。当前能力没有实现 SGLang 的 host-backed per-head HiSparse：147K 独立历史的 BS128 不能用缩短上下文或共享前缀冒充对齐。原始 Excel 的 31.071ms 与所给 trace 也不是同一个配置，应分别保留证据。
