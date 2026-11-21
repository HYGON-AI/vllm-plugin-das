# SWA Layer Split + Decoder Bounded Replay（2026-11-12）

## 当前状态与范围

已实现候选代码及 CPU 回归，**尚未启动本轮 GPU 服务**。用户会先预约显卡。
改动位于本项目 `vllm-plugin-das/adapter_extension`；官方 Core 和 Adapter 源码未改。
旧的主 KV Layer Split 之外，现在也为 SWA 分配 owner，并保留现有 decoder bounded
replay 的两阶段执行。此处不是 ROCm 尚未支持的 prefix-cache SWA replay。

## 实现

- 主 KV / indexer 沿用源层列表的 owner；SWA owner 为 `layer_id % PCP_size`。
  40 层、PCP8 时，每个 Rank 长期保存 5 层 SWA。SWA 和主 KV 同层的 owner 可以不同。
- 保留 Core 的逻辑缓存组、block ID 生命周期和容量核算。每组在各 Rank 上放置
  owner 的持久数据，加一个由该组远端层复用的接收区。通信预算仍为每物理块 256 字节。
- 非 owner 在独立的 metadata 副本中把 `cache_slot_mapping` 设为 -1，保留原生 PCP
  all-gather、query 准备、压缩器和 indexer 顺序；不修改共享 metadata，不改变 query slots。
- 普通 fused preparation 和 decoder replay 的 `forward_kv` 全批写入都经过同一 owner
  规则。回放继续使用原生 `ReplayBatch` / `override_forward_context`，退出后恢复外层 context。
- 每层读取 SWA 前进行 owner 广播，零 query Rank 也参与。Decode 根据实际 paged indices/lens
  选择页（支持原生 `[tokens, 1, window]`）；prefill 根据原生 `seq_len/gather_len`
  选择读取区间，包含 PCP / replay 元数据已经计算的窗口边界。
- 不广播整个历史 block table：过期前缀、填充尾部和 null block 不得覆盖其他缓存组。
- Mooncake 也按 SWA owner 发送持久页；协议升级到 `hcu-v41-pcp-owners-v2`，避免旧版
  rank-0 SWA 发送规则与新版混用。P/D 必须同版重启；router 的 `/query` 拓扑格式不变。

主要文件：

- `adapter_extension/src/hcu_vllm_das/dsa_cache_layer_split.py`
- `adapter_extension/src/hcu_vllm_das/dsa_cache_layout.py`
- `adapter_extension/src/hcu_vllm_das/v41_pcp.py`
- `adapter_extension/src/hcu_vllm_das/mooncake_layer_split.py`

## CPU 容量计算（不是 GPU 实测）

使用之前 FP8 实机记录的 9 组真实缓存 schema、相同 10 GiB 预算、max-model-len 16384、
max-in-flight-tokens 16384，由安装的 vLLM 0.33 规划器计算：

| 配置 | 每物理块字节数（含通信预算） | 块数 | 框架计算的 Token 容量 |
| --- | ---: | ---: | ---: |
| 关闭 Layer Split | 116928 | 91829 | 390989 |
| 历史版本：只分主 KV | 114304 | 93937 | 399964 |
| 当前候选：主 KV + SWA | 98560 | 108942 | 463852 |

当前公共块跨度由未分层的 compressor state 组决定。不能把本表当作 43.63 GiB GPU
实测，也不能按某一组的空间降幅直接宣称整个模型收益。GPU 激活峰值、可用缓存预算和
新增通信耗时仍待测。

## 验证与证据

本次日志：项目根目录 `logs/swa_layer_split_20261112/`。
原始文件备份：该目录的 `before/`。历史失败日志保留，不覆盖。

- `cpu_v2.log`：布局、SWA、Mooncake 回归 105 项通过。
- `extra.log`：追加原生 fused preparation 的 owner/non-owner/off 路径及旧 PD 协议拒绝，4 项通过。
- `capacity.json`：上述 CPU 规划结果，确认未初始化 GPU。
- `adapter.log`、`pcp_replay_final.json`：最终 Adapter seam 与原生 PCP source replay 的结果。
  PCP 发布测试现在明确关闭独立的 Host/FP4 选择器，验证 PCP 自己的 off 路径；不改生产选择策略。
  Adapter seam 4 项通过，PCP source replay 13/13 通过；合计 113 项 pytest 和 13 项 source replay。
  测试脚本的 Bash/Python 语法及 Replay 开/关两种命令生成也已检查，未调用起服入口。

覆盖真实 Gloo 双进程、无本地 query 的 owner、复用接收区、跨组字节保护、滑动窗口与回放读取
区间、原生 decoder replay 全批写入/回放两阶段、真实 ZMQ/HTTP 和 CPU 内存复制模拟 PD。
HIP 算子和 RDMA 引擎仍是测试替身，CPU 通过不代表 GPU 数值或性能已通过。

复现 CPU（在现有容器，使用已安装 Core wheel）：

```bash
cd /workspace/deepseekv4.1-flash/vllm033_adapter
VLLM_TEST_PYTHON=/tmp/vllm033_validation_20261110/.venv/bin/python \
  bash vllm-plugin-das/tests/vllm033/run_swa_layer_split_cpu.sh \
  logs/swa_layer_split_cpu_repeat
```

## 待预约显卡后进行

先做单 P FP8 / PCP8 / eager 四种组合。保留原输入 128/4096/12288、每条输出 16、两轮预热，
不改变比较条件。测试脚本新增 `--swa-bounded-replay`，默认仍关闭；开启组核验 8 Rank 的
主 KV/SWA owner 日志，Replay 开关核验实际生效日志，并记录到结果。

```bash
# 仅在用户预约并确认节点、两次检查八卡显存/利用率/真实进程空闲后运行。
cd /workspace/deepseekv4.1-flash/vllm033_adapter
bash accuracy/test_fp8_layer_split.sh --mode both
bash accuracy/test_fp8_layer_split.sh --mode both --swa-bounded-replay
```

现有脚本的占用阈值不能替代宿主机真实进程检查。当前没有运行这些命令，也没有设置自动启动。
核验相同显存预算下实际 `GPU KV cache size`、greedy token/logprob 对照、耗时和报错。
之后才验证 PD / HumanEval / 实际 Mooncake 传输；不复用旧的正在等待的 run 计划或旧 connector 哈希。
新增的逐层通信可能影响性能，最终结论必须以 GPU 数据为准。
