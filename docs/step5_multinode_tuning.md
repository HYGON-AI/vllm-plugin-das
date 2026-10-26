# Step5 四节点迁移、配置与性能记录

记录日期：2026-10-19。硬件为四节点、每节点四张 BW1300_LC（gfx938），总计16卡。
模型为 Step5-Channel-FP8-w8a8，channel FP8 W8A8 权重、FP8 E4M3 KV。

后续64.79ms Decode正式轮、对应原生集成与验证范围见
[`step5_decode_itl6479.md`](step5_decode_itl6479.md)。下文保留此前各轮的历史记录。

推荐保留 EP16/E22 参数表与以下分阶段配置：Prefill 使用 DP16/TP1/EP16，
Decode 使用 DP4/TP4/EP16。本轮 Decode 三参数候选未确认稳定加速，不提升为默认。

## 代码改动清单

本分支原有的7个优化提交与 HYGON `49beac0a` 上的重放版本逐条 range-diff 等价。
原基线 `f04d9f63` 与上游 squash `75aa1370` 的树完全相同；基线合并使用这一等价关系，
结果树与已验证的 `b4db68e3` 相同。保留分支历史，不强制重写原提交。
上游同步单独成一个合并提交，包含 HYGON 原有的其他模型、兼容性和测试更新。

以下是本次插件化迁移的具体归属。历史来源指原 vLLM 主体提交，不表示本次新写了这些算法。

| 历史来源 | 行为 | 当前插件文件 |
| --- | --- | --- |
| `684a73f` | 将 indexer q/k/z/w 行并入 QKVG FP8 投影；indexer 接收预投影列视图 | `vllm_hcu/models/step5_text.py`、`vllm_hcu/model_executor/layers/step4_dsa.py` |
| `7c779c1` | causal FP8 prefill 评分、无效 tile 跳过、resident top-k 打包 | `vllm_hcu/model_executor/layers/step4_dsa.py`、`step4_dsa_kernels.py` |
| `6a9ad96` | 先确认并触碰本步仍需使用的 pending 行，再分配新行，避免过早复用 | `vllm_hcu/model_executor/layers/step4_dsa.py` |
| `f10f778` | 在插件 checkpoint loader 内同步排列专家权重/scale 与 router weight/bias | `vllm_hcu/models/step5_text.py` |
| `12907f8` | 插值式 top-k 阈值搜索 | `vllm_hcu/model_executor/layers/step4_dsa_kernels.py` |

配套改动：

- `vllm_hcu/v1/hcu_dbo.py`、`hcu_model_runner_v2.py`、`worker.py`：
  DBO 切片和 host mirror 兼容干净的上游 `InputBatch`，保持插件自有 DSA 调度。
- `vllm_hcu/patch/worker/framework_opt/patch_mtp_prefill_tail.py`：
  MTP tail 路径保留动态 host mirror。
- `vllm_hcu/patch/worker/__init__.py`：兼容导入和注册。
- `vllm_hcu/model_executor/layers/fused_moe/ep_table.json`：补齐
  `E=22,N=3072,K=4096,topk=8`。EP16 每 rank22 个专家；旧表只含 EP8/E44 键，
  EP16 会使用通用 fallback。E44 配置保持不变。
- 删除 `patches/vllm/upstream/0001` 至 `0005` 的源补丁文件，并更新其 README。
  安装和启动不改写已安装的 vLLM 源码。
- 迁移测试位于 `tests/model_executor/test_step5_upstream_delta.py`、
  `tests/runtime_patch/test_step5_dbo_compat.py` 及既有 Step5/patch 测试中。
  cold-import fixture 与已审查的 GLM5Next 上游依赖对齐。

原有 shared-expert stream、双路 AG/MoE/RS pipeline、sorted-A、MTP tail、
direct combine、DBO 等7项优化保留。其中 pipeline、DBO、direct-sum 在此次最终 P/D
测试均关闭，不将它们计为本次实测收益。

## 配置差异

| 配置 | D5 同布局基线 / E22 同布局 | Prefill 推荐 | Decode 推荐 / 本轮候选 |
| --- | --- | --- | --- |
| TP / DP / EP | 4 / 4 / 16 | 1 / 16 / 16 | 4 / 4 / 16 |
| 每节点 local DP | 1 | 4 | 1 |
| C64 下每 DP 的 BS | 16 | 4 | 16 |
| 每 DP batched-token 上限 | 32768 | 8192 | 32768 |
| 全局 batched-token 上限 | 131072 | 131072 | 131072 |
| max-model-len | 1048576 | 70000 | 1048576 |
| GPU memory utilization | 0.85 | 0.875 | 0.85 |
| NCCL max channels | 自动 | 32 | 自动 |
| pipeline / DBO / direct-sum | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

均使用 `allgather_reducescatter`、严格 FA DSA、关闭 prefix cache、
`FULL_AND_PIECEWISE` 图模式、compile sizes `[16,64]`、capture max64、
`HCU_MTP_PREFILL_TAIL=1536`、shared-expert stream min4096。
Prefill 的提升是布局和配置组合的结果；原 TP4 自动模式也记录了32个 collective 通道，
不能把全部提升单独归到 NCCL 的一个开关。实际 `enable_sp=false`，TP4 本身不代表启用 SP。

### Decode 可选候选，默认关闭

[`ep_table_decode_candidate.json`](../benchmarks/step5/ep_table_decode_candidate.json)
仅改变 E22/M256 的三个数值：

| 参数 | 推荐表 | 候选 |
| --- | --- | --- |
| GEMM1 `num_warps` | 2 | 4 |
| GEMM2 `BLOCK_SIZE_N` | 128 | 64 |
| GEMM2 `num_warps` | 2 | 4 |

其他尺寸、大 Prefill 条目及 E44 全部相同。算子微测从0.35320ms降至0.33974ms，
约低3.81%；整模型筛选轮2642.64tps，但独立重启后三轮均值只有2618.18tps，
较2608.56tps基线高0.37%，样本标准差13.04tps（约0.50%）。
三轮分别为2630.66、2619.24、2604.65tps，最后一轮低于原基线。
基线只有一轮正式结果，不能据此给出跨重启的显著加速结论。

DP8/TP2/EP16 的正式轮为2354.04tps，下降9.76%，未采用。
基于高斯路由代理拟合的静态专家排列正式轮为2617.01tps，低于同批无排列筛选轮，未采用；
代理负载指标改善不等于真实模型收益，推荐启动配置不设置专家排列文件。

## 实测结果

P=Prefill，D=Decode。各阶段单独使用四节点16卡，最后两行的 P/D 采用不同并行布局，
不代表一个运行实例同时使用两种布局。Prefill 推荐结果来自此前三轮，本轮没有重复测试。

| 平台 / 精度 | 机器数 | 每机卡数 | 上下文长度 | Decode 总 BS | 每个 DP 的 BS | MTP 配置 | 接受率 | Prefill 吞吐（tps） | Prefill 达到 H200 基线 | Decode 吞吐（tps） | Decode 达到 H200 基线 | ITL（ms） |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BW1300_LC / FP8（D5，尚无E22） | 4 | 4 | 64K | 64 | P16 / D16 | MTP3 synthetic60% | 59.99% | 9,025.54 | 13.02% | 1,460.17 | 41.92% | 122.79 |
| BW1300_LC / FP8（E22，同DP4布局） | 4 | 4 | 64K | 64 | P16 / D16 | MTP3 synthetic60% | 59.83% | 25,169.65 | 36.32% | 2,608.56 | 74.89% | 68.64 |
| BW1300_LC / FP8（当前推荐） | 4 | 4 | 64K | 64 | P4 / D16 | MTP3 synthetic60% | 59.83% | 29,085.03 | 41.97% | 2,608.56 | 74.89% | 68.64 |
| BW1300_LC / FP8（Decode候选三轮） | 4 | 4 | 64K | 64 | P4 / D16 | MTP3 synthetic60% | 59.92% | 29,085.03 | 41.97% | 2,618.18 | 75.17% | 68.46 |

H200 分母为用户提供的双机16卡 Prefill69306、Decode3483tps，未重新测量 H200。
H200 精度、MTP 接受率和 ITL 的完整设置未提供，百分比仅按给定吞吐计算。
当前16卡结果不验证单机8卡24700tps目标，也没有新增8卡/C32弱扩展测试。

## 测试口径与校验

- 只测 C64；输入65536，Prefill输出1，Decode输出4096。seed42、随机token IDs、
  temperature0、ignore_eos、返回token IDs。benchmark源码 SHA256：
  `d29320570bf0f70406b249444452ad47607b6038ef80ec7e7e8c0bd37586da9a`。
- MTP3 运行真实 draft，拒绝采样为 synthetic，位置参数
  `[0.838992,0.598322,0.362686]`，目标60%。接受率为整轮服务计数器
  accepted/drafted，包含脚本内短预热；多轮按 draft token数加权。
  这不是业务文本的真实接受率或完整模型质量验证。
- Decode TPS 使用原脚本公共 Decode 窗口内 token数/窗口时长。
  ITL 为该窗口内相邻流式 chunk 的平均间隔，MTP 一个 chunk 可含多个token。
  候选等效 TPOT=`64/2618.184735*1000`=24.44ms，是推导值。
- 最终候选预热一整轮，之后正式三轮，全部64/64成功、零抢占。
  16个rank的FA、源码、表一致，正式窗口没有编译事件或相应缓存更新。
  DSA实际 KV 绑定后的图宽为8704；早期512日志来自dummy profiling KV。
- 推荐 Prefill 三轮为29094.33、29086.89、29073.87tps，均值29085.03，
  标准差10.35。完整预热轮和一次含两次MoE编译的冷轮不计入均值。
- 候选 MoE 表通过15个尺寸与 fallback 的数值比较，覆盖表选择边界，
  最大相对L2约3.57e-6、最大绝对误差0.0001221；4个尺寸清零输出后做graph replay检查。
  历史迁移验证覆盖389个原回归用例（分批），及DBO/MTP重点检查。

聚合数据和逐轮值见 [`results-20261019.json`](../benchmarks/step5/results-20261019.json)。

## 复现启动配置

[`launch_multinode.sh`](../benchmarks/step5/launch_multinode.sh) 是去除节点私有路径后、
参数化的启动配置。它是 synthetic 接受率的压测入口。
依赖实测匹配的 Python3.12/DTK2610 镜像与 FlashAttention
`2.8.4+dtk2610.torch2110.2610152108.g6f61e1`；使用镜像匹配的 CP312 `hcu_ops` 扩展。
源码树运行前将匹配扩展安装或链接到 `vllm_hcu/`，不复制 CP310 扩展。
权重、tokenizer 和原始客户端是外部运行资产，不纳入源码提交。

每节点分别设置自身IP、数据盘目录和相同的头节点IP，然后以节点rank0/1/2/3运行。
两阶段分别运行，停止前一阶段后再启动后一阶段；下面IP占位值需要替换：

```bash
export STEP5_HEAD_IP='HEAD_NODE_IP'
export VLLM_HOST_IP='THIS_NODE_IP'
export STEP5_TOKENIZER=/path/to/benchmark-tokenizer
export STEP5_STATE_DIR=/var/lib/step5-benchmark  # 必须由数据盘支撑
export STEP5_SOURCE_DIR=/path/to/this/repository
bash benchmarks/step5/launch_multinode.sh prefill 0
# Decode阶段单独启动：
bash benchmarks/step5/launch_multinode.sh decode 0
```

运行候选时显式设置，未设置时始终使用推荐默认表：

```bash
export STEP5_EP_TABLE="$STEP5_SOURCE_DIR/benchmarks/step5/ep_table_decode_candidate.json"
bash benchmarks/step5/launch_multinode.sh decode 0
```

`STEP5_DRY_RUN=1` 仅打印实际命令，不启动模型或创建缓存目录。
复测使用新结果目录、完整预热及至少三轮正式测量，保留原始样本和编译审计。
代码、模型、缓存、日志均放在数据盘，不向节点家目录写入大文件。
