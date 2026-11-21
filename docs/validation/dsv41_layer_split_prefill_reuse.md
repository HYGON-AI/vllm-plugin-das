# PCP Layer Split：首次 prefill 的 KV 页号和接收区复用

使用 Adapter 提供的 vLLM 平台接口注册启动参数：

```bash
vllm serve <model> --enable-layer-split
```

这里仍需原服务的 PCP、模型和其他参数。未传此开关时默认关闭；
开启后，满足条件的 prefill 自动使用主 KV 和 SWA 优化。参数进入原生
`additional_config`，与同一命令中的静态 EPLB JSON 合并，传给引擎与 worker。
早期 A/B 所用的 Layer Split 环境变量和额外 trace 标记不再是运行接口。

首次纯文本 prefill 的页号来自 `SchedulerOutput`。每个 PCP rank 按相同的
请求长度和物理块表算出同一份页号，进入模型前把每组页号上传一次。
主 KV 和 Indexer 继续发送原来覆盖整批请求的有效页，不依赖 Top-K。
这条路径不再读取 GPU 块表、序列长度或跨 rank 汇总后的页 bitmap。

同一个源的主 KV 已在接收区且未被覆盖，后续层直接读它。源层重写时失效；
Indexer 广播占用同组接收区时，下一次主 KV 读取重新广播。复用状态随
当前 forward 结束清除，不跨请求步骤或 dSpark 草稿阶段。当前独立池布局
没有增加缓存分配。第一轮只优化主 KV 和 Indexer，SWA 仍走原路径。不符合条件的 batch
也走原路径，包括有前缀命中、
已缓存的分块 prefill、混合 decode 和 KV 加载。SWA bounded replay 仍走
原计算路径，其首次 prefill 的裁剪视图由原 batch 的全局页号覆盖。

旧 32K trace 在 0-based 第 2–39 层进入主 KV 广播路径 38 次。按当前主源
2/8/14/20 和后续四次重算 Indexer 的顺序，CPU 回放预计有效载荷广播 8 次。
第 16 层的 17.033 ms CPU 等待与前面 GPU 工作大部分重叠；旧 trace 在
stream 101 上观察到 1.157 ms 空档。次数和 CPU 等待时间不能换算成 TTFT。

## SWA 页号计划

在同一套首次纯文本 prefill 条件下，SWA 页号也按缓存组提前计算。
普通视图覆盖整段新输入；单请求 Bounded Replay 在切入层保留窗口之前的
读取范围，后续层只读取保留的尾段。选择依据是 Core 实际使用的 replay
forward context；请求返回 prompt logprobs 或 routed experts 时按 Core 的保留规则。

多请求在 replay 切入层的零查询行仍可能有历史读取范围，该层保留旧路径；
其余层复用各自视图的页号。缓存命中、续段 prefill、混合 decode、dSpark
草稿和外部 KV 加载仍走原路径。页号清单不改变每层 KV 内容、owner 或
广播次数，也不增加 KV 接收区。计划路径跳过 `_swa_live_block_ids`、
页 bitmap all-reduce 和 bitmap 读回。

历史 A/B 两组都启用主 KV 复用，只切换当时实验版的
`VLLM_HCU_DSV41_LAYER_SPLIT_SWA_PREFILL_PLAN`。
检查 32K prefill 的 SWA 同步读回与 bitmap 汇总、首次 step 时间、GPU
未记录活动的区间、实际 payload 和输出 token/logprob；优化组跑 HumanEval 20 题。
单独测多轮不启用 profiler 的 TTFT，不能把累计 CPU 等待直接当成 GPU 收益。

2026-11-20 nmz30 实测只运行 SWA 计划开启组，复用 11-19 已开启主 KV
复用的 32K trace 作基线。首个 prefill GPU step（八卡取最慢）从
1,221.4 ms 降到 1,168.3 ms，少 53.1 ms（4.3%）。Rank 0 的 SWA
同步读回 160 → 0 次，实际 SWA 广播仍是 40 次。输出 token/logprob
一致，优化组 HumanEval 20/20，两组 KV 容量均为 2,818,048 tokens。
这是单次 profile 对照，服务结束后八卡显存均回到 2.05 MiB，无 KFD 进程。

## 主 KV 历史设备验收

在 nmz30 上使用同一套 W4A8、PCP8、FP4 Indexer、原生 FlashMLA、SWA bounded
replay、dSpark(5)、静态 EPLB32 和自动 KV 预算，两组均开启 Layer Split。
使用相同源码、EPLB map、GPU UUID、预热输入、32K token ID 和 profiler
设置。用验收脚本的同一 `--stage on --replay 1 --expert-map …` 配置，分别生成
两个 profile，只在 `runtime_env` 中改变 `VLLM_HCU_DSV41_LAYER_SPLIT_PREFILL_REUSE`
（`0` 旧路径，`1` 新路径，默认开启）；每组使用不同输出目录，两组都开启
`VLLM_HCU_DSV41_LAYER_SPLIT_TRACE=1`。脚本会保存实际启动环境。
验收脚本会配置 profiler，采集时用 `--keep-service` 保留当前组服务，读取其
`capacity_on/input_32768.json`，发两次同样的 32K 请求预热；然后用
`POST /start_profile`、同一请求、`POST /stop_profile` 采集一次。确认文件
写完后只结束本组服务、复查显卡，再开始下一组。

- 在第一个 32K step 内分别统计 `dsv41.layersplit.payload.main`、`.index`、`.swa`。
  标记按逻辑广播计数，同一广播拆成几段不重复计数。记录第 16 层 D2H 读回、
  主 GPU stream 空档和整步 GPU 时间；单独用未开 profiler 的请求测首 token 延迟。
- 比较相同输入的生成 token ID 和对应 logprob，报告最大绝对差、HTTP 错误和截断；
  优化组运行相同的 HumanEval 前 20 题。记录实际每卡 KV 预算及容量并检查日志。
- 若主 KV 同步已经消除，按 trace 判断剩余空档来自 SWA、页打包还是通信。
  只有剩余广播仍在关键路径上，才尝试独立通信 stream；额外接收区需要另算显存成本。

## CPU 验证记录（2026-11-19）

nmz30 已安装 wheel `0.33.0+das.dtk2610.gfx936gfx938.g5afd482`，屏蔽 GPU 后
294 项 pytest 和 13 项 PCP 源码回放通过。覆盖页号与旧跨 rank 并集、零查询
rank、接收区被 Indexer 覆盖、SWA、Mooncake、dSpark、静态 EPLB。nmz30 的
主 KV A/B 已完成：32K 首次 prefill 的八卡最慢 GPU 时间为
1,316.5 → 1,221.4 ms，主 KV 广播 38 → 8 次，SWA 40 次不变。
输出 token/logprob 一致，优化组 HumanEval 20/20。此结果不包含新增 SWA 页号计划。
