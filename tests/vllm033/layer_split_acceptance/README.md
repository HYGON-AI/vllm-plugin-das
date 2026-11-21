# Layer Split 容量 → HumanEval 验收脚本

只测试当前单节点 PCP8 的前向/生成能力，不启动 PD 服务。默认只打印计划，
`--check` 只做 CPU 和数据集检查，只有显式 `--run --gpus ...` 才使用显卡。

本目录的 JSON 是部署示例。复制所需 profile，填写目标节点 IP、模型、EvalScope、
数据、tuning 和依赖清单路径，再通过 `--profile` 传入。
`shenzhen.json` 和 `s7.json` 分别提供带自定义算子工具链和使用镜像算子的 FP8 配置。
Adapter 和 Plugin 从工作区源码加载。预检核对 vLLM wheel、扩展入口、模型配置和算子哈希。

## 测试口径

| 参数 | 关闭组 | 开启组 |
|---|---|---|
| Layer Split | 不传启动参数 | `--enable-layer-split`，主 KV + SWA |
| FP4 Indexer | 开启 | 开启 |
| SWA bounded decode replay | 默认开启 | 同样开启 |
| KV 预算 | 默认每卡 24 GiB | 同样每卡 24 GiB |
| 并行 | TP1 / DP1 / PCP8 / EP8 | 完全相同 |
| 其他 | eager、block64、batch32768、model_len65536、max_seqs128 | 完全相同 |
| PC / dSpark / EPLB / KV transfer | 关闭 | 关闭 |

通过 `--kv-cache-memory-bytes` 固定 KV 预算，所以 KV 预算不再由
`gpu-memory-utilization=0.9` 自动推导。24 GiB 是可修改的测试默认值，
不是最大可分配容量，也不是已经实机验证可运行的承诺。上一轮深圳记录的可用
KV 约为 30.42 GiB；新代码和指定卡仍需实测显存需求。不要因 OOM 自动降低
输入长度、改变模型或切换算子来宣称通过。

gfx938 的原生 DeepSeek V4.1 eager PCP 默认选择 HCU Inverse RoPE 和 PCP Q RoPE。
起服脚本无需传入 `HCU_CONFIG` 或 RoPE 环境变量。启动日志
`DSV4.1 native PCP RoPE policy: inverse=hcu query=hcu` 记录两个实现的选择。

容量直接读取 Core 的 **GPU KV cache size: ... tokens**，不把主 KV 页数、
理论计算或八卡 Token 数求和当作实际容量。还检查固定预算日志、FP4 与 Replay
实际开启记录，以及开启组八个 Rank 的主 KV / SWA owner 和独立池日志。
预算取整和实际池分配可能产生少量余量，保留全部原始日志供复核。

顺序为：

1. 关闭组起服，固定输入 128/4096/12288/32768 Token，每种预热两次再测一次，
   固定输出 16 Token，保存真实 token IDs、logprobs、响应和容量，清理该服务。
2. 开启组重复同样检查，生成容量绝对增量、倍数和百分比；检查两组生成 Token
   是否一致。容量没有增长、证据缺失或生成 Token 不同则保存失败结果并停止。
   logprob 差值独立报告，不用未约定的容差宣称数值等价。
3. 两组容量与固定输入检查通过后，只在仍运行的开启组服务上测 HumanEval 前 32 题。
4. 清理开启组，结束测试；不再运行关闭组的精度测试。

这样完整流程只需两次起服。`--stage capacity` 只做前两步。
HumanEval 固定前 32 题（HumanEval/0～31）、temperature=0、并发16、输出上限2048、thinking关闭；
使用现有 EvalScope 官方 benchmark 的生成和评分方式，报告通过题数、pass@1、失败题和截断。
结果范围为开启组的 32 题子集。正式验收保存原始 predictions、reviews 和逐题
结果，并用隔离的评测环境保留同一批输出的 normalized 结果和历史失败题。
未规定质量分数阈值；执行完整性通过不等于模型精度验收通过。HTTP失败、漏题和截断仍检查。

## 运行方式（容器内）

W4A8 全功能联测使用 `nmz18_w4a8_dspark.json`、`--stage on` 和
`--expert-map "$EXPERT_MAP"`。提供与模型匹配的校准映射后，脚本加载
静态 EPLB32（`locality_fair`）、SWA Bounded Replay、dSpark(5)、MXFP4
Indexer，以及原生 FlashMLA、AITER mHC、PCP Q RoPE 和反向 RoPE 的配置。
KV 显存由 vLLM 起服时测算。验收记录包含显存预算、实际 Token 容量、
32K 输入结果和 HumanEval 前 32 题成绩。

```bash
cd /workspace/layer_split
SCRIPT=vllm-plugin-das/tests/vllm033/layer_split_acceptance/run.sh
PROFILE=/path/to/profile.json

# 只看完整计划；不会查询/使用 GPU。
bash "$SCRIPT" --profile "$PROFILE"

# CPU 准入、源码来源、私有算子哈希、模型文件和原始 32 题数据检查。
bash "$SCRIPT" --check --profile "$PROFILE"

# 预约并核对实际卡号后执行。下方0..7仅为示例，不代表当前空闲。
# 先只测容量：
bash "$SCRIPT" --run --profile "$PROFILE" --stage capacity --gpus 0,1,2,3,4,5,6,7

# 完整容量对比和开启组 HumanEval 32 题：
bash "$SCRIPT" --run --profile "$PROFILE" --stage all --gpus 0,1,2,3,4,5,6,7

# 预算、Replay 可调，但一次对比中的两组必须一致：
bash "$SCRIPT" --run --profile "$PROFILE" --gpus 0,1,2,3,4,5,6,7 --kv-cache-gib 24 --replay 1
```

显卡号按 hy-smi 的物理编号输入；脚本核对 PCI/KFD/ROCr UUID，明确绑定这八张卡，
不自动换卡。每次启动前复查两次利用率、显存和真实 KFD 进程；默认空闲基线每卡
不超过2 MiB、利用率0，未知进程归属也拒绝启动。资源忙时退出，不挂后台轮询，
不停止别人任务。要求 `kernel.numa_balancing=0`，不会自行修改宿主设置。

控制器持有工作区锁并使用新的结果目录；拒绝覆盖旧结果。
各服务/评测独立进程组，保存 PID、日志和退出码。失败、超时、SIGINT/SIGTERM
都清理自身进程，并核验显存释放；不会清理其他工作区或停止共享容器。
启动前检查不等于资源预约，运行中发生资源竞争仍可能失败，须保留该次记录。

用户要求等待资源并保留服务时，可显式加 `--wait-idle-interval 60 --keep-service`。
等待只发生在首次起服前，八卡显存、利用率和 KFD 客户端必须全部满足空闲条件；
之后仍由 `service()` 做两次启动前复查。等待期间源码或 GPU 身份改变则报错。
`--keep-service` 保留最终开启组服务；验证失败时也保留尚存活的服务供排障，
在 `retained_service.json` 和最终状态中记录 PID，不将其标记为已清理或验证通过。
成功的关闭组仍会结束以切换到开启组。默认不加这些选项时仍按原规则清理。

默认结果在 `/workspace/layer_split/acceptance_runs/layer_split_<时间>/`：

- `plan.json` / `preflight.json`：参数、源码提交/差异哈希、依赖和数据来源。
- `capacity_comparison.json`：实际 Token 容量变化。
- `smoke_comparison.json`：固定输入的生成 Token 与 logprob 对照。
- `capacity_off/`、`capacity_on/`：起服、输入、响应、GPU和评测证据。
- `summary.json` / `status.json`：最终结果。只有显式执行完才能产生实测结论。

本脚本并未证明四倍容量或任何模型精度；`--check` 和控制器单测仅证明脚本的
离线检查通过。checkpoint 记录 config/index 哈希和分片 stat，未全量哈希权重。

## CPU 自测

NMZ18 的 W4A8 环境使用 `--profile .../nmz18_w4a8.json`，模型为
`/llm-models/DeepSeek-V4.1-Flash-Channel-INT4-w4a8`。
该配置严格检查 CT INT4 权重/动态 INT8 激活，要求 CT W4A8 扩展及 DeepGEMM HIPC 算子，
显式传递 `--quantization compressed-tensors --dtype bfloat16`；FP4 Indexer、
SWA bounded replay 及 OFF→ON→ON-only HumanEval32 协议保持一致。
NMZ18 的 W4A8 实测在分配24GiB KV时只剩约19.6GiB显存，
此节点两组固定使用每卡12GiB KV预算，为32K输入和评测保留运行空间。
新节点必须先通过 `--check`；模型实际兼容性、显存峰值和精度须等待八卡实测。

NMZ18 的 dSpark 配置 `nmz18_w4a8_dspark.json` 默认使用 vLLM 启动时测出的
KV 预算。两组都传 `--gpu-memory-utilization 0.9`，各自从预算中扣除实测的模型
权重、激活峰值和其他运行开销。对比报告同时记录两组日志给出的可用 GiB 和
实际 Token 容量。需要按相同的 KV 字节数复现历史结果时，可显式传
`--kv-cache-gib 8`。

NMZ18 的宿主启动脚本 `launch_dspark_nmz18.sh` 使用 `--stage on`。仅启动 Layer Split
开启组，记录 vLLM 实际给出的 KV Token 容量，发一次恰好 32768 Token 的输入并验证
16 个输出 Token。通过后在同一个服务上运行 HumanEval 前 32 题。

```bash
HCU_ADAPTER_HOST=none VLLM_PLUGINS=__disabled__ \
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
python3 -B vllm-plugin-das/tests/vllm033/layer_split_acceptance/test_controller.py
```

覆盖不增容/证据缺失的拒绝、假空闲、重复卡号、漏题/HTTP失败/截断、分数与逐题结果不一致、
单组 32 题与起服顺序，以及容量失败时不启动精度测试和清理顺序。测试中的分数是构造数据，
不属于实际模型成绩。
