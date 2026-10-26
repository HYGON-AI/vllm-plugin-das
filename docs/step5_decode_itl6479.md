# Step5 Decode 64.79ms：历史成绩与原生迁移验证

2026-10-19，p5c1–p5c4，四节点各四张 BW1300_LC。DP4/TP4/EP16，
Channel FP8 W8A8、FP8 E4M3 KV，输入65536、输出4096、并发64，
每DP 16个请求。MTP3，synthetic拒绝采样参数
`[0.838992,0.598322,0.362686]`，保持严格FA DSA。

| 配置 | Decode tokens/s | ITL均值 ms | ITL P99 ms |
| --- | ---: | ---: | ---: |
| 本轮重测基线 | 2628.4730 | 68.3991 | 73.5808 |
| 真实专家重排＋打包AG＋路由/共享专家配置 | 2763.5631 | 64.7934 | 69.3880 |

候选完成一整轮预热和一轮正式测量，64/64完成，零抢占；正式窗口没有编译或缓存更新。
接受率为服务计数器accepted/drafted=59.8287%，为synthetic压测设置下的值。
公共Decode窗口91.491670秒、252843个token、90311个ITL样本。
ITL是相邻流式chunk的间隔；等效TPOT为23.1585ms，二者口径不同。
结果尚未达到ITL<60ms，也没有完成三轮稳定性验证。

## 已纳入插件的代码

- `vllm_hcu/distributed/device_communicators/packed_all_gather.py`：
  FP8激活、FP32路由权重、整数专家ID、FP32 scale按原始字节打包。
  四个已分组的AllGather合成一次，随后解包，数值与dtype保持一致。
  仅支持本次uniform M64对应布局；其他布局保留原路径。
  非连续输入先本地整理，避免某些rank因stride不同进入不同collective。
- `patch_cuda_communicator.py`：沿用已有适配器注册，
  `VLLM_HCU_DP_PACKED_AG=1`显式启用，默认关闭；已有单节点IPC路径优先。
- `router_gemm.py`：`HCU_STEP5_ROUTER_SHARED_MAX_M=32`限制路由/共享激活融合范围，
  默认仍为64。共享专家沿用已有`HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=64`配置。
- 专家映射`expert_placement_decode_real_c64_ep16.pt`是外部运行资产，不纳入源码仓库。
  性能资料包将其放在`assets/`，shape为[88,352]，由本次真实Decode路由统计拟合。它与原先缺失的
  `expert_placement_prefill.pt`是不同文件。

## 原生启动

```bash
# 在xuwq_vllm_1017容器内，每节点分别运行rank0/1/2/3。
export STEP5_STATE_DIR=/var/lib/xuwq_step5_itl6479_native
export STEP5_TOKENIZER=/opt/xuwq_step5_new_20261018/tokenizer-bytelevel
export STEP5_DECODE_PLACEMENT=/opt/xuwq_step5_itl6479_native/assets/expert_placement_decode_real_c64_ep16.pt
bash benchmarks/step5/launch_decode_itl6479.sh 0
```

启动入口通过插件已有checkpoint loader加载上述PT映射，启用打包AG并设置融合上限。
四节点容器必须均能读取`STEP5_DECODE_PLACEMENT`指定的文件；未指定时脚本直接报错。
无需worker extension、RPC注入或另行安装优化库。权重、缓存和日志应放在数据盘支撑的目录。

**验证范围：64.79ms来自运行时扩展方案，不能作为原生版的成绩。**
`/data`恢复后，同日已完成相同C64、64K/4096、MTP3、FA DSA口径的复测：

| 实际执行版本 | Decode tokens/s | ITL均值 ms | ITL P99 ms |
| --- | ---: | ---: | ---: |
| 外挂历史方案复现 | 2774.0950 | 64.8157 | 70.3956 |
| Python基于`ae60f22`、缺少E22表的旧部署包 | 1571.6223 | 114.3302 | 120.7674 |
| 同一主体代码，生效表切为Git中的完整E22表 | 2785.1913 | 64.5636 | 69.5529 |

各正式轮均完成一轮完整预热，64/64完成、零抢占，四节点正式窗口无编译或缓存更新。
修复部署后的完整预热为65.0224ms；正式Decode为2785.1913tps，同一Decode实例的
Prefill为23718.3827tps。此轮没有加入下述共享激活修复，唯一变量是生效的MoE参数表。

退化根因已通过单变量复测确认：旧打包脚本在D5快照上只覆盖`f6fee53f..ae60f22`
之间的改动，但E22表在`f6fee53f`时已经存在，因此没有被覆盖进包。部署表实际只有
E44键，EP16落到了未调优的library配置。排除CRLF差异后，411个核心Git文件的唯一
内容差异就是`ep_table.json`。仓库中的完整表从一开始就有E22，故114ms不能表示
完整Git源码的性能。

`check_ep_table.py`现在在加载模型前核对生效表；`package_runtime.py`从指定Git提交
完整导出文件，并在解包时逐个核对Git对象，避免将不匹配的旧快照标为新提交。
PT仍外置。延后重排/捕图实验是在缺表环境下进行的，已经归档，不作为当前启动依赖。

这里的两种文件作用不同：`ep_table.json`是MoE算子的分块和warp等调优参数，
属于源码包应携带的配置；`expert_placement*.pt`是专家负载均衡映射，仍由使用者
通过`STEP5_DECODE_PLACEMENT`提供，完整源码导出不会将PT或模型权重收入包中。
启动前的表检查与Git文件校验分别验证生效配置和导出内容，不能互相替代。

原生调用链沿用插件自身的注册入口：通信适配器调用内置的打包AG实现，checkpoint
loader读取外置专家映射，`MoERunner`把路由器返回的共享激活显式交给`Step5MLP`。
最终服务不需要`itl60_extension.py`、`itl60_controls.py`或RPC热替换步骤。

2026-10-20代码复核新增：已检查的rank0主模型285个编译子图缓存键在两版间全部相同；
四节点16卡直接对照的打包AllGather及更换输入后Graph重放结果逐字节一致。
这证明了该通信实现可以内置，但不证明整模型运行路径或性能已经等价。
实际主体轨迹也确认执行了打包AG、EP优化核和FA稀疏注意力。参数表遗漏修复后已恢复
到参考性能，但ITL<60ms目标尚未达到。

本地还修复了内部路由路径丢弃共享激活的问题：`MoERunner`仅接收显式声明
`returns_fused_shared_activation`的gate第二返回值，并交给`Step5MLP`使用。
普通Linear的bias不作激活处理。该修复有3项CPU语义检查，以及实际GPU上
K4096/路由352/共享中间维384、M16/32/64/65的数值和更换输入后Graph重放检查；
后者相对非融合GPU实现的最大相对L2误差约0.02%。这验证的是局部激活传递，
不能作为完整模型精度验证或额外性能收益证明。CPU检查进一步经过真实的
`SharedExperts`调度封装及`MoERunner._apply_quant_method`，三项均通过，验证共享专家
在gate之后消费激活，以及连续调用后的缓存和输出清空。

完整Git导出的`b4137e47`随后在四节点正常启动：每节点844个Git文件逐个核对对象ID，
归档共850个文件（另含匹配的CP312扩展和tokenizer），默认E22表21个尺寸，
没有表路径临时覆盖、worker extension或RPC热替换。
在共享激活修复、RSG上限64、shared-stream阈值4096、NCCL min=max4的组合下，
完成一轮完整预热与一轮正式C64测量：

| 完整Git部署的候选组合 | Decode tokens/s | ITL均值 ms | ITL P99 ms |
| --- | ---: | ---: | ---: |
| 共享激活修复＋RSG64/串行共享专家/NCCL4 | 2709.9075 | 66.3994 | 71.5609 |

64/64成功、零抢占，四节点正式窗口无编译或缓存更新，接受率60.1953%。
同一Decode实例的Prefill为15469.8790tps；这不是DP16的Prefill专用成绩。
该组合慢于E22单变量修复的64.5636ms，因此不作为推荐配置，也没有达到60ms。
它同时改变了融合阈值、调度和通信配置，不能把差异单独归因于激活交接修复。
`itl6479`启动配置的RSG32、shared-stream阈值64及NCCL自动选择默认值保持不变。

共享激活缓存、提前finalize重叠及通信器热切换试验均不属于原64.79ms成绩。

本轮Decode实例的Prefill吞吐为23727.3937tps。此前DP16/TP1/EP16的
Prefill专用配置29085.03tps是另一实例的独立结果，不并入同一配置成绩。

## 原生专家网格候选，默认关闭

`step5_expert_grid.py`为E22、全局M256的连续Channel-FP8布局提供有界GEMM网格。
排序核同时输出每个本地专家的起始块和块数，每个专家保留4个M方向worker，
减少为未使用排序行预留的CTA，并以BM16减少填充计算。
两次GEMM仍使用FP32累加、原per-token/per-channel scale及BF16输出；
SwigluStep、路由权重、专家映射、FA DSA和MTP3均保持原路径。
其他尺寸、标量activation scale、非连续布局和显式`configs`调用保留通用实现。

算子筛选使用真实路由统计和合成FP8权重环，在单个GPU上顺序检查16个rank的
分布，覆盖4层、2个采样点及2张映射，共256个分布/映射场景。保留旧PT的
4/4-worker组合在这些场景的最慢rank耗时均值低约5.05%；这不是16卡同时测速。
新映射虽降低微测均值，但有个别退化，因此正式测试保留旧PT。
主体实现另通过28项CPU/meta回归，以及GPU上的M128/256/272、预量化/BF16输入、
int32/int64路由、零输入及更换输入后清空输出的Graph重放；检查范围内结果逐位一致。
这些算子检查不构成完整语言模型质量验证。

四节点正常启动提交`0fb5f652`，每节点846个Git文件核对对象ID，默认完整E22表。
RSG32、shared-stream阈值64、NCCL自动选择、外置旧PT及测试口径保持原配置。
完整预热一轮后，正式轮64/64完成、零抢占，四节点正式窗口零JIT、零缓存写入：

| 原生配置 | Decode tokens/s | ITL均值 ms | ITL P99 ms |
| --- | ---: | ---: | ---: |
| 完整E22表，原专家GEMM | 2785.1913 | 64.5636 | 69.5529 |
| 有界专家网格，保留原PT | 2821.7612 | 63.7657 | 69.8667 |

单轮ITL低约1.24%，吞吐高约1.31%，尚未完成三轮稳定性验证，仍未达到60ms。
完整预热为63.5433ms；正式接受率60.1953%。同一Decode实例的Prefill为
23667.0890tps，不能作为DP16专用Prefill结果。

候选开关仅在Decode启动时显式设置；未设置时继续使用原专家GEMM：

```bash
export STEP5_DECODE_EXPERT_GRID=1
bash benchmarks/step5/launch_decode_itl6479.sh 0
```

Prefill独立测试支持`STEP5_PREFILL_BATCHED_TOKENS=4096|8192|16384|32768`及
`STEP5_PREFILL_PIPE_MIN_TOKENS=0|16384|32768|65536|131072|262144`。
默认仍是8192和0。这些开关不改变Decode配置；流水候选仍需16-rank数值和整模型
验证，不能将已有代码注释里的旧版微测收益视为当前版本的成绩。

## 原生 DP16 Prefill 独立结果

2026-10-20，以提交`0fb5f652`的完整Git部署运行DP16/TP1/EP16，四节点各4卡，
C64、每DP4请求、输入65536/输出1、MTP3配置、FA DSA、FP8 KV、关闭前缀缓存。
batched-token上限8192，NCCL max channels32、pipeline关闭、无专家映射PT。
两轮完整预热排除后，三轮正式Prefill分别为29057.1713、29068.0756、29007.7270tps，
平均29044.3246tps，样本标准差32.1600tps。各轮64/64成功、零抢占；每轮结束后、
下一轮开始前独立检查四节点，测量窗口内均无JIT或缓存更新。

数据见`benchmarks/step5/results-prefill-native-20261020.json`。此成绩来自正常启动的
主体代码，未使用worker extension或RPC热替换。与此前独立实例的29085.03tps接近，
不是新的吞吐提升；更大batched-token上限的整模型比较仍在进行。

## 融合路由后的共享下投影重叠候选

`HCU_STEP5_FUSED_ROUTER_STREAM`默认关闭。显式启用时，内部路由器先返回共享激活，
主体在生产者流上取出并清空缓存，再让辅助流等待生产者、登记张量生命周期，只执行
共享下投影。独立的激活张量允许主流继续路由、DP dispatch及专家计算；消费共享输出
前等待辅助流。共享输出用独立状态跟踪，避免被串行路径重复计算或提前交给EP combine。
DBO、序列并行、PCP、禁用共享流及非默认EPLB等受保护场景保留原路径。

这是默认关闭的候选，正式整模型测试为64.6237ms、2784.0134tokens/s，未提高吞吐。
对应Decode启动设置为`STEP5_ROUTER_SHARED_MAX_M=64`与
`STEP5_DECODE_FUSED_ROUTER_STREAM=1`，仍通过插件自身入口启动。
`check_fused_router_stream.py`检查真实MoERunner/SharedExperts/Step5MLP交接：
路由专家测试替身会覆盖输入，检查下投影实际流、单次消费、缓存清空和更换输入后的
Graph重放。它验证局部调度，不能替代整模型质量和正式性能测试。

已完成14项CPU检查，及GPU上M16/32/64/65、K4096、路由352、共享中间维384的
实际交接检查。M64确认在辅助流执行，M65确认走原路径；36次零输入/更换输入的
Graph重放与同一路由融合核的串行执行逐位一致。另有6项保护条件及4项Linux启动
参数检查通过。详情见`results-fused-router-stream-check-20261020.json`；这些数据
没有整模型吞吐或ITL成绩。

Prefill另提供`STEP5_PREFILL_MOE_CHUNK=16384|32768|65536|131072|262144`，
默认32768。它设置既有`EP_FUSED_MOE_CHUNK`，控制DP汇集后单次专家调用的分块，
与每DP的`STEP5_PREFILL_BATCHED_TOKENS`含义不同。例如DP16、batched tokens8192
会汇集到131072行，默认拆成4次M32768专家调用；把调度批量设为16384后，默认拆成
8次M32768调用。Decode启动入口固定保持32768，避免继承Prefill实验的分块值。
8项Linux启动参数检查已通过；更大分块仍需空闲GPU数值/性能检查和完整C64测量，
此参数的加入不代表已经取得Prefill性能提升。

启动时序和执行标记随后补充修复：旧的模块级布尔值在导入后再设置开关时仍保持
关闭，已改为使用时读取，并通过14项CPU及36次GPU Graph重放检查。还确认HCU模块
日志器的有效级别为WARNING，INFO标记缺失不能证明路径未执行；先前因该标记核对
而取消的四节点实例没有性能成绩。新标记在实际发出辅助流下投影后直接打印，每进程
一次；单GPU验证确认标记存在且只出现一次。后续四节点实例也已核对16个执行标记，
完整测试结果如下。

维护结束后已取回该实例的完整测试结果。16个进程均确认实际执行辅助流下投影，
正式轮64/64完成、零抢占，四节点测量窗口内无JIT或缓存更新；ITL均值64.6237ms、
P99为68.5472ms，Decode为2784.0134tokens/s，接受率60.1953%。这是一个正式轮，
缓存核查在网络恢复后补做，尚无三轮稳定性结论。该配置不优于此前有界专家网格的
63.7657ms，因此保持默认关闭，也未达到60ms。原始汇总见
`results-fused-router-stream-native-20261020.json`。

131072行的空闲单GPU Prefill算子筛选已完成，两种合成路由分布共18个计时场景，
其中16个非基准比较和8个零输入/更换输入检查逐位一致。均匀分布下32768、65536、
131072分块耗时为10.5976、10.0849、10.2838ms；偏斜分布分别为30.1087、31.0777、
32.4068ms。所试128/256/512-program持久化配置在这两种分布中均更慢。
因此默认仍为32768，65536仅保留为整模型候选。计时不含DP通信，不是16卡并发结果，
也不是Prefill吞吐成绩。原始计时见`results-prefill-operator-20261020.json`。

## 分块输出保护与 Prefill workspace 候选

16K调度批量、32768专家分块的四节点实例可以正常启动，但每DP的实际KV容量仅
215015–246209token，未通过原C64/64K容量检查，未启动正式测速，不能记为提速结果。
通用DeepGEMM workspace仍按汇集后的262144行预留，未利用EP内部32768行分块上限。

本轮另用真实Python分块流程复现了输出覆盖：当输出与workspace13共用存储时，
后续GEMM1会覆盖前面已完成的分块。已让多分块调用保护整个输出范围，并按实际
字节区间区分重叠与同一底层分配中的不重叠视图。单分块Decode保留原分配路径。
4个修复前失败用例在修复后通过；扩展后的55项CPU/meta合约检查全部通过。
这些检查不代表完整模型质量或性能验证，历史测速也不能作为本修复后的实测成绩。

`STEP5_PREFILL_BOUNDED_WORKSPACE=1`为默认关闭的Prefill候选，只在实际支持的
E22 Channel-FP8/SWIGLUSTEP、普通DP/EP且无SP、LoRA、DBO的路径上按内部chunk
申请临时缓冲区，并为完整输出保留独立字节范围。Decode入口固定关闭此开关。
真实GPU检查已覆盖256/131072/131073/262144/262145行，使用每token各不重复的
8个专家，普通执行及15次更换输入后的Graph重放与独立输出参考逐位一致。
131072行workspace由17272143872降到5460983808字节，262144行由34452013056
降到6534725632字节；M256单分块Decode的输出与workspace大小保持一致。
检查调用真实DeepGemmExperts、原生modular runner与当前镜像的WorkspaceManager，
不替换线上worker方法。最初包含重复专家ID的校验输入不符合真实topk约束，该次
失败不计为通过；修正输入后在相同不可变主体代码上通过。详情见
`results-ep-workspace-check-20261020.json`。

四节点已正常启动提交`48945c1c`的16K候选，仍须取得实际KV容量和同口径正式成绩。
还要在修复后的相同源码上重测8192基准，才能判断批量调整的真实收益；目前没有新的
整模型Prefill吞吐或低于60ms的Decode结果，也没有完整模型质量验证结论。

该四节点实例随后正常就绪，但workspace执行标记为0、每DP容量仍只有228104–254569
token，因此在测速前被检查拒绝，未生成正式成绩。镜像的FP8 oracle实际返回
`TritonOrDeepGemmExperts`外层选择器；此前仅在内层`DeepGemmExperts`提供分块接口，
外层没有透传。已补为查询该选择器实际选中的实现，保留其他后端、未知派生类、
SP、DBO及关闭开关时的原路径。使用真实oracle返回的类复现了原先仍按全M申请的
问题；修复后包括这些选择保护的61项CPU检查通过。GPU检查也改为经过同一外层
选择器，旧的直接内层检查不能单独证明正式服务已启用此候选。

提交`04c59a4b`随后经过真实oracle返回的外层选择器完成相同5个GPU尺寸及15次
更换输入的Graph重放，结果逐位一致；Decode单分块输出和workspace大小保持一致。
原始记录见`results-ep-workspace-dispatch-20261020.json`。四节点新的
`prefill-b16384-p0-c32768-selected`实例已启动，正式测速仍以实际执行标记、KV容量
及逐轮缓存检查通过为前提；此处暂不提供修复后的整模型成绩。

该实例已完成正式测试：16个进程的workspace执行标记齐全，每DP实际KV容量为
547042–554016token，通过原0.9容量检查。两轮完整预热排除后，三轮C64、输入65536/
输出1的Prefill分别为27747.8772、27725.2865、27750.5837tokens/s，均值27741.2491，
样本标准差13.8901。三轮各64/64成功、零抢占，每轮结束后立即核查四节点，测量窗口
均无JIT及缓存更新。DP16/TP1/EP16、MTP3配置、FA DSA、FP8 KV、关闭前缀缓存，
batched tokens16384、专家chunk32768、pipeline0，无PT映射和WorkerExtension。

详见`results-prefill-selected-20261020.json`。这是修复后的完整服务实测，不能用旧
源码的8192成绩直接计算批量调整收益；同源码8192、专家分块及负载均衡比较仍待完成。
此Prefill成绩也不表示Decode已达到60ms以下。
