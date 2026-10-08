<h3 align="center">
vLLM HCU Plugin
</h3>

[English](README.md) | [简体中文](README.zh-CN.md)

---

`vllm-plugin-das` 为 [vLLM](https://github.com/vllm-project/vllm) 提供 HCU 平台、
模型和算子集成。请选择与所部署 HCU/OpenDAS vLLM 环境匹配的仓库分支。

## 版本兼容

包版本及集成基线记录在 [vllm_hcu/version.py](vllm_hcu/version.py) 中，
wheel 版本的生成逻辑位于 [setup.py](setup.py)。安装时应查看目标分支中的这些文件，
README 不重复维护会持续变化的版本号。

[兼容性检查](vllm_hcu/compatibility.py) 要求已安装 vLLM 包的 PEP 440 epoch 和
release tuple 与 `__vllm_target_version__` 一致，允许预发布、开发、后发布和本地构建
后缀不同。缺失依赖、无法解析的版本号或不同发布线会在补丁注册前被拒绝。

源码中记录的提交用于标识集成基线，并不将运行环境锁定到某一个 wheel。
通过版本检查不代表 PyTorch、DTK 和算子的二进制接口已经验证兼容，也不保证任意
上游 vLLM wheel 都可以使用。请使用相互匹配的 HCU/OpenDAS 运行环境。

## 上游、许可证与第三方声明

本仓库采用 Apache License 2.0，参见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
部分文件改编自 vLLM 及其他许可证兼容的第三方项目，来源与修改说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。修改方为海光信息技术股份有限公司。

第三方声明中的历史来源记录不等同于当前分支的运行时依赖要求，兼容要求以源码定义为准。

## 安装

请在 Linux HCU 环境中构建和运行，预先准备匹配的 DTK 工具链、PyTorch 和 HCU vLLM
基础包。Python 版本应满足 [setup.py](setup.py) 的要求；该脚本会导入 PyTorch 扩展
构建工具，不会自动配置完整的运行时依赖。

[docker/Dockerfile](docker/Dockerfile) 提供容器构建流程。AITER、FlashMLA、DeepGEMM、
LightOp、DeepEP 等算子包需要与所选模型、后端及运行时二进制接口匹配，不能简单地用
通用 PyPI 环境替代对应的 HCU 构建。

[CI 环境配置](.github/workflows/configs/hcu-runner-environment.json) 使用
`match: release_line` 检查配置中指定的 DTK 主、次版本发布线，允许该发布线内的数字
补丁版本以及 `-`、`+` 构建后缀，拒绝其他发布线和格式异常的版本。
预检报告保留完整的实际版本号。自定义配置未指定 `rocm.match` 时仍执行精确匹配。
其他依赖检查独立保留；放宽版本匹配不等于所有构建的二进制接口均已验证兼容。

准备好运行环境后，在所选分支的仓库根目录构建并安装：

```bash
python3 setup.py bdist_wheel
python3 -m pip install --no-deps dist/vllm_hcu-*.whl
```

请使用干净的构建目录；如果 `dist/` 中有多个 wheel，应明确选择要安装的文件。
`--no-deps` 用于保留已准备好的运行环境，不会安装或验证缺失的依赖。

### 可选构建配置

下面是按需使用的环境准备示例，并非每次安装都必须执行。仅在缺少构建工具时安装，
需要显式指定 DTK 位置时设置路径，并按机器资源调整编译并发数：

```bash
# 仅在环境尚未提供构建工具时安装。
python3 -m pip install ninja wheel setuptools
# 仅在需要显式指定 DTK 路径时设置，并替换为实际安装位置。
export ROCM_PATH=/opt/dtk
# 可选的编译并发限制，数值仅为示例。
export MAX_JOBS=16
```

`MAX_JOBS` 未设置时默认为主机 CPU 数量。设置 `ROCM_PATH` 后，构建脚本还会用它
查找 DTK 元数据并写入 wheel 版本后缀。

`ADD_GIT_VERSION=1` 为默认行为，会将可获取的 Git 提交信息加入本地版本后缀。
设置为 `0` 只省略 Git 提交信息，`+das` 和检测到的 DTK 元数据仍会保留。

安装后不需要执行源码打补丁步骤。安装及插件启动不会重写 vLLM 包文件，也不会创建
源码目录软链接。构建脚本通过只读 Git 和环境信息生成版本，不改写跟踪中的
`vllm_hcu/version.py`，也不修改 Git 全局 `safe.directory` 配置。运行时优先从已安装
发行包的元数据读取完整版本，没有安装包时才回退到源码版本。
扩展构建仍可能将生成的 `.so` 复制到当前仓库，因此建议使用干净的构建目录。

历史命令 `vllm-hcu-apply-patches` 仅执行只读兼容性检查，新的自动化流程应使用：

```bash
vllm-hcu-doctor
```

只检查元数据和源码、不注册平台补丁时，可使用 `vllm-hcu-doctor --no-arm --json`。
默认模式还检查当前进程的平台补丁注册状态，但两种模式都不能代替模型精度或设备执行验证。

## 运行配置

当前 [HCU Worker](vllm_hcu/v1/worker.py) 使用 `HcuGPUModelRunnerV2`，
不支持回退到旧版 runner。启动服务时应使用 Model Runner V2：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

以下配置的作用不同：

| 配置项 | 当前行为 |
| --- | --- |
| `VLLM_USE_V2_MODEL_RUNNER` | 必须选择 V2，HCU Worker 没有旧 runner 回退路径。 |
| `VLLM_HCU_USE_CUSTOM_OPS` | 默认启用；设为 `0` 关闭受总开关控制的可选 HCU 优化路径，不等于禁用整个插件或移除全部原生依赖。 |

HCU 专用开关参见 [环境配置](vllm_hcu/platforms/envs.py)。
Attention/MoE 后端、KV cache 格式、推测解码及并行拓扑应按对应模型配置选择，
并非所有组合都经过验证。

### MHA/GQA 使用 FlashAttention

对于使用 `--attention-backend FLASH_ATTN` 的 MHA/GQA 模型，建议在模型选用的
所有后端均支持 HND 布局时设置 `VLLM_KV_CACHE_LAYOUT=HND`，与对应的
[模型配置](tests/models/) 保持一致：

```bash
VLLM_KV_CACHE_LAYOUT=HND vllm serve /path/to/model \
  --attention-backend FLASH_ATTN
```

这是支持该布局的 FlashAttention 路径的推荐配置，并非所有模型和后端的通用要求。
混合注意力模型即使指定了 `FLASH_ATTN`，也可能同时使用不支持 HND 的其他注意力后端。
此时应取消设置 `VLLM_KV_CACHE_LAYOUT`，由 vLLM 解析共同支持的布局。
MLA 等其他后端应遵循对应模型配置，不要全局强制使用 HND；模型特定限制参见
[验证记录](docs/validation/)。

## 运行时集成

vLLM 通过标准插件入口发现 HCU 平台、模型和算子注册表。插件使用精确匹配、
进程内生效的导入回调，并将补丁分为两个阶段：

- `apply_platform_patches()`：准备平台修正和框架集成。
- `apply_worker_patches(vllm_config)`：准备 Worker 使用的模型、算子及通信集成。

HCU 专用配置放在 `vllm_config.additional_config["hcu"]` 中，不修改 vLLM 配置类。
`patch_report()` 可报告进程角色、目标符号、补丁状态、失败详情及功能激活状态。
三个插件入口和两个补丁阶段共用源码定义的兼容性检查，不兼容时停止注册；
对应的 doctor 检查项为 `vllm_compatible`。

对于 `FLASHMLA_SPARSE`，平台将公开参数 `--kv-cache-dtype fp8_e4m3` 转换为内部的
`fp8_ds_mla` 格式。这是该稀疏 MLA 后端专用的打包布局，并非所有 Attention 后端
共享的通用 cache 格式。

## 模型与验证

模型已注册或存在 YAML 配置，不代表精度验证已经通过。复现时应匹配原记录中的
checkpoint、量化方式、硬件、后端和并行拓扑：

- [模型配置](tests/models/)：各 checkpoint 的 TP/DP/EP、MTP、Attention/MoE 后端及 KV cache 设置。
- [模型配置说明](tests/models/README.md)：配置选择方法，以及 Kimi、DeepSeek-V4 DSpark、GLM PCP 和多模态测试路径。
- [验证记录](docs/validation/)：冻结的环境、硬件、各配置得分、诊断失败和验证范围。

已有结果不代表其他量化、并行拓扑、计算图策略或更新构建具有相同精度。
诊断配置不能计为精度验收通过；历史结果也不代表后续每个提交都重新执行过验证。

HumanEval 会执行模型生成的代码，应在真正隔离的评测环境中运行，避免访问凭据及
主机敏感数据，并遵守验证记录中的隔离要求。仅设置一个环境变量并不会创建沙箱。

## 代码结构

```text
vllm_hcu/
├── __init__.py                  # 三个 vLLM 插件入口
├── compatibility.py             # 共用的兼容性检查
├── doctor.py                    # 只读安装诊断
├── patch/                       # 补丁注册、顺序、目标验证及小型适配器
│   ├── import_coordinator.py    # 精确导入回调与模块替换
│   ├── module_exchange.py       # vLLM 到 HCU 的模块映射
│   ├── runtime_state.py         # 进程角色、幂等、失败状态和报告
│   ├── config.py                # HcuFeatureConfig 配置契约
│   ├── platform/               # 进程级平台及框架适配
│   └── worker/                 # Worker 模型、算子及通信适配
├── runtime_compat/              # HCU 维护的兼容实现
├── model_executor/layers/       # 线性层、MoE 和量化实现
├── models/                      # HCU 模型实现
├── ops/                         # HCU 算子及回退实现
├── platforms/                   # 平台接口和环境配置
└── v1/
    ├── worker.py                # Worker 补丁边界与设备生命周期
    ├── hcu_model_runner_v2.py    # 当前 Model Runner V2 实现
    ├── hcu_model_runner.py       # 保留的旧代码，不是 Worker 回退路径
    ├── attention/               # Attention 后端、元数据和算子
    ├── executor/                # 多进程执行器
    └── spec_decode/             # 推测解码运行时
```

`patch/` 只负责注册、顺序、目标校验及小型适配。调度仍由上游 vLLM 负责；
Mooncake、Attention、MoE、通信器和执行器等主体逻辑位于各自的 HCU 实现模块中。
若 `module_exchange.py` 已声明映射，调用方应使用标准的 `vllm.*` 导入路径，
由导入协调器在上游模块加载前解析到 HCU 实现。

平台补丁在插件发现阶段注册。Worker 补丁在 `HcuGPUWorker.__init__` 中绑定到
反序列化后的 `vllm_config`，早于父类导入模型 runner 和自定义算子。
必需功能不兼容时会失败并保留进程内诊断状态，不回退到重写源码。

## 开发检查

在仓库根目录执行源码边界和补丁测试覆盖清单检查：

```bash
python3 tools/check_production_boundary.py --json
python3 tools/check_patch_test_coverage.py --json
```

准备测试依赖和匹配的 HCU vLLM 源码后，可执行可移植契约测试：

```bash
python3 -m pip install -r requirements-test.txt
python3 tools/run_patch_tests.py --suite contract --vllm-source /path/to/vllm-root
```

`/path/to/vllm-root` 应包含匹配的 `vllm/` 包，测试套件说明见
[tests/README.md](tests/README.md)。可移植契约测试不能代替 HCU 算子、分布式和模型精度测试。

`tools/check_production_boundary.py` 用于检查迁移专用元数据、带版本号的私有标记及
运行时模块名没有进入 `vllm_hcu/`。内部运行时标记统一使用不带版本号的 `_vllm_hcu_*` 前缀。
安装新 wheel 后应重启全部 Python/vLLM 进程，避免旧模块和算子注册状态残留。

---
