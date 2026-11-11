# vLLM HCU Plugin — v0.33.0

[English](README.md) | [简体中文](README.zh-CN.md)

本发布线通过 `adapter-das` 为官方 vLLM 0.33 提供可选 HCU 优化。
构建入口为独立项目 [`adapter_extension/`](adapter_extension/)，分发名为
`hcu-vllm-das-plugin`，Python 包名为 `hcu_vllm_das`。

## 版本兼容与三仓职责

目标为 epoch 为零的 vLLM 0.33 发布线，接受 patch、RC、dev、post 和 local
后缀。包版本与官方提交见
[version.py](adapter_extension/src/hcu_vllm_das/version.py)。版本准入通过
不等于 native ABI、算子或模型验收通过。

- vLLM 负责官方模型、调度、分配与执行接口。
- Adapter（宿主分发名 `hcu-adapter`、内核分发名 `sgl-kernel`，
  包名 `hcu_adapter`）负责真实 HCU 身份、
  runtime 准入、平台/通用插件入口和进程健康。
- 本 Plugin 经 `hcu_adapter.extensions` 提供可选 RMSNorm、显式
  FlashAttention、SlimQuant 和显式 INT8 ASM 候选。模型路由、CT 权重加载
  和通用后端构造沿用官方实现；各能力分别检查前置条件与验收门槛，
  INT8 候选不代表通用模型或拓扑支持。

Plugin 分发版本统一为 `0.33.0`，与 vLLM 发布线对应。仓库根目录的
`vllm_hcu` 源码保留为 vLLM 0.28.1 历史开发参考；根目录 `setup.py`
会在导入 Torch 前拒绝构建并提示使用 `./adapter_extension`，避免本分支
误产出旧 wheel。历史构建应使用对应的原发布分支。

## 构建与安装

在隔离环境中准备 ABI 匹配的 DTK PyTorch、Triton 和所需 vendor 库。
从仓库根目录执行：

```bash
python3 -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir ./dist ./adapter_extension
python3 -m pip install --no-deps /wheels/vllm-0.33.0+*.whl
python3 -m pip install --no-deps /wheels/sgl_kernel-*.whl /wheels/hcu_adapter-*.whl
python3 -m pip install --no-deps ./dist/hcu_vllm_das_plugin-*.whl
```

安装、快关、回退与运行态核验见 [Plugin 使用说明](adapter_extension/README.md)。

## 验证与开发

本发布线使用 [`tests/vllm033/`](tests/vllm033/)。实际测试命令、实现边界与
设备/模型待验项目见 [v0.33 验证记录](docs/validation/vllm033-upgrade.md)。
CPU 契约、设备数值、变输入 Graph、模型精度和性能分别验收；源码测试
不代替安装态 wheel 的验证。

本轮用户允许 7B 以下模型出现精度误差；原始分数、适用范围及 PR330 的
审查口径见 [验收更新](docs/validation/vllm033-small-model-acceptance.md)。

`validation/v0.32.0`、`tests/vllm032` 及旧版本设计/验证文档为历史证据，
其中的分数和产物不能转授本发布线。历史兼容锁测试与其输入保持对应。

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
