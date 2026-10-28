# adapter-das + vLLM 0.32 跨机器迁移与验证手册

更新时间：2026-10-28

本文面向在 HCU 服务器部署官方 vLLM `v0.32.0`、`adapter-das` 与 DAS
优化插件的工程人员。本文只描述新组合；旧 Adapter 的验证结论不能直接继承。

## 1. 组件边界

```text
official vLLM 0.32.0
  └─ adapter-das (distribution: sgl-kernel)
       ├─ vllm.platform_plugins / vllm.general_plugins
       └─ hcu_adapter.extensions API v1
            └─ hcu-vllm-das-plugin
                 ├─ LightOp RMSNorm
                 └─ LightOp SiluAndMul
```

- Adapter 独占 vLLM platform/general plugin 入口和版本、平台、DTK 门禁。
- DAS Plugin 只声明 `hcu_adapter.extensions` 组的 `das_hcu` 入口。
- 官方 vLLM/ROCm/Triton 已有能力直接复用；Plugin 只承载官方没有的可选优化。
- 卸载或禁用 Plugin 后，vLLM 基线路径仍由 Adapter 提供。

## 2. 固定源码

| 组件 | 仓库/分支或提交 |
| --- | --- |
| 官方 vLLM | tag `v0.32.0`，commit `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Adapter | `https://github.com/alexanderbin123/adapter-das.git`，branch `feat/vllm032-runtime-integration`，验证基线 `81361880e203cb33cf631f77716ac8244f5f299f` |
| DAS Plugin | `https://github.com/alexanderbin123/vllm-plugin-das.git`，branch `feat/adapter-das-extension-api` |

Plugin 仓库根目录仍保留其他 vLLM 版本的开发代码；vLLM 0.32 只构建
`adapter_extension/`。

## 3. 当前候选 wheel

`validation/v0.32.0/compat-lock.json` 是机器可读锁。当前候选为：

| wheel | SHA256 |
| --- | --- |
| `vllm-0.32.0+rocm723-cp312-cp312-linux_x86_64.whl` | `de61890d0a66ae12b84908962c5426bc7a64d66ac8d8c64ceb2c2bc0c31f722a` |
| `sgl_kernel-0.1.0-py3-none-any.whl` | `bbc7de3745d736b5b8e82f55dfdab5e3a80ad8e8908e5388de76d394e7e0f50a` |
| `hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl` | `7186603c9c5aa09ad2a19c38589bebc68944b42ac90f51fe0922a44f2c73729c` |

这些哈希只标识本轮候选，不等于生产批准。重新构建后必须生成新哈希和新证据。

## 4. 构建

```bash
python3.12 -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /opt/build/vllm032 /opt/src/adapter-das

python3.12 -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /opt/build/vllm032 \
  /opt/src/vllm-plugin-das/adapter_extension
```

检查 Plugin wheel 的 `entry_points.txt`，必须只有：

```ini
[hcu_adapter.extensions]
das_hcu = hcu_vllm_das.extension:extension_spec
```

不得出现 `vllm.platform_plugins` 或 `vllm.general_plugins`。

## 5. 安装

目标环境必须已经具有 ABI 匹配的 HCU Torch、DTK/HIP 和 LightOp。按顺序安装，
并使用 `--no-deps` 防止 pip 从公网替换本地 HCU 组件：

```bash
python3.12 -m venv --system-site-packages /opt/hcu-vllm032/venv
export HCU_VENV=/opt/hcu-vllm032/venv
export HCU_WHEELS=/opt/hcu-vllm032/wheels

"$HCU_VENV/bin/python" -m pip install --no-deps \
  "$HCU_WHEELS/vllm-0.32.0+rocm723-cp312-cp312-linux_x86_64.whl"
"$HCU_VENV/bin/python" -m pip install --no-deps \
  "$HCU_WHEELS/sgl_kernel-0.1.0-py3-none-any.whl"
"$HCU_VENV/bin/python" -m pip install --no-deps \
  "$HCU_WHEELS/hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl"
```

共享节点若预装旧 vLLM 或旧 HCU 平台包，系统 `sitecustomize` 可能把
`/usr/local` 放到 venv wheel 前面。安装态验收必须检查 `vllm.__file__` 和各
distribution 的来源；不能只看 `pip install` 成功。

## 6. 安装态验收

```bash
export HCU_PLUGIN_SRC=/opt/src/vllm-plugin-das
VLLM_PLUGINS=hcu_adapter \
HSA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
"$HCU_VENV/bin/python" \
  "$HCU_PLUGIN_SRC/adapter_extension/tools/verify_installed.py"
```

验收脚本会 fail closed 地检查：

- vLLM release line 为 `0.32.0`；
- Adapter distribution 是 `sgl-kernel`，并拥有
  `hcu_adapter.bridges.vllm` 与 `hcu_adapter.extensions`；
- Adapter 是期望的 vLLM platform/general entry-point owner；
- 当前平台精确为 `hcu_adapter.bridges.vllm.HcuPlatform`；
- Plugin wheel 拥有 `das_hcu`，扩展状态为 `active`；
- RMSNorm/SiluAndMul capability 和 OOT class owner 精确匹配。

editable 安装、空 RECORD、模块来源越界、入口缺失、平台错误或扩展降级都不通过。

## 7. 便携与实机测试

```bash
cd "$HCU_PLUGIN_SRC"
HCU_ADAPTER_HOST=none \
VLLM_TARGET_ROOT=/opt/src/vllm-0.32.0 \
"$HCU_VENV/bin/python" -m pytest -q tests/vllm032
```

未设置 live 开关时，两项 HCU 测试应 skip；skip 不能当作实机通过。
本轮新组合的便携契约结果为 `62 passed, 2 skipped`。

在确认卡空闲后分别执行：

```bash
VLLM_PLUGINS=hcu_adapter HCU_VLLM032_RMSNORM_LIVE=1 \
HSA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
"$HCU_VENV/bin/python" -m pytest -q -rs \
  tests/vllm032/test_rms_norm_hcu.py

VLLM_PLUGINS=hcu_adapter HCU_VLLM032_LIVE=1 \
HSA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
"$HCU_VENV/bin/python" -m pytest -q -rs \
  tests/vllm032/test_silu_and_mul_hcu.py
```

测试只接受 `gfx936`/`gfx938`，并同时验证优化数值和 master/算子级回滚。
本轮在空闲 `gfx936` card 0 上 RMSNorm 与 SiluAndMul 分别得到
`1 passed`；结束后 8 张卡均回到 `0%` 使用率、`2 MiB` 显存基线。

## 8. 关闭 eager 模式

vLLM 默认 `enforce_eager=False`。服务命令不要传 `--enforce-eager`，并在日志或
配置中确认 compilation/cudagraph 已启用。例如：

```bash
VLLM_PLUGINS=hcu_adapter \
VLLM_HCU_USE_CUSTOM_OPS=1 \
VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1 \
VLLM_HCU_USE_CUSTOM_RMS_NORM=1 \
HIP_VISIBLE_DEVICES=0 \
"$HCU_VENV/bin/python" -m vllm.entrypoints.cli.main \
  serve /model/Qwen3-8B --dtype bfloat16 --max-model-len 4096
```

模型路径按目标机实际目录替换。

## 9. 回滚

```bash
# 全部优化回到 vLLM 原生实现
export VLLM_HCU_USE_CUSTOM_OPS=0

# 或只关闭一个算子
export VLLM_HCU_USE_CUSTOM_RMS_NORM=0
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=0
```

也可卸载 `hcu-vllm-das-plugin`。卸载后必须重启进程，不能在运行中的进程动态
撤销 OOT 注册。Adapter 和官方 vLLM 保持安装。

## 10. 尚未完成的生产门禁

新 Adapter/Plugin wheel 组合必须重新执行，不能继承旧组合结果：

- `/model` 下目标模型的 enabled/disabled 精度 A/B；
- HumanEval `0-63` 共 64 条；
- 非 eager 服务健康、日志 fatal 扫描与显存回收；
- 长稳、并发、拓扑和性能门禁。

在这些证据完成并写回 compat lock 前，状态保持 `pending`，不得宣称生产支持。
