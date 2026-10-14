# 诊断 3 次 All-Gather 的来源

## 问题描述
在 CUDA graph 模式下启动模型时，出现以下错误：
```
torch.AcceleratorError: CUDA error: operation n
```

错误发生在 `sparse_attn_indexer.py:980`，执行 `cache_k[valid_slots]` 时。

同时，profile 显示有 3 次 `nccl_all_gather` 操作。

## 根本原因

### CUDA Graph 错误
`forward_hip` 在 line 980 执行 `cache_k[valid_slots]`，这是**动态形状**操作：
- `valid_slots` 是布尔掩码，来自 `slot_mapping >= 0`
- 每次迭代的 valid slots 数量不同
- CUDA graph 要求所有操作有静态张量形状

**对比**：`forward_cuda` 使用 `@eager_break_during_capture` 装饰器在 graph capture 期间强制 eager 模式。

### 3 次 All-Gather 的可能来源

根据代码分析，3 次 all_gather 可能来自：

1. **q_b_proj / q_proj** 的 TP all_gather（如果 `gather_output=True`）
2. **kv_b_proj** 的 TP all_gather（如果 `gather_output=True`）
3. **PCP prefill cache gather**（`_gather_prefill_cache_inputs`）

或者：
1. **kv_b_proj** TP gather
2. **Indexer 内部的某个投影** TP gather
3. **DCP sinks gather** 或 **DCP Q replicate**

## 已实现的优化

### 1. kv_b_proj 手动 gather（与 g_proj 重叠）

**目标**：将 kv_b_proj 的 all_gather 与 g_proj GEMM 重叠执行。

**实现位置**：
- `__init__` (line 1053-1064): 根据 `gate_proj_stream` 决定是否禁用自动 gather
- `forward` (line 1294-1302): 提前调用 kv_b_proj local GEMM
- `_indexer_and_attn` (line 1391-1410): 在最优时机手动 gather

**关键代码**：
```python
# __init__
kv_b_gather_output = gate_proj_stream is None
self.kv_b_manual_gather = not kv_b_gather_output
self.kv_b_proj = ColumnParallelLinear(
    ...,
    gather_output=kv_b_gather_output,  # False when gate_proj_stream is set
)

# forward
if self.kv_b_manual_gather:
    original_gather_output = self.kv_b_proj.gather_output
    self.kv_b_proj.gather_output = False
    try:
        with torch.profiler.record_function("kv_b_proj_local_gemm"):
            kv_b_local, _ = self.kv_b_proj(kv_c_normed)
    finally:
        self.kv_b_proj.gather_output = original_gather_output

# _indexer_and_attn
if kv_b_local is not None:
    if tp_size > 1:
        with torch.profiler.record_function("kv_b_manual_all_gather"):
            gathered_list = [torch.empty_like(kv_b_local) for _ in range(tp_size)]
            dist.all_gather(gathered_list, kv_b_local)
            kv_b_gathered = torch.cat(gathered_list, dim=-1)
```

### 2. Profiler Markers

添加了以下 markers 来标识各个操作：
- `"kv_b_proj_local_gemm"`: kv_b_proj 的本地 GEMM（无 gather）
- `"kv_b_manual_all_gather"`: 手动 kv_b all_gather（应与 g_proj 重叠）

## 下一步诊断步骤

### 1. 添加更多 Profiler Markers

在以下位置添加 markers：

**q_proj / q_b_proj**:
```python
# 在 forward 中找到 q_proj 调用
with torch.profiler.record_function("q_proj_with_gather"):
    q, _ = q_proj_layer(...)
```

**linear_gate (g_proj)**:
```python
# 在 _indexer_and_attn 或 forward 中
with torch.profiler.record_function("linear_gate_proj"):
    gate_score = self.linear_gate(hidden_states)[0]
```

**PCP gather**:
```python
# 在 pcp.py 的 _gather_prefill_cache_inputs 中
with torch.profiler.record_function("pcp_prefill_cache_gather"):
    gathered = ...
```

### 2. 运行 Profile

```bash
# 启用 profiler
VLLM_TORCH_PROFILER_DIR=/path/to/profile python ...

# 或在代码中手动添加
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    # 运行一个 iteration
    ...

prof.export_chrome_trace("trace.json")
```

### 3. 分析 Trace

在 Chrome 中打开 `chrome://tracing`，加载 `trace.json`，搜索：
- `nccl_all_gather`
- `kv_b_proj_local_gemm`
- `kv_b_manual_all_gather`
- `linear_gate_proj`

确认：
1. kv_b_manual_all_gather 是否与 g_proj 重叠
2. 3 次 all_gather 分别来自哪里
3. 是否还有优化空间

## 修复 CUDA Graph 错误的方案

### 方案 A（推荐）：添加 eager_break_during_capture

将 `forward_hip` 内 lightop 分支抽成独立函数：

```python
@eager_break_during_capture
def _lightop_indexer_and_cache(self, ...):
    # 包含 line 980 的动态索引逻辑
    lightop_k = cache_k[valid_slots]
    ...

def forward_hip(self, ...):
    ...
    if use_lightop:
        self._lightop_indexer_and_cache(...)
    ...
```

### 方案 B：强制使用 safe cache writer

修改 line 974：
```python
# 检测是否在 CUDA graph capture 中
capturing = torch.cuda.is_current_stream_capturing()
use_safe_cache_writer = self.dcp_world_size > 1 or capturing
```

### 方案 C（临时验证）：全局启用 safe writer

```python
use_safe_cache_writer = True  # line 974
```

如果错误消失，证明诊断正确。

### 临时绕过：禁用 CUDA Graph

启动时添加：
```bash
--enforce-eager
```

## 文件修改清单

- `vllm_hcu/models/hy_v4/attention.py`:
  - Line 1053-1064: `__init__` 中设置 `kv_b_manual_gather`
  - Line 1294-1302: `forward` 中提前调用 kv_b_proj local GEMM
  - Line 1391-1402: `_indexer_and_attn` 中手动 gather kv_b
  - Line 1300, 1398: 添加 profiler markers

- `vllm_hcu/model_executor/layers/sparse_attn_indexer.py`:
  - Line 980: **待修复** - 添加 eager_break_during_capture 或条件分支
