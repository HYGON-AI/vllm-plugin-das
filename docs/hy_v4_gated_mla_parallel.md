# HY V4 Gated-MLA g_proj Parallel Execution (PCP+EP)

## 概述

在 PCP+EP (Prefill Context Parallel + Expert Parallel) 场景下，实现 Gated-MLA 的 `g_proj` (gate projection) 与 Q/KV projections 的并行执行，减少端到端延迟。

## 背景

HY V4 模型的 Gated-MLA attention 包含以下计算流程：

```
hidden_states
    ├─> Q/KV projections ──> DSA indexer ──> Sparse MLA ──> attn_out
    └─> g_proj ──────────────────────────────────────────> gate_score

最终输出: sigmoid(gate_score) × attn_out
```

**现有实现的问题**：
- 串行执行：g_proj 在 attention 完成后才计算
- DCP A2A overlap：仅在 DCP (Decode Context Parallel) 场景下，g_proj 与 A2A 通信并行
- PCP+EP 场景无优化：g_proj 与 attention 路径串行，浪费计算资源

## 实现方案

### 核心思路

在 PCP+EP 场景下（`prefill_context_parallel_size > 1` 且 `expert_parallel_size > 1`），使用辅助 CUDA stream 并行执行：

1. **主流（default stream）**：
   - Q/KV projections
   - DSA indexer
   - Sparse MLA attention

2. **辅助流（gate_proj_stream）**：
   - g_proj(hidden_states)

3. **同步点**：
   - attention 完成后，等待 gate_proj_stream
   - 应用门控：`attn_out = sigmoid(gate_score) × attn_out`

### 执行流程

```
                    ┌─────────────────────────────────────┐
                    │         hidden_states               │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────┴────────────────┐
                    │   record start_event        │
                    └────────────┬────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │ (default stream)  │  (aux stream)   │
              ▼                   ▼                  │
    ┌─────────────────┐   ┌──────────────┐         │
    │ Q/KV projections│   │   g_proj()   │         │
    └────────┬────────┘   └──────┬───────┘         │
             │                    │                  │
             ▼                    │ record done_event│
    ┌─────────────────┐          │                  │
    │  DSA indexer    │          │                  │
    └────────┬────────┘          │                  │
             │                    │                  │
             ▼                    │                  │
    ┌─────────────────┐          │                  │
    │  Sparse MLA     │          │                  │
    └────────┬────────┘          │                  │
             │                    │                  │
             ▼                    │                  │
    ┌─────────────────┐          │                  │
    │ wait done_event │◄─────────┘                  │
    └────────┬────────┘                             │
             │                                       │
             ▼                                       │
    ┌─────────────────────────────────┐             │
    │ sigmoid(gate_score) × attn_out  │             │
    └─────────────────────────────────┘             │
```

### 代码变更

#### 1. 环境变量控制

**文件**: `vllm_hcu/platforms/envs.py`

```python
VLLM_HCU_ENABLE_HYV4_GATE_PROJ_PARALLEL: bool = False
```

启用方式：
```bash
export VLLM_HCU_ENABLE_HYV4_GATE_PROJ_PARALLEL=1
```

#### 2. Model 层创建 stream

**文件**: `vllm_hcu/models/hy_v4/model.py`

```python
def enable_hyv4_gate_proj_parallel(vllm_config: VllmConfig) -> bool:
    """Return whether HY V4 may overlap g_proj with Q/KV projections in PCP+EP."""
    parallel_config = vllm_config.parallel_config
    return bool(
        henvs.VLLM_HCU_ENABLE_HYV4_GATE_PROJ_PARALLEL
        and parallel_config.prefill_context_parallel_size > 1
        and parallel_config.expert_parallel_size > 1
    )

# 在 HYV4ForCausalLM.__init__ 中：
self.gate_proj_stream = (
    torch.cuda.Stream()
    if getattr(config, "gated_mla", False)
    and enable_hyv4_gate_proj_parallel(vllm_config)
    else None
)
```

#### 3. Attention 层接收 stream

**文件**: `vllm_hcu/models/hy_v4/attention.py`

```python
class HYV4MLAAttention(nn.Module):
    def __init__(
        self,
        ...,
        gate_proj_stream: torch.cuda.Stream | None = None,
    ):
        ...
        self.gate_proj_stream = gate_proj_stream
        self.gate_proj_start_event = (
            torch.cuda.Event() if gate_proj_stream is not None else None
        )
        self.gate_proj_done_event = (
            torch.cuda.Event() if gate_proj_stream is not None else None
        )
```

#### 4. Forward 方法并行执行

```python
def forward(self, positions, hidden_states, llama_4_scaling=None):
    gate_score_parallel: torch.Tensor | None = None
    
    if (
        self.gate_proj_stream is not None
        and self.gated_mla
        and self.linear_gate is not None
        and self.gate_a2a_stream is None  # PCP+EP path only
    ):
        def compute_qkv_projections():
            # Q/KV projections...
            return q_c, q, kv_lora

        def compute_gate_proj():
            return self.linear_gate(hidden_states)[0]

        (q_c, q, kv_lora), gate_score_parallel = maybe_execute_in_parallel(
            compute_qkv_projections,
            compute_gate_proj,
            self.gate_proj_start_event,
            self.gate_proj_done_event,
            self.gate_proj_stream,
        )
    else:
        # 串行路径（向后兼容）
        q_c, q, kv_lora = compute_qkv_projections()
    
    # ... RoPE, indexer, attention ...
    
    # 应用 gate
    if gate_score_parallel is not None:
        attn_out = attn_out * torch.sigmoid(gate_score_parallel)
    elif self.gated_mla and self.linear_gate and not self.gate_a2a_stream:
        gate_score = self.linear_gate(hidden_states)[0]
        attn_out = attn_out * torch.sigmoid(gate_score)
```

## 与 DCP A2A overlap 的区别

| 特性 | DCP A2A Overlap | PCP+EP g_proj Parallel |
|------|----------------|----------------------|
| **触发条件** | `decode_context_parallel_size > 1` | `prefill_context_parallel_size > 1` AND `expert_parallel_size > 1` |
| **并行对象** | g_proj 与 DCP A2A 通信 | g_proj 与 Q/KV projections |
| **stream** | `gate_a2a_stream` | `gate_proj_stream` |
| **环境变量** | `VLLM_HCU_ENABLE_HYV4_GATE_A2A_OVERLAP` | `VLLM_HCU_ENABLE_HYV4_GATE_PROJ_PARALLEL` |
| **互斥关系** | **是** - 两者不能同时启用（DCP 优先） | |

## 性能预期

### Prefill 阶段收益

假设单层 attention 的计算分解：
- Q/KV projections: ~30% 时间
- g_proj: ~10% 时间
- DSA indexer + Sparse MLA: ~60% 时间

**串行执行**：
```
总时间 = 30% + 10% + 60% = 100%
```

**并行执行（PCP+EP）**：
```
并行段: max(30%, 10%) = 30%
串行段: 60%
总时间 = 30% + 60% = 90%
理论加速: ~10%
```

### 适用场景

1. **Large batch prefill**: batch size > 512, 多个请求并发 prefill
2. **Long context**: 上下文长度 > 8K，Q/KV projections 和 g_proj 都较重
3. **EP sharding**: expert_parallel_size = 2/4/8, 专家权重分片后 MoE 计算变快，attention 成为瓶颈

## 限制与注意事项

1. **互斥性**：`gate_proj_stream` 和 `gate_a2a_stream` 互斥
   - DCP 场景优先使用 `gate_a2a_stream`（因为 DCP A2A 通信更慢，收益更大）
   - PCP+EP 场景使用 `gate_proj_stream`

2. **小 batch 开销**：batch size < 64 时，stream 同步开销可能抵消收益
   - 建议仅在 prefill batch size > 256 时启用

3. **CUDA stream 数量**：每层创建一个 aux stream
   - 对于 60 层模型，会创建 60 个 stream
   - HCU driver 需要支持大量并发 stream

4. **Cudagraph 兼容性**：
   - 当前实现在 eager mode 中工作
   - Cudagraph capture 时需要关闭（通过环境变量控制）

## 使用示例

### 启用并行

```bash
export VLLM_HCU_ENABLE_HYV4_GATE_PROJ_PARALLEL=1

python -m vllm.entrypoints.openai.api_server \
    --model /path/to/hy_v4_model \
    --tensor-parallel-size 8 \
    --prefill-context-parallel-size 4 \
    --expert-parallel-size 2 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192
```

### 验证日志

启动时应看到：
```
INFO: HY V4 gated-MLA g_proj/projection parallel (PCP+EP) enabled
```

### 性能对比

```bash
# Baseline (串行)
export VLLM_HCU_ENABLE_HYV4_GATE_PROJ_PARALLEL=0
# 运行 benchmark...

# Optimized (并行)
export VLLM_HCU_ENABLE_HYV4_GATE_PROJ_PARALLEL=1
# 运行 benchmark...
```

## 测试

### 单元测试

```python
# tests/models/hy_v4/test_gate_proj_parallel.py
def test_gate_proj_parallel_vs_serial_correctness():
    """Verify parallel g_proj produces identical results to serial."""
    ...

def test_stream_synchronization():
    """Verify gate_score is properly synchronized before use."""
    ...
```

### 集成测试

```bash
pytest tests/integration/models/test_hy_v4_pcp_ep.py \
    -k "test_gated_mla_parallel"
```

## 后续优化

1. **动态 stream 池**：共享 stream 而非每层独立创建
2. **自适应启用**：根据 batch size 动态开关并行
3. **Cudagraph 支持**：将并行路径集成到 cudagraph capture
4. **多 stream 扩展**：进一步并行化其他 GEMMs（如 kv_a_layernorm 前的计算）

## 参考

- **DCP A2A overlap 实现**: `vllm_hcu/models/hy_v4/attention.py:1364-1386`
- **DeepSeek V4 multi-stream**: `vllm_hcu/model_executor/layers/deepseek_v4_attention.py:390-442`
- **Multi-stream utils**: `vllm_hcu/utils/multi_stream_utils.py`
