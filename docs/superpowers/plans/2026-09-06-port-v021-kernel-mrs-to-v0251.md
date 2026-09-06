# Port v0.21 Kernel MRs to v0.25.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the four v0.21 HCU kernel-routing MRs to the v0.25.1 sidecar plugin with default-on selectors, deterministic fallbacks, real-model validation, and a reviewed pull request.

**Architecture:** Extend the existing exact post-import adapters instead of copying v0.21 source-replacement patches. Each wrapper remains scoped to its audited vLLM 0.25.1 module, selects optional HCU kernels in a fixed priority order, and calls the original target function when a selector, dependency, or supported-call contract does not apply.

**Tech Stack:** Python 3.10, PyTorch/HIP, vLLM 0.25.1, vLLM-HCU callback adapters, AITER HIP/Triton kernels, `causal_conv1d`, pytest, GitHub CLI/API.

**Spec:** `docs/superpowers/specs/2026-09-06-port-v021-kernel-mrs-to-v0251-design.md`

## Global Constraints

- Base every code commit on `origin/v0.25.1` commit `88f8a07` or its reviewed descendants already present on this branch.
- Change only `HYGON-AI/vllm-plugin-das`; do not modify either vLLM source checkout, AITER, `causal_conv1d`, or model files.
- Preserve exact vLLM 0.25.1 signature checks, callback idempotence, Qwen-local GDN ownership, and canonical-module isolation.
- Default every new Boolean selector to enabled; explicit `0` and `false` disable it.
- Catch optional-import failure only. Once a kernel is selected, propagate its runtime, shape, dtype, and launch errors.
- Run portable tests with `VLLM_V0251_SOURCE_ROOT=/models/vllm_0251` and `VLLM_PLUGINS=__disabled__`.
- Validate `/models/Qwen3.5-35B-A3B` with all new selectors absent from the environment.
- Use commit author `alexanderbin123 <1414695739@qq.com>` and never store access tokens in files, remotes, commits, test output, or PR text.

---

### Task 1: Add default-on environment contracts

**Files:**

- Modify: `tests/accuracy/test_environment_routing.py`
- Modify: `vllm_hcu/platforms/envs.py:11-68`
- Modify: `vllm_hcu/platforms/envs.py:224-236`

**Interfaces:**

- Consumes: lazy module-level resolution through `vllm_hcu.platforms.envs.__getattr__(name)`.
- Produces: Boolean attributes `VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE`, `VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP`, and `VLLM_HCU_USE_CHUNK_FWD_KERNEL_O`.

- [ ] **Step 1: Write failing default and opt-out tests**

Add these tests after `test_boolean_environment_values_are_lazily_parsed`:

```python
MIGRATED_KERNEL_FLAGS = (
    "VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE",
    "VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP",
    "VLLM_HCU_USE_CHUNK_FWD_KERNEL_O",
)


@pytest.mark.parametrize("name", MIGRATED_KERNEL_FLAGS)
def test_migrated_kernel_flags_default_on(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.delenv(name, raising=False)
    assert getattr(hcu_envs, name) is True
    assert hcu_envs.is_set(name) is False


@pytest.mark.parametrize("name", MIGRATED_KERNEL_FLAGS)
@pytest.mark.parametrize("value", ["0", "false", "FALSE"])
def test_migrated_kernel_flags_allow_explicit_opt_out(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    assert getattr(hcu_envs, name) is False
    assert hcu_envs.is_set(name) is True
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/accuracy/test_environment_routing.py \
  -k migrated_kernel_flags
```

Expected: collection succeeds and each case fails with `AttributeError` because
the new environment names are not registered.

- [ ] **Step 3: Register the three lazy environment variables**

Add `bool = True` annotations inside `TYPE_CHECKING`, then add these entries
beside the existing FLA and causal-convolution switches:

```python
    "VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP":
    lambda: _environment_flag(os.environ.get(
        "VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP", "True"
    )),
    "VLLM_HCU_USE_CHUNK_FWD_KERNEL_O":
    lambda: _environment_flag(os.environ.get(
        "VLLM_HCU_USE_CHUNK_FWD_KERNEL_O", "True"
    )),
    "VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE":
    lambda: _environment_flag(os.environ.get(
        "VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE", "True"
    )),
```

- [ ] **Step 4: Run the environment tests and verify GREEN**

Run the command from Step 2, then run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/accuracy/test_environment_routing.py
```

Expected: all environment-routing tests pass.

- [ ] **Step 5: Commit the environment contract**

```bash
git add vllm_hcu/platforms/envs.py tests/accuracy/test_environment_routing.py
git diff --cached --check
git commit -m "feat(hcu): add default-on kernel routing flags"
```

---

### Task 2: Add HIP-first `chunk_delta_h` routing

**Files:**

- Modify: `tests/runtime_patch/test_attention_mla_fla_mamba.py:940-980`
- Modify: `vllm_hcu/patch/worker/op_opt/patch_fla_chunk_delta_h.py:19-89`

**Interfaces:**

- Consumes: the Task 1 flag `VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP`, existing `VLLM_HCU_USE_CUSTOM_AITER_FLA`, and existing `VLLM_HCU_USE_CUSTOM_OPS`.
- Produces: wrapper `hcu_chunk_delta_h(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=FLA_CHUNK_SIZE, save_new_value=True, cu_seqlens=None, chunk_indices=None, chunk_offsets=None, use_exp2=False)` with HIP → AITER Triton → original routing.

- [ ] **Step 1: Replace the obsolete missing-import assertion with routing tests**

Use `_install_fake_module` to install both optional modules. Record calls from
the following fake HIP function and assert it receives the caller's values:

```python
def hip(k, w, u, g=None, gk=None, initial_state=None,
        initial_state_indices=None, output_final_state=True,
        chunk_size=64, save_new_value=True, cu_seqlens=None,
        chunk_indices=None, chunk_offsets=None, use_exp2=False,
        transpose_state_layout=True, kernel_cfg=None):
    calls.append(("hip", chunk_size, output_final_state, save_new_value,
                  cu_seqlens, chunk_indices, chunk_offsets,
                  transpose_state_layout, kernel_cfg))
    return "hip-h", "hip-v", "hip-final"
```

Cover the three tiers with this helper and tests:

```python
def _chunk_delta_module(adapter, original):
    return _module(
        adapter.TARGET_MODULE,
        FLA_CHUNK_SIZE=64,
        chunk_gated_delta_rule_fwd_h=original,
        prepare_chunk_indices=lambda seq, size: torch.tensor([[0, 0]]),
        prepare_chunk_offsets=lambda seq, size: torch.tensor([0]),
        triton=SimpleNamespace(cdiv=lambda value, divisor: (value + divisor - 1) // divisor),
        torch=torch,
    )


def _chunk_delta_original(k, w, u, g=None, gk=None, initial_state=None,
                          output_final_state=False, chunk_size=64,
                          save_new_value=True, cu_seqlens=None,
                          chunk_indices=None, chunk_offsets=None,
                          use_exp2=False):
    return "original-h", "original-v", "original-final"


def test_fla_chunk_delta_h_prefers_hip_and_preserves_metadata(monkeypatch):
    adapter = _adapter("patch_fla_chunk_delta_h")
    calls = []

    def hip(k, w, u, g=None, gk=None, initial_state=None,
            initial_state_indices=None, output_final_state=True,
            chunk_size=64, save_new_value=True, cu_seqlens=None,
            chunk_indices=None, chunk_offsets=None, use_exp2=False,
            transpose_state_layout=True, kernel_cfg=None):
        calls.append((chunk_size, output_final_state, save_new_value,
                      cu_seqlens, chunk_indices, chunk_offsets,
                      use_exp2, transpose_state_layout, kernel_cfg))
        return "hip-h", "hip-v", "hip-final"

    _install_fake_module(
        monkeypatch,
        "aiter.ops.fla",
        chunk_gated_delta_rule_fwd_vllm_hip_blockdim64=hip,
    )
    _install_fake_module(
        monkeypatch,
        "aiter.ops.triton.fla.vllm.chunk_delta_h",
        launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64=lambda **kwargs: pytest.fail(
            "HIP must have priority"
        ),
    )
    module = _chunk_delta_module(adapter, _chunk_delta_original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP", True
    )
    cu_seqlens = torch.tensor([0, 2])
    result = module.chunk_gated_delta_rule_fwd_h(
        torch.zeros(1, 2, 1, 4),
        torch.zeros(1, 2, 1, 4),
        torch.zeros(1, 2, 1, 4),
        output_final_state=True,
        chunk_size=32,
        save_new_value=False,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
    )
    assert result == ("hip-h", "hip-v", "hip-final")
    assert calls[0][0:3] == (32, True, False)
    assert calls[0][3] is cu_seqlens
    assert calls[0][6:] == (True, True, None)


def test_fla_chunk_delta_h_uses_aiter_triton_when_hip_is_unavailable(
    monkeypatch,
):
    adapter = _adapter("patch_fla_chunk_delta_h")
    calls = []
    _install_fake_module(monkeypatch, "aiter.ops.fla")
    _install_fake_module(
        monkeypatch,
        "aiter.ops.triton.fla.vllm.chunk_delta_h",
        launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64=(
            lambda **kwargs: calls.append(kwargs)
        ),
    )
    module = _chunk_delta_module(adapter, _chunk_delta_original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP", True
    )
    module.chunk_gated_delta_rule_fwd_h(
        torch.zeros(1, 2, 1, 4),
        torch.zeros(1, 2, 1, 4),
        torch.zeros(1, 2, 1, 4),
        chunk_size=32,
        use_exp2=True,
    )
    assert calls[0]["BT"] == 32
    assert calls[0]["use_exp2"] is True
    assert calls[0]["transpose_state_layout"] is True


def test_fla_chunk_delta_h_uses_original_when_optional_aiter_is_unavailable(
    monkeypatch,
):
    adapter = _adapter("patch_fla_chunk_delta_h")
    _install_fake_module(monkeypatch, "aiter.ops.fla")
    _install_fake_module(monkeypatch, "aiter.ops.triton.fla.vllm.chunk_delta_h")
    module = _chunk_delta_module(adapter, _chunk_delta_original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP", True
    )
    assert module.chunk_gated_delta_rule_fwd_h(
        torch.zeros(1, 2, 1, 4),
        torch.zeros(1, 2, 1, 4),
        torch.zeros(1, 2, 1, 4),
    ) == ("original-h", "original-v", "original-final")


def test_fla_chunk_delta_h_propagates_selected_hip_runtime_error(monkeypatch):
    adapter = _adapter("patch_fla_chunk_delta_h")

    def failing_hip(*args, **kwargs):
        raise RuntimeError("hip delta launch failed")

    _install_fake_module(
        monkeypatch,
        "aiter.ops.fla",
        chunk_gated_delta_rule_fwd_vllm_hip_blockdim64=failing_hip,
    )
    module = _chunk_delta_module(adapter, _chunk_delta_original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP", True
    )
    with pytest.raises(RuntimeError, match="hip delta launch failed"):
        module.chunk_gated_delta_rule_fwd_h(
            torch.zeros(1, 2, 1, 4),
            torch.zeros(1, 2, 1, 4),
            torch.zeros(1, 2, 1, 4),
        )
```

- [ ] **Step 2: Run the four tests and verify RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/runtime_patch/test_attention_mla_fla_mamba.py \
  -k 'fla_chunk_delta_h'
```

Expected: HIP-priority and original-fallback assertions fail; the current
adapter knows only the AITER Triton route and raises when it is absent.

- [ ] **Step 3: Implement deterministic three-tier routing**

Keep the exact target-signature check. Inside the wrapper, prepare sequence
indices and offsets once, then select the HIP callable only when all three
flags are true:

```python
def _hip_enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs
    return bool(
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_CUSTOM_AITER_FLA
        and henvs.VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP
    )
```

Resolve optional imports with independent `try/except ImportError` blocks.
When the HIP callable exists, call it with:

```python
return hip_kernel(
    k, w, u, g=g, gk=gk, initial_state=initial_state,
    initial_state_indices=None, output_final_state=output_final_state,
    chunk_size=chunk_size, save_new_value=save_new_value,
    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    chunk_offsets=chunk_offsets, use_exp2=use_exp2,
    transpose_state_layout=True, kernel_cfg=None,
)
```

If HIP is unavailable, retain the current AITER Triton allocation and launch
when `_enabled()` is true. If that import is absent, invoke `original` with
all thirteen original arguments instead of raising a synthetic error.

- [ ] **Step 4: Run focused and shared FLA tests and verify GREEN**

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/runtime_patch/test_attention_mla_fla_mamba.py \
  -k 'fla_chunk_delta_h or fla_chunk_o'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the state-update router**

```bash
git add vllm_hcu/patch/worker/op_opt/patch_fla_chunk_delta_h.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py
git diff --cached --check
git commit -m "feat(hcu): prefer HIP gated-delta state kernel"
```

---

### Task 3: Add HIP-first `chunk_o` routing

**Files:**

- Modify: `tests/runtime_patch/test_attention_mla_fla_mamba.py:910-940`
- Modify: `vllm_hcu/patch/worker/op_opt/patch_fla_chunk_o.py:19-68`

**Interfaces:**

- Consumes: the Task 1 flag `VLLM_HCU_USE_CHUNK_FWD_KERNEL_O`, existing `VLLM_HCU_USE_CUSTOM_AITER_FLA`, and existing `VLLM_HCU_USE_CUSTOM_OPS`.
- Produces: wrapper `hcu_chunk_o(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_indices=None, chunk_size=FLA_CHUNK_SIZE, core_attn_out=None)` with HIP → AITER Triton → original routing and output-buffer preservation.

- [ ] **Step 1: Add HIP priority, fallback, and output-buffer tests**

Add this helper and the five tests; the output-buffer assertion uses real CPU
tensors:

```python
def _chunk_o_module(adapter, original):
    return _module(
        adapter.TARGET_MODULE,
        FLA_CHUNK_SIZE=64,
        chunk_fwd_o=original,
        prepare_chunk_indices=lambda seq, size: torch.tensor([[0, 0]]),
        triton=SimpleNamespace(cdiv=lambda value, divisor: (value + divisor - 1) // divisor),
        torch=torch,
    )


def _chunk_o_original(q, k, v, h, g=None, scale=None, cu_seqlens=None,
                      chunk_indices=None, chunk_size=64,
                      core_attn_out=None):
    return "original-output"


def test_fla_chunk_o_prefers_hip_and_preserves_call_contract(monkeypatch):
    adapter = _adapter("patch_fla_chunk_o")
    calls = []

    def hip(**kwargs):
        calls.append(kwargs)
        return torch.ones_like(kwargs["v"])

    _install_fake_module(
        monkeypatch,
        "aiter.ops.fla",
        chunk_fwd_o_vllm_hip_blockdim64=hip,
    )
    _install_fake_module(
        monkeypatch,
        "aiter.ops.triton.fla.vllm.chunk_o",
        launch_chunk_fwd_kernel_o=lambda **kwargs: pytest.fail(
            "HIP must have priority"
        ),
    )
    module = _chunk_o_module(adapter, _chunk_o_original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CHUNK_FWD_KERNEL_O", True)
    cu_seqlens = torch.tensor([0, 2])
    indices = torch.tensor([[0, 0]])
    tensor = torch.zeros(1, 2, 1, 4)
    module.chunk_fwd_o(
        tensor, tensor, tensor, tensor, scale=0.25,
        cu_seqlens=cu_seqlens, chunk_indices=indices, chunk_size=32,
    )
    assert calls[0]["scale"] == 0.25
    assert calls[0]["cu_seqlens"] is cu_seqlens
    assert calls[0]["chunk_indices"] is indices
    assert calls[0]["chunk_size"] == 32
    assert calls[0]["transpose_state_layout"] is True


def test_fla_chunk_o_copies_hip_result_into_core_output(monkeypatch):
    adapter = _adapter("patch_fla_chunk_o")
    hip_result = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 4)
    _install_fake_module(
        monkeypatch,
        "aiter.ops.fla",
        chunk_fwd_o_vllm_hip_blockdim64=lambda **kwargs: hip_result,
    )
    module = _chunk_o_module(adapter, _chunk_o_original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CHUNK_FWD_KERNEL_O", True)
    core = torch.full((16,), -1.0)
    tensor = torch.zeros(1, 2, 1, 4)
    result = module.chunk_fwd_o(
        tensor, tensor, tensor, tensor, core_attn_out=core
    )
    assert result.data_ptr() == core.data_ptr()
    torch.testing.assert_close(result, hip_result)


def test_fla_chunk_o_uses_aiter_triton_when_hip_is_unavailable(monkeypatch):
    adapter = _adapter("patch_fla_chunk_o")
    calls = []
    _install_fake_module(monkeypatch, "aiter.ops.fla")
    _install_fake_module(
        monkeypatch,
        "aiter.ops.triton.fla.vllm.chunk_o",
        launch_chunk_fwd_kernel_o=lambda **kwargs: calls.append(kwargs),
    )
    module = _chunk_o_module(adapter, _chunk_o_original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CHUNK_FWD_KERNEL_O", True)
    tensor = torch.zeros(1, 2, 1, 4)
    module.chunk_fwd_o(tensor, tensor, tensor, tensor, chunk_size=32)
    assert calls[0]["BT"] == 32
    assert calls[0]["transpose_state_layout"] is True


def test_fla_chunk_o_uses_original_when_optional_aiter_is_unavailable(
    monkeypatch,
):
    adapter = _adapter("patch_fla_chunk_o")
    _install_fake_module(monkeypatch, "aiter.ops.fla")
    _install_fake_module(monkeypatch, "aiter.ops.triton.fla.vllm.chunk_o")
    module = _chunk_o_module(adapter, _chunk_o_original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CHUNK_FWD_KERNEL_O", True)
    tensor = torch.zeros(1, 2, 1, 4)
    assert module.chunk_fwd_o(tensor, tensor, tensor, tensor) == (
        "original-output"
    )


def test_fla_chunk_o_propagates_selected_hip_runtime_error(monkeypatch):
    adapter = _adapter("patch_fla_chunk_o")

    def failing_hip(**kwargs):
        raise RuntimeError("hip output launch failed")

    _install_fake_module(
        monkeypatch,
        "aiter.ops.fla",
        chunk_fwd_o_vllm_hip_blockdim64=failing_hip,
    )
    module = _chunk_o_module(adapter, _chunk_o_original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CHUNK_FWD_KERNEL_O", True)
    tensor = torch.zeros(1, 2, 1, 4)
    with pytest.raises(RuntimeError, match="hip output launch failed"):
        module.chunk_fwd_o(tensor, tensor, tensor, tensor)
```

Keep the existing feature-off numeric test and explicitly disable both the
HIP selector and `VLLM_HCU_USE_CUSTOM_AITER_FLA` in it.

- [ ] **Step 2: Run the `chunk_o` tests and verify RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/runtime_patch/test_attention_mla_fla_mamba.py \
  -k 'fla_chunk_o'
```

Expected: HIP route and import-fallback tests fail against the current
Triton-only implementation.

- [ ] **Step 3: Implement output routing and buffer preservation**

Add a HIP selector independent of the AITER-Triton selector:

```python
def _hip_enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs
    return bool(
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_CHUNK_FWD_KERNEL_O
    )
```

Import `chunk_fwd_o_vllm_hip_blockdim64` and
`launch_chunk_fwd_kernel_o` independently. Call HIP with the exact caller
metadata and `transpose_state_layout=True`. Preserve the buffer contract with:

```python
hip_output = hip_kernel(
    q=q, k=k, v=v, h=h, g=g, g_gamma=None, scale=scale,
    cu_seqlens=cu_seqlens, chunk_size=chunk_size,
    chunk_indices=chunk_indices, use_exp2=False,
    transpose_state_layout=True, kernel_cfg=None,
)
if core_attn_out is None:
    return hip_output
out.copy_(hip_output)
return out
```

Retain the existing AITER Triton launch as tier two. If neither optional
callable is selected, call the original target function with all ten original
arguments.

- [ ] **Step 4: Run all FLA adapter tests and verify GREEN**

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/runtime_patch/test_attention_mla_fla_mamba.py \
  -k 'fla_chunk_o or fla_chunk_delta_h'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the output router**

```bash
git add vllm_hcu/patch/worker/op_opt/patch_fla_chunk_o.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py
git diff --cached --check
git commit -m "feat(hcu): prefer HIP gated-delta output kernel"
```

---

### Task 4: Route compatible Qwen GDN convolution updates

**Files:**

- Modify: `tests/runtime_patch/test_gdn_v0251_ownership.py:280-380`
- Modify: `tests/runtime_patch/test_attention_mla_fla_mamba.py:1120-1280`
- Modify: `vllm_hcu/patch/worker/op_opt/patch_gdn_causal_conv1d.py:87-111`

**Interfaces:**

- Consumes: existing `VLLM_HCU_USE_CUSTOM_OPS`, `VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D`, `use_nn_layout()`, and the audited v0.25.1 `causal_conv1d_update` signature.
- Produces: Qwen-local compatible-call router to `causal_conv1d.causal_conv1d_interface.causal_conv1d_update`; unsupported v0.25.1 calls and missing imports return through the original function.

- [ ] **Step 1: Add external-route and unsupported-call tests**

Install a fake external module using `_install_fake_module`. Its callable must
expose the real seven-argument contract and record every value:

```python
def custom_update(x, conv_state, weight, bias=None, activation=None,
                  cache_seqlens=None, conv_state_indices=None):
    calls["custom"] = {
        "x": x,
        "conv_state": conv_state,
        "weight": weight,
        "bias": bias,
        "activation": activation,
        "cache_seqlens": cache_seqlens,
        "conv_state_indices": conv_state_indices,
    }
    return "custom-update"
```

Add these tests. Existing ownership tests continue to prove that canonical
and non-Qwen consumers are unchanged:

```python
def _causal_route_module(adapter, original_update):
    original_update.__signature__ = inspect.signature(  # type: ignore[attr-defined]
        _gdn_causal_conv1d_update
    )
    return _module(
        adapter.TARGET_MODULE,
        causal_conv1d_fn=_gdn_causal_conv1d_fn,
        causal_conv1d_update=original_update,
    )


def test_qwen_causal_update_routes_compatible_decode_to_external(monkeypatch):
    adapter = _adapter("patch_gdn_causal_conv1d")
    calls = {}

    def original(*args, **kwargs):
        calls["original"] = (args, kwargs)
        return "original-update"

    def custom_update(x, conv_state, weight, bias=None, activation=None,
                      cache_seqlens=None, conv_state_indices=None):
        calls["custom"] = {
            "x": x,
            "conv_state": conv_state,
            "weight": weight,
            "bias": bias,
            "activation": activation,
            "cache_seqlens": cache_seqlens,
            "conv_state_indices": conv_state_indices,
        }
        return "custom-update"

    _install_fake_module(
        monkeypatch,
        "causal_conv1d.causal_conv1d_interface",
        causal_conv1d_update=custom_update,
    )
    module = _causal_route_module(adapter, original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D", True)
    x = torch.empty(2, 8)
    state = torch.empty(1, 8, 3)
    physical_weight = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    indices = torch.tensor([0, 1])
    assert module.causal_conv1d_update(
        x, state, physical_weight, activation="silu",
        conv_state_indices=indices, validate_data=True,
    ) == "custom-update"
    assert "original" not in calls
    torch.testing.assert_close(
        calls["custom"]["weight"], physical_weight.T.contiguous()
    )
    assert calls["custom"]["conv_state_indices"] is indices


def test_qwen_causal_update_falls_back_for_spec_metadata(monkeypatch):
    adapter = _adapter("patch_gdn_causal_conv1d")
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "original-update"

    _install_fake_module(
        monkeypatch,
        "causal_conv1d.causal_conv1d_interface",
        causal_conv1d_update=lambda *args, **kwargs: pytest.fail(
            "spec metadata is unsupported by the external kernel"
        ),
    )
    module = _causal_route_module(adapter, original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_USE_NN", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D", True)
    accepted = torch.tensor([1])
    query_start = torch.tensor([0, 1])
    result = module.causal_conv1d_update(
        torch.empty(1, 8), torch.empty(1, 8, 3), torch.empty(8, 4),
        num_accepted_tokens=accepted, query_start_loc=query_start,
        max_query_len=1,
    )
    assert result == "original-update"
    assert calls[0][1]["num_accepted_tokens"] is accepted
    assert calls[0][1]["query_start_loc"] is query_start


def test_qwen_causal_update_falls_back_when_external_module_is_missing(
    monkeypatch,
):
    adapter = _adapter("patch_gdn_causal_conv1d")
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "original-update"

    _install_fake_module(monkeypatch, "causal_conv1d.causal_conv1d_interface")
    module = _causal_route_module(adapter, original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_USE_NN", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D", True)
    assert module.causal_conv1d_update(
        torch.empty(1, 8), torch.empty(1, 8, 3), torch.empty(8, 4)
    ) == "original-update"
    assert len(calls) == 1


def test_qwen_causal_update_propagates_selected_kernel_error(monkeypatch):
    adapter = _adapter("patch_gdn_causal_conv1d")

    def original(*args, **kwargs):
        return "original-update"

    def failing_custom(*args, **kwargs):
        raise RuntimeError("causal update launch failed")

    _install_fake_module(
        monkeypatch,
        "causal_conv1d.causal_conv1d_interface",
        causal_conv1d_update=failing_custom,
    )
    module = _causal_route_module(adapter, original)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_USE_NN", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D", True)
    with pytest.raises(RuntimeError, match="causal update launch failed"):
        module.causal_conv1d_update(
            torch.empty(1, 8), torch.empty(1, 8, 3), torch.empty(8, 4)
        )
```

- [ ] **Step 2: Run the GDN causal tests and verify RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/runtime_patch/test_gdn_v0251_ownership.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py \
  -k 'gdn and causal'
```

Expected: the compatible call returns the original sentinel because no
external routing exists yet.

- [ ] **Step 3: Implement contract-aware routing after layout normalization**

Bind and apply defaults once per call. Normalize `weight` when NN layout is
enabled, then decide whether the external API can preserve semantics:

```python
def _supports_external_update(arguments: dict[str, object]) -> bool:
    return (
        arguments["num_accepted_tokens"] is None
        and arguments["query_start_loc"] is None
        and arguments["max_query_len"] == -1
        and arguments["null_block_id"] == 0
        and arguments["block_idx_last_scheduled_token"] is None
        and arguments["initial_state_idx"] is None
    )
```

When both selectors are enabled and the call is supported, import the external
callable. On `ImportError`, use the bound original call. Otherwise call:

```python
return custom_update(
    bound.arguments["x"],
    bound.arguments["conv_state"],
    bound.arguments["weight"],
    bound.arguments["bias"],
    bound.arguments["activation"],
    conv_state_indices=bound.arguments["conv_state_indices"],
)
```

Never pass `validate_data` to the external API. Use `causal_update(*bound.args,
**bound.kwargs)` for every fallback so NN-layout normalization remains active.

- [ ] **Step 4: Run the full Qwen GDN ownership suites and verify GREEN**

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/runtime_patch/test_gdn_v0251_ownership.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py -k gdn
```

Expected: all selected tests pass, including idempotence, signature rejection,
and real v0.25.1 cold-import ownership.

- [ ] **Step 5: Commit the causal-convolution router**

```bash
git add vllm_hcu/patch/worker/op_opt/patch_gdn_causal_conv1d.py \
  tests/runtime_patch/test_gdn_v0251_ownership.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py
git diff --cached --check
git commit -m "feat(hcu): route compatible Qwen GDN causal updates"
```

---

### Task 5: Route Qwen GDN sigmoid-gating updates

**Files:**

- Modify: `tests/runtime_patch/test_gdn_v0251_ownership.py:280-470`
- Modify: `tests/runtime_patch/test_attention_mla_fla_mamba.py:1270-1330`
- Modify: `vllm_hcu/patch/worker/op_opt/patch_gdn_linear_attention.py:24-118`

**Interfaces:**

- Consumes: Task 1 flag `VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE`, existing `VLLM_HCU_USE_CUSTOM_OPS`, Qwen's module-local official sigmoid function, optional `aiter.vllm_fused_sigmoid_gating_delta_rule_update`, and optional AITER Triton sigmoid function.
- Produces: Qwen-local sigmoid wrapper with compatible AITER HIP → AITER Triton → official routing while preserving the existing native fused-convolution NN-layout wrapper.

- [ ] **Step 1: Replace retired-path assertions with selector tests**

Retain the assertion that `fused_recurrent_gated_delta_rule_packed_decode`
stays target-owned. Add these helpers and independent tests using CPU tensors:

```python
def _gdn_sigmoid_update(A_log, a, b, dt_bias, q, k, v, beta=1.0,
                        threshold=20.0, scale=None, initial_state=None,
                        inplace_final_state=True, cu_seqlens=None,
                        ssm_state_indices=None, num_accepted_tokens=None,
                        use_qk_l2norm_in_kernel=False, is_kda=False):
    return "official-sigmoid"


def _sigmoid_module(adapter, official=_gdn_sigmoid_update):
    return _module(
        adapter.TARGET_MODULE,
        GDN_AITER_TRITON_AVAILABLE=False,
        fused_recurrent_gated_delta_rule_packed_decode=(
            lambda *args, **kwargs: "official-recurrent"
        ),
        fused_sigmoid_gating_delta_rule_update=official,
    )


def _sigmoid_arguments(dtype):
    return {
        "A_log": torch.zeros(1, dtype=torch.float32),
        "a": torch.zeros(1, dtype=dtype),
        "b": torch.zeros(1, dtype=dtype),
        "dt_bias": torch.zeros(1, dtype=torch.float32),
        "q": torch.zeros(1, dtype=dtype),
        "k": torch.zeros(1, dtype=dtype),
        "v": torch.zeros(1, dtype=dtype),
    }


def test_qwen_sigmoid_prefers_aiter_hip_for_matching_dtype(monkeypatch):
    adapter = _adapter("patch_gdn_linear_attention")
    _install_fake_module(
        monkeypatch,
        "aiter",
        vllm_fused_sigmoid_gating_delta_rule_update=(
            lambda *args, **kwargs: "hip-sigmoid"
        ),
    )
    _install_fake_module(
        monkeypatch,
        "aiter.ops.triton.fla.fused_sigmoid_gating",
        fused_sigmoid_gating_delta_rule_update=lambda *args, **kwargs: pytest.fail(
            "matching dtype must prefer HIP"
        ),
    )
    module = _sigmoid_module(adapter)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE",
        True,
    )
    assert module.fused_sigmoid_gating_delta_rule_update(
        **_sigmoid_arguments(torch.float32)
    ) == "hip-sigmoid"


def test_qwen_sigmoid_uses_aiter_triton_for_mixed_a_log_dtype(monkeypatch):
    adapter = _adapter("patch_gdn_linear_attention")
    _install_fake_module(
        monkeypatch,
        "aiter",
        vllm_fused_sigmoid_gating_delta_rule_update=lambda *args, **kwargs: pytest.fail(
            "mixed A_log dtype must skip HIP"
        ),
    )
    _install_fake_module(
        monkeypatch,
        "aiter.ops.triton.fla.fused_sigmoid_gating",
        fused_sigmoid_gating_delta_rule_update=(
            lambda *args, **kwargs: "triton-sigmoid"
        ),
    )
    module = _sigmoid_module(adapter)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE",
        True,
    )
    assert module.fused_sigmoid_gating_delta_rule_update(
        **_sigmoid_arguments(torch.bfloat16)
    ) == "triton-sigmoid"


@pytest.mark.parametrize("mode", ["disabled", "missing"])
def test_qwen_sigmoid_uses_official_when_disabled_or_aiter_missing(
    monkeypatch,
    mode,
):
    adapter = _adapter("patch_gdn_linear_attention")
    if mode == "disabled":
        _install_fake_module(
            monkeypatch,
            "aiter",
            vllm_fused_sigmoid_gating_delta_rule_update=(
                lambda *args, **kwargs: pytest.fail("selector is disabled")
            ),
        )
        _install_fake_module(
            monkeypatch,
            "aiter.ops.triton.fla.fused_sigmoid_gating",
            fused_sigmoid_gating_delta_rule_update=(
                lambda *args, **kwargs: pytest.fail("selector is disabled")
            ),
        )
    else:
        _install_fake_module(monkeypatch, "aiter")
        _install_fake_module(
            monkeypatch, "aiter.ops.triton.fla.fused_sigmoid_gating"
        )
    module = _sigmoid_module(adapter)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE",
        mode != "disabled",
    )
    assert module.fused_sigmoid_gating_delta_rule_update(
        **_sigmoid_arguments(torch.bfloat16)
    ) == "official-sigmoid"


def test_qwen_sigmoid_wrapper_is_qwen_local_and_idempotent(monkeypatch):
    adapter = _adapter("patch_gdn_linear_attention")
    canonical = _gdn_sigmoid_update
    module = _sigmoid_module(adapter, canonical)
    recurrent = module.fused_recurrent_gated_delta_rule_packed_decode
    assert adapter.apply_to_module(module) is True
    assert adapter.apply_to_module(module) is False
    assert module.fused_sigmoid_gating_delta_rule_update is not canonical
    assert module._vllm_hcu_original_fused_sigmoid is canonical
    assert module.fused_recurrent_gated_delta_rule_packed_decode is recurrent


def test_qwen_sigmoid_propagates_selected_kernel_error(monkeypatch):
    adapter = _adapter("patch_gdn_linear_attention")

    def failing_hip(*args, **kwargs):
        raise RuntimeError("sigmoid update launch failed")

    _install_fake_module(
        monkeypatch,
        "aiter",
        vllm_fused_sigmoid_gating_delta_rule_update=failing_hip,
    )
    module = _sigmoid_module(adapter)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE",
        True,
    )
    with pytest.raises(RuntimeError, match="sigmoid update launch failed"):
        module.fused_sigmoid_gating_delta_rule_update(
            **_sigmoid_arguments(torch.float32)
        )
```

Update the cold-import test to assert Qwen's sigmoid is wrapped,
`_vllm_hcu_original_fused_sigmoid` is the canonical FLA function, and the
canonical FLA export is unchanged.

- [ ] **Step 2: Run the sigmoid tests and verify RED**

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/runtime_patch/test_gdn_v0251_ownership.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py \
  -k 'gdn and sigmoid'
```

Expected: routing assertions fail because the current adapter deliberately
leaves the sigmoid function target-owned.

- [ ] **Step 3: Compose the sigmoid wrapper with the existing native wrapper**

Keep the native AITER update target at `TARGETS[0]` for compatibility and add
the Qwen sigmoid target as `TARGETS[1]`. Always validate and wrap the local
sigmoid symbol, even when `GDN_AITER_TRITON_AVAILABLE` is false. Import the HIP
and Triton callables independently at invocation time.

Select HIP only when `A_log.dtype == q.dtype`; otherwise select AITER Triton.
Use the exact original call signature by declaring the wrapper as `*args,
**kwargs`, binding through `inspect.signature(official_sigmoid)`, applying
defaults, and forwarding `*bound.args, **bound.kwargs` to the selected
callable. If both optional imports fail or either selector is off, call the
official function with the same bound arguments.

Mark the wrapper with `_vllm_hcu_qwen_gdn_sigmoid_wrapper`, store the original
as `_vllm_hcu_original_fused_sigmoid`, include it in the `already_applied`
contract, and leave the recurrent symbol untouched.

- [ ] **Step 4: Run all GDN and focused baseline tests and verify GREEN**

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/runtime_patch/test_gdn_v0251_ownership.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py \
  tests/accuracy/test_environment_routing.py
```

Expected: the focused baseline and all newly added tests pass.

- [ ] **Step 5: Commit the sigmoid router**

```bash
git add vllm_hcu/patch/worker/op_opt/patch_gdn_linear_attention.py \
  tests/runtime_patch/test_gdn_v0251_ownership.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py
git diff --cached --check
git commit -m "feat(hcu): route Qwen GDN sigmoid updates through AITER"
```

---

### Task 6: Verify the complete migration on tests and hardware

**Files:**

- Inspect: `docs/superpowers/specs/2026-09-06-port-v021-kernel-mrs-to-v0251-design.md`
- Inspect: all files changed since `origin/v0.25.1`
- Runtime artifact: `/tmp/vllm-hcu-integration/logs/`

**Interfaces:**

- Consumes: all adapters and environment contracts from Tasks 1-5.
- Produces: fresh test, static-check, and real-model evidence suitable for the pull-request description.

- [ ] **Step 1: Run whitespace, syntax, and patch inventory checks**

```bash
git diff --check origin/v0.25.1...HEAD
python -m compileall -q vllm_hcu tests
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
python tools/run_patch_tests.py --suite inventory --vllm-source /models/vllm_0251
```

Expected: all commands exit zero.

- [ ] **Step 2: Run the full portable contract suite**

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
python tools/run_patch_tests.py --suite contract --vllm-source /models/vllm_0251
```

Expected: zero failed tests. Record passed/skipped counts exactly.

- [ ] **Step 3: Verify default selectors in a fresh interpreter**

```bash
env -u VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE \
    -u VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP \
    -u VLLM_HCU_USE_CHUNK_FWD_KERNEL_O \
    PYTHONPATH=/models/vllm_0251:$PWD \
    VLLM_PLUGINS=__disabled__ \
    python -c 'from vllm_hcu.platforms import envs; names=("VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE","VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP","VLLM_HCU_USE_CHUNK_FWD_KERNEL_O"); print({name: getattr(envs, name) for name in names})'
```

Expected: all three printed values are `True`.

- [ ] **Step 4: Run one real Qwen3.5 TP2/EP2 smoke case with defaults**

```bash
env -u VLLM_HCU_USE_AITER_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE \
    -u VLLM_HCU_USE_AITER_CHUNK_GATED_DELTA_RULE_HIP \
    -u VLLM_HCU_USE_CHUNK_FWD_KERNEL_O \
    VLLM_HCU_QWEN35_35B_A3B_MODEL=/models/Qwen3.5-35B-A3B \
    VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
    python tools/run_patch_tests.py --suite model \
      --vllm-source /models/vllm_0251 \
      --target 'tests/integration/parallel/test_tp_ep_models.py::test_qwen35_35b_a3b_tp_ep_smoke[aiter-auto-shuffle-tp2-ep2]'
```

Expected: the model loads on two gfx938 HCU devices, generates two non-empty
responses of one to four tokens, and exits with zero failures. Inspect the
newest Qwen log under `/tmp/vllm-hcu-integration/logs/` and fail the task if it
contains a patch compatibility error, optional-import traceback, or kernel
launch exception.

- [ ] **Step 5: Review the requirement-to-evidence mapping**

Check each spec section against the committed diff and record:

```bash
git log --oneline origin/v0.25.1..HEAD
git diff --stat origin/v0.25.1...HEAD
git diff origin/v0.25.1...HEAD -- \
  vllm_hcu/platforms/envs.py \
  vllm_hcu/patch/worker/op_opt/patch_fla_chunk_delta_h.py \
  vllm_hcu/patch/worker/op_opt/patch_fla_chunk_o.py \
  vllm_hcu/patch/worker/op_opt/patch_gdn_causal_conv1d.py \
  vllm_hcu/patch/worker/op_opt/patch_gdn_linear_attention.py
```

Expected: the diff contains only scoped production code, tests, the approved
spec, and this plan; all four source MR behaviors map to a tested target
adapter.

---

### Task 7: Independent review, remediation, and pull-request delivery

**Files:**

- Inspect: complete diff `origin/v0.25.1...HEAD`
- Modify only if review identifies a concrete defect in a changed file.

**Interfaces:**

- Consumes: verified Task 6 commits and exact source-MR mapping.
- Produces: reviewed remote branch and one GitHub pull request targeting `v0.25.1`.

- [ ] **Step 1: Request an independent code review**

Set the review range:

```bash
BASE_SHA=$(git rev-parse origin/v0.25.1)
HEAD_SHA=$(git rev-parse HEAD)
```

Ask the reviewer to compare `$BASE_SHA..$HEAD_SHA` against the approved spec,
with special attention to optional-import semantics, v0.25.1 argument
preservation, output-buffer aliasing, dtype routing, module ownership,
idempotence, and default-enabled switches.

- [ ] **Step 2: Resolve every critical or important review finding**

For each valid finding, first add a failing regression test and run its exact
node id to observe RED. Apply the smallest production fix, rerun the node to
observe GREEN, then rerun all focused files:

```bash
VLLM_V0251_SOURCE_ROOT=/models/vllm_0251 \
PYTHONPATH=/models/vllm_0251:$PWD \
VLLM_PLUGINS=__disabled__ \
pytest -q tests/accuracy/test_environment_routing.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py \
  tests/runtime_patch/test_gdn_v0251_ownership.py
```

Commit reviewed corrections with:

```bash
git add -u
git diff --cached --check
git commit -m "fix(hcu): address kernel migration review"
```

If there are no critical or important findings, do not create an empty commit.

- [ ] **Step 3: Re-run completion verification after review**

Repeat Task 6 Steps 1-4 against the final `HEAD`. Completion claims and PR
creation are blocked unless the fresh commands exit zero.

- [ ] **Step 4: Push the reviewed branch without persisting credentials**

```bash
git status --short --branch
git push --set-upstream origin feat/port-021-kernel-mrs-v0251
```

Expected: the remote branch is created and the configured remote remains
`https://github.com/HYGON-AI/vllm-plugin-das.git` without embedded credentials.

- [ ] **Step 5: Create one pull request targeting `v0.25.1`**

Use title:

```text
feat(hcu): port v0.21 GDN and FLA kernel routes to v0.25.1
```

The body must include the five source commit hashes, explain why target-native
adapters replace v0.21 text patches, list all three default-on environment
variables, report exact portable/model test commands and counts, summarize
the independent review outcome, and state that AI assistance was used and the
human submitter must review every changed line.

Create it with:

```bash
gh pr create \
  --repo HYGON-AI/vllm-plugin-das \
  --base v0.25.1 \
  --head feat/port-021-kernel-mrs-v0251 \
  --title 'feat(hcu): port v0.21 GDN and FLA kernel routes to v0.25.1' \
  --body-file /tmp/vllm-hcu-port-pr-body.md
```

The temporary body file must contain no credentials and may be removed after
the command returns. Confirm the returned PR URL and query its changed-file
list before reporting completion.
