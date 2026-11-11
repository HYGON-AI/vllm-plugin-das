# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only contracts for the Kimi-K3/K3.1 DSpark draft HCU attention path.

Covers ``patch_kimi_k3_dspark_draft_attention`` (draft-scoped KV dtype and
FLASHMLA / TRITON_MLA pin), the platform's dense-MLA routing for the draft
signature (576 latent, dense, BF16 / fp8_ds_mla KV), the dense backends'
validate_configuration matrix, and the per-group builder dtype helper.

No test initializes a GPU/HIP device: backend kernels are never imported for
execution, device capability is a fixed DeviceCapability(9, 3), and every
test asserts CUDA stayed uninitialized.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.models import kimi_k3_dspark as policy
from vllm_hcu.patch.worker.op_opt import (
    patch_kimi_k3_dspark_draft_attention as adapter,
)
from vllm_hcu.v1.attention.backends.mla.kv_dtype import (
    effective_group_kv_cache_dtype,
    is_quantized_kv_cache_name,
)

REPO = Path(__file__).resolve().parents[2]
DSPARK_HEAD_SIZE = 576  # kv_lora_rank 512 + qk_rope_head_dim 64
KV_DTYPE_ENV = "VLLM_HCU_K3_DSPARK_DRAFT_KV_DTYPE"
BACKEND_ENV = "VLLM_HCU_K3_DSPARK_DRAFT_MLA_BACKEND"


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(KV_DTYPE_ENV, raising=False)
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    yield
    # Strict on devices-less hosts. On a GPU deployment node, merely importing
    # vllm initializes a probe context (platform-level behavior, not something
    # any test here does -- none of them allocates tensors on a device), so the
    # no-init assertion only holds where no device is visible.
    if torch.cuda.device_count() == 0:
        assert not torch.cuda.is_initialized()


def _selector_config(kv_cache_dtype, *, use_sparse=False, head_size=DSPARK_HEAD_SIZE):
    from vllm.v1.attention.selector import AttentionSelectorConfig

    return AttentionSelectorConfig(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=None,
        use_mla=True,
        use_sparse=use_sparse,
    )


def _hcu_platform(monkeypatch: pytest.MonkeyPatch, rejected=()):
    """HCUPlatform with a fixed capability and per-backend validators.

    ``rejected`` maps backend name -> reasons; every other backend accepts
    only the dtypes its HCU class declares (mirrors the deployment probe:
    FLASHMLA / TRITON_MLA reject fp8_ds_mla, sparse backends reject dense).
    """

    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm_hcu.platforms.hcu import HCUPlatform

    rejected = dict(rejected)
    dense_kv = ("auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2")
    sparse = {"FLASHMLA_SPARSE", "ROCM_AITER_MLA_SPARSE"}

    def get_class(backend):
        class Validator:
            @classmethod
            def validate_configuration(cls, **kwargs):
                if backend.name in rejected:
                    return list(rejected[backend.name])
                reasons = []
                if backend.name in sparse:
                    if not kwargs.get("use_sparse"):
                        reasons.append("non-sparse not supported")
                elif kwargs.get("kv_cache_dtype") not in dense_kv:
                    reasons.append("kv_cache_dtype not supported")
                return reasons

        return Validator

    monkeypatch.setattr(AttentionBackendEnum, "get_class", get_class)
    monkeypatch.setattr(
        HCUPlatform,
        "get_device_capability",
        classmethod(lambda _cls, device_id=0: DeviceCapability(9, 3)),
    )
    return HCUPlatform


# ---------------------------------------------------------------------------
# Environment switches
# ---------------------------------------------------------------------------


def test_env_defaults_are_bf16_draft_kv_and_platform_priority() -> None:
    from vllm_hcu.platforms import envs as henvs

    assert henvs.VLLM_HCU_K3_DSPARK_DRAFT_KV_DTYPE == "auto"
    assert henvs.VLLM_HCU_K3_DSPARK_DRAFT_MLA_BACKEND == "auto"
    assert KV_DTYPE_ENV in henvs.hcu_vllm_environment_variables
    assert BACKEND_ENV in henvs.hcu_vllm_environment_variables


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("flashmla", "FLASHMLA"),
        ("FLASHMLA", "FLASHMLA"),
        ("flash_mla", "FLASHMLA"),
        ("triton", "TRITON_MLA"),
        ("TRITON_MLA", "TRITON_MLA"),
        (" triton_mla ", "TRITON_MLA"),
    ],
)
def test_backend_env_aliases(monkeypatch, raw, expected) -> None:
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setenv(BACKEND_ENV, raw)
    assert henvs.k3_dspark_draft_mla_backend() == expected


@pytest.mark.parametrize("raw", ["auto", "bfloat16", "fp8_e4m3", "inherit", " BFloat16 "])
def test_kv_dtype_env_accepts_documented_values(monkeypatch, raw) -> None:
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setenv(KV_DTYPE_ENV, raw)
    assert henvs.k3_dspark_draft_kv_dtype() == raw.strip().lower()


@pytest.mark.parametrize(
    ("name", "raw"),
    [(KV_DTYPE_ENV, "fp8_ds_mla"), (KV_DTYPE_ENV, "fp16"), (BACKEND_ENV, "FLASHMLA_SPARSE")],
)
def test_invalid_env_values_fail_closed(monkeypatch, name, raw) -> None:
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError, match=name):
        getattr(henvs, name)


# ---------------------------------------------------------------------------
# Pure dtype / spec mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("run_dtype", "requested", "expected"),
    [
        # The crash signature: sparse-only run dtype -> dense draft dtype.
        ("fp8_ds_mla", "auto", "auto"),
        ("fp8_ds_mla", "bfloat16", "bfloat16"),
        ("fp8_ds_mla", "fp8_e4m3", "fp8_e4m3"),
        # Switch off: keep the run dtype (pre-adapter behavior).
        ("fp8_ds_mla", "inherit", None),
        # Dense-servable run dtypes are never rewritten.
        ("auto", "auto", None),
        ("bfloat16", "fp8_e4m3", None),
        ("fp8_e4m3", "auto", None),
        ("fp8", "bfloat16", None),
        (None, "auto", None),
    ],
)
def test_resolve_draft_kv_cache_dtype_matrix(run_dtype, requested, expected) -> None:
    assert policy.resolve_draft_kv_cache_dtype(run_dtype, requested) == expected


def test_resolve_draft_kv_cache_dtype_rejects_unknown_request() -> None:
    with pytest.raises(ValueError, match="fp8_ds_mla"):
        policy.resolve_draft_kv_cache_dtype("fp8_ds_mla", "fp8_ds_mla")


@pytest.mark.parametrize(
    ("cache_dtype", "torch_dtype", "bytes_per_token"),
    [
        ("auto", torch.bfloat16, 1152),
        ("bfloat16", torch.bfloat16, 1152),
        ("fp8_e4m3", torch.float8_e4m3fn, 576),
    ],
)
def test_latent_page_bytes_match_cpu_tensor_storage(
    cache_dtype, torch_dtype, bytes_per_token
) -> None:
    # One 64-token kernel page of the 576-wide latent, materialized on CPU.
    page = torch.zeros((64, DSPARK_HEAD_SIZE), dtype=torch_dtype)
    assert policy.mla_latent_bytes_per_token(cache_dtype) == bytes_per_token
    assert page.nbytes == 64 * policy.mla_latent_bytes_per_token(cache_dtype)


def test_fp8_ds_mla_record_size_and_bf16_draft_page_cost() -> None:
    # 512 fp8 NoPE + 4 fp32 tile scales + 64 bf16 RoPE = 656 B.
    nope = torch.zeros(512, dtype=torch.float8_e4m3fn).nbytes
    scales = torch.zeros(4, dtype=torch.float32).nbytes
    rope = torch.zeros(64, dtype=torch.bfloat16).nbytes
    assert policy.mla_latent_bytes_per_token("fp8_ds_mla") == nope + scales + rope
    draft_page = 64 * policy.mla_latent_bytes_per_token("auto")
    target_page = 64 * policy.mla_latent_bytes_per_token("fp8_ds_mla")
    assert (draft_page, target_page) == (73728, 41984)
    # Not page-divisible: the draft forms its own KV-cache group.
    assert draft_page % target_page != 0
    with pytest.raises(ValueError):
        policy.mla_latent_bytes_per_token("fp8_ds_mla", head_size=512)


def test_bf16_latent_round_trip_is_exact_on_cpu() -> None:
    # The draft's BF16 pages hold the normalized latent bit-exactly, unlike
    # the fp8_ds_mla record which quantizes the 512 NoPE lanes.
    torch.manual_seed(0)
    latent = torch.randn(7, DSPARK_HEAD_SIZE).to(torch.bfloat16)
    cache = torch.zeros((2, 64, DSPARK_HEAD_SIZE), dtype=torch.bfloat16)
    slots = torch.tensor([0, 1, 2, 63, 64, 65, 127])
    cache.view(-1, DSPARK_HEAD_SIZE)[slots] = latent
    assert torch.equal(cache.view(-1, DSPARK_HEAD_SIZE)[slots], latent)
    fp8 = latent[:, :512].to(torch.float8_e4m3fn).to(torch.bfloat16)
    assert not torch.equal(fp8, latent[:, :512])


@pytest.mark.parametrize(
    ("spec", "run_dtype", "expected"),
    [
        (SimpleNamespace(dtype=torch.bfloat16), "fp8_ds_mla", "auto"),
        (SimpleNamespace(dtype=torch.float16), "fp8_e4m3", "auto"),
        (SimpleNamespace(dtype=torch.float32), "fp8_ds_mla", "auto"),
        (SimpleNamespace(dtype=torch.uint8), "fp8_ds_mla", "fp8_ds_mla"),
        (SimpleNamespace(dtype=torch.float8_e4m3fn), "fp8_e4m3", "fp8_e4m3"),
        (SimpleNamespace(dtype=torch.bfloat16), "auto", "auto"),
        (SimpleNamespace(dtype=torch.uint8), "auto", "auto"),
        (SimpleNamespace(), "fp8_ds_mla", "fp8_ds_mla"),
        (
            SimpleNamespace(
                kv_cache_specs={
                    "a": SimpleNamespace(dtype=torch.bfloat16),
                    "b": SimpleNamespace(dtype=torch.bfloat16),
                }
            ),
            "fp8_ds_mla",
            "auto",
        ),
        (
            SimpleNamespace(
                kv_cache_specs={
                    "a": SimpleNamespace(dtype=torch.bfloat16),
                    "b": SimpleNamespace(dtype=torch.uint8),
                }
            ),
            "fp8_ds_mla",
            "fp8_ds_mla",
        ),
    ],
)
def test_effective_group_kv_cache_dtype(spec, run_dtype, expected) -> None:
    assert effective_group_kv_cache_dtype(spec, run_dtype) == expected


def test_quantized_name_mirror() -> None:
    assert not is_quantized_kv_cache_name("auto")
    assert not is_quantized_kv_cache_name("bfloat16")
    assert is_quantized_kv_cache_name("fp8_ds_mla")
    assert is_quantized_kv_cache_name("fp8_e4m3")


def test_dense_builders_plan_from_group_storage_dtype() -> None:
    # The FlashMLA/Triton dense builders must not plan FP8 scheduler metadata
    # or the FP8 non-causal fast path from the run dtype alone.
    flashmla = (REPO / "vllm_hcu/v1/attention/backends/mla/flashmla.py").read_text()
    triton = (REPO / "vllm_hcu/v1/attention/backends/mla/triton_mla.py").read_text()
    for source in (flashmla, triton):
        assert "effective_group_kv_cache_dtype(" in source
        assert "kv_cache_spec, vllm_config.cache_config.cache_dtype" in source
    assert "is_quantized_kv_cache(\n            vllm_config.cache_config.cache_dtype" not in (
        flashmla.replace("\r\n", "\n")
    )
    assert "if spec_nc and subpage and self.is_fp8_kvcache:" in flashmla


# ---------------------------------------------------------------------------
# Construction scope
# ---------------------------------------------------------------------------


def test_scope_restores_shared_config_and_selector_even_on_failure() -> None:
    cache_config = SimpleNamespace(cache_dtype="fp8_ds_mla")
    original = object()
    mla_module = SimpleNamespace(__name__=adapter.MLA_MODULE, get_attn_backend=original)

    with pytest.raises(RuntimeError, match="boom"):
        with policy.draft_attention_scope(cache_config, "auto", mla_module, "TRITON_MLA"):
            assert cache_config.cache_dtype == "auto"
            assert mla_module.get_attn_backend._vllm_hcu_dspark_pin == "TRITON_MLA"
            raise RuntimeError("boom")
    assert cache_config.cache_dtype == "fp8_ds_mla"
    assert mla_module.get_attn_backend is original

    with policy.draft_attention_scope(cache_config, None, mla_module, "auto"):
        assert cache_config.cache_dtype == "fp8_ds_mla"
        assert mla_module.get_attn_backend is original


def test_pinned_selector_delegates_non_draft_requests() -> None:
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "original"

    pinned = policy.make_pinned_get_attn_backend("FLASHMLA", original)
    assert pinned(576, torch.bfloat16, "auto", use_mla=True, use_sparse=True) == "original"
    assert pinned(128, torch.bfloat16, "auto") == "original"
    assert pinned(576, torch.bfloat16, "auto", True) == "original"
    assert len(calls) == 3
    with pytest.raises(ValueError, match="FLASHMLA_SPARSE"):
        policy.make_pinned_get_attn_backend("FLASHMLA_SPARSE", original)


def test_selector_kwargs_follow_running_schema() -> None:
    from vllm.v1.attention.selector import AttentionSelectorConfig

    kwargs = policy.selector_config_kwargs(
        AttentionSelectorConfig._fields,
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        use_sparse=False,
        decoder_attn_type="decoder",
    )
    config = AttentionSelectorConfig(**kwargs)
    assert (config.head_size, config.kv_cache_dtype, config.use_mla) == (576, "auto", True)
    assert config.use_sparse is False
    assert set(kwargs) <= set(AttentionSelectorConfig._fields)


# ---------------------------------------------------------------------------
# Platform routing for the draft signature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("custom_ops", "rejected", "expected"),
    [
        ("1", {}, "FLASHMLA"),
        ("0", {}, "TRITON_MLA"),
        ("1", {"FLASHMLA": ["flash_mla is not available."]}, "TRITON_MLA"),
    ],
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16", "fp8_e4m3"])
def test_platform_default_priority_routes_draft_signature(
    monkeypatch, custom_ops, rejected, expected, kv_cache_dtype
) -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", custom_ops)
    platform = _hcu_platform(monkeypatch, rejected)
    path = platform.get_attn_backend_cls(None, _selector_config(kv_cache_dtype))
    assert path == AttentionBackendEnum[expected].get_path()


def test_platform_still_rejects_dense_fp8_ds_mla_with_actionable_hint(monkeypatch) -> None:
    platform = _hcu_platform(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        platform.get_attn_backend_cls(None, _selector_config("fp8_ds_mla"))
    message = str(excinfo.value)
    assert "No valid attention backend found" in message
    assert "kv_cache_dtype not supported" in message
    assert KV_DTYPE_ENV in message


def test_platform_sparse_fp8_ds_mla_unchanged_and_hint_free(monkeypatch) -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    platform = _hcu_platform(monkeypatch)
    path = platform.get_attn_backend_cls(
        None, _selector_config("fp8_ds_mla", use_sparse=True)
    )
    assert path == AttentionBackendEnum.FLASHMLA_SPARSE.get_path()

    platform = _hcu_platform(
        monkeypatch,
        {"FLASHMLA_SPARSE": ["x"], "ROCM_AITER_MLA_SPARSE": ["y"]},
    )
    with pytest.raises(ValueError) as excinfo:
        platform.get_attn_backend_cls(
            None, _selector_config("fp8_ds_mla", use_sparse=True)
        )
    assert KV_DTYPE_ENV not in str(excinfo.value)


# ---------------------------------------------------------------------------
# validate_configuration / supports_combination matrix (real HCU classes)
# ---------------------------------------------------------------------------


def _dense_backend_class(name: str):
    if name == "FLASHMLA":
        module = pytest.importorskip("vllm_hcu.v1.attention.backends.mla.flashmla")
        return module.HcuFlashMLABackend
    module = pytest.importorskip("vllm_hcu.v1.attention.backends.mla.triton_mla")
    return module.HcuTritonMLABackend


@pytest.mark.parametrize("backend_name", ["FLASHMLA", "TRITON_MLA"])
@pytest.mark.parametrize(
    ("kv_cache_dtype", "valid"),
    [
        ("auto", True),
        ("bfloat16", True),
        ("fp8_e4m3", True),
        ("fp8_ds_mla", False),
    ],
)
def test_dense_backend_validate_configuration_matrix(
    monkeypatch, backend_name, kv_cache_dtype, valid
) -> None:
    from vllm.platforms.interface import DeviceCapability
    from vllm_hcu.v1.attention.ops import flashmla as flashmla_ops

    # flash_mla's extension is not installed on CPU hosts; supports_combination
    # resolves its availability probe at call time.
    monkeypatch.setattr(flashmla_ops, "is_flashmla_dense_supported", lambda: (True, None))
    backend = _dense_backend_class(backend_name)
    reasons = backend.validate_configuration(
        device_capability=DeviceCapability(9, 3),
        **_selector_config(kv_cache_dtype)._asdict(),
    )
    if valid:
        assert reasons == []
    else:
        assert any("kv_cache_dtype" in reason for reason in reasons), reasons


@pytest.mark.parametrize("backend_name", ["FLASHMLA", "TRITON_MLA"])
@pytest.mark.parametrize("requested", ["auto", "bfloat16", "fp8_e4m3"])
def test_every_draft_remap_target_is_declared_by_both_dense_backends(
    backend_name, requested
) -> None:
    backend = _dense_backend_class(backend_name)
    remapped = policy.resolve_draft_kv_cache_dtype("fp8_ds_mla", requested)
    assert remapped in backend.supported_kv_cache_dtypes
    assert "fp8_ds_mla" not in backend.supported_kv_cache_dtypes


# ---------------------------------------------------------------------------
# Adapter on a structural stand-in of nvidia.dspark_mla
# ---------------------------------------------------------------------------


def _install_draft_modules(monkeypatch):
    calls: list[tuple] = []
    mla_module = ModuleType(adapter.MLA_MODULE)

    def stock_get_attn_backend(head_size, dtype, kv_cache_dtype, **kwargs):
        calls.append((head_size, kv_cache_dtype, kwargs))
        return "stock-selector"

    class MultiHeadLatentAttention:
        def __init__(self, *, config, cache_config, prefix, **kwargs):
            del config, kwargs
            self.layer_name = prefix
            self.kv_cache_dtype = cache_config.cache_dtype
            # Global lookup in the K3 MLA module, exactly like the stock layer.
            self.attn_backend = sys.modules[adapter.MLA_MODULE].get_attn_backend(
                DSPARK_HEAD_SIZE,
                torch.bfloat16,
                self.kv_cache_dtype,
                use_mla=True,
                use_sparse=False,
                num_heads=8,
            )

    MultiHeadLatentAttention.__module__ = adapter.MLA_MODULE
    mla_module.MultiHeadLatentAttention = MultiHeadLatentAttention
    mla_module.get_attn_backend = stock_get_attn_backend

    target = ModuleType(adapter.TARGET_MODULE)
    target.MultiHeadLatentAttention = MultiHeadLatentAttention

    class K3DSparkDecoderLayer:
        def __init__(self, *, vllm_config, config, layer_idx, start_layer_id, prefix):
            self.self_attn = MultiHeadLatentAttention(
                config=config,
                cache_config=vllm_config.cache_config,
                prefix=f"{prefix}layers.{start_layer_id + layer_idx}.self_attn",
            )

    target.K3DSparkDecoderLayer = K3DSparkDecoderLayer
    monkeypatch.setitem(sys.modules, adapter.MLA_MODULE, mla_module)
    assert adapter.apply_to_module(target) is True
    assert adapter.apply_to_module(target) is False
    return target, mla_module, stock_get_attn_backend, calls


def _build_layer(target, run_dtype):
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=run_dtype))
    layer = target.K3DSparkDecoderLayer(
        vllm_config=vllm_config,
        config=SimpleNamespace(),
        layer_idx=0,
        start_layer_id=93,
        prefix="model.",
    )
    return layer, vllm_config


def test_adapter_metadata_and_inventory() -> None:
    from vllm_hcu.patch import worker as worker_dispatcher

    assert adapter.TARGETS == (
        "vllm.models.kimi_k3.nvidia.dspark_mla.K3DSparkDecoderLayer.__init__",
    )
    callbacks = worker_dispatcher.worker_callback_names()
    positions = {patch_id: index for index, (patch_id, _) in enumerate(callbacks)}
    assert (adapter.PATCH_ID, adapter.TARGET_MODULE) in callbacks
    assert positions["worker.op_opt.kimi_k31.dsa_layers"] < positions[adapter.PATCH_ID]
    assert worker_dispatcher._patch_features()[adapter.PATCH_ID] == "always"


def test_default_remaps_fp8_ds_mla_draft_to_bf16_and_restores_run_dtype(monkeypatch) -> None:
    target, mla_module, stock, calls = _install_draft_modules(monkeypatch)
    layer, vllm_config = _build_layer(target, "fp8_ds_mla")

    assert layer.self_attn.kv_cache_dtype == "auto"
    assert vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
    assert mla_module.get_attn_backend is stock
    assert calls == [
        (DSPARK_HEAD_SIZE, "auto", {"use_mla": True, "use_sparse": False, "num_heads": 8})
    ]


@pytest.mark.parametrize("requested", ["bfloat16", "fp8_e4m3"])
def test_explicit_draft_kv_dtype(monkeypatch, requested) -> None:
    monkeypatch.setenv(KV_DTYPE_ENV, requested)
    target, _mla, _stock, calls = _install_draft_modules(monkeypatch)
    layer, vllm_config = _build_layer(target, "fp8_ds_mla")
    assert layer.self_attn.kv_cache_dtype == requested
    assert calls[0][1] == requested
    assert vllm_config.cache_config.cache_dtype == "fp8_ds_mla"


@pytest.mark.parametrize(
    ("kv_env", "run_dtype"),
    [
        ("inherit", "fp8_ds_mla"),  # switch off -> stock construction
        (None, "auto"),  # Kimi-K3 BF16 run
        (None, "bfloat16"),
        (None, "fp8_e4m3"),  # Kimi-K3 FP8 run (already dense-servable)
    ],
)
def test_switch_off_or_dense_run_keeps_stock_behavior(monkeypatch, kv_env, run_dtype) -> None:
    if kv_env is not None:
        monkeypatch.setenv(KV_DTYPE_ENV, kv_env)
    target, mla_module, stock, calls = _install_draft_modules(monkeypatch)
    layer, vllm_config = _build_layer(target, run_dtype)
    assert layer.self_attn.kv_cache_dtype == run_dtype
    assert layer.self_attn.attn_backend == "stock-selector"
    assert calls[0][1] == run_dtype
    assert vllm_config.cache_config.cache_dtype == run_dtype
    assert mla_module.get_attn_backend is stock


@pytest.mark.parametrize("backend_name", ["FLASHMLA", "TRITON_MLA"])
@pytest.mark.parametrize("run_dtype", ["fp8_ds_mla", "auto"])
def test_env_pins_draft_backend_through_platform(monkeypatch, backend_name, run_dtype) -> None:
    import vllm.platforms
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setenv(BACKEND_ENV, backend_name.lower())
    platform = _hcu_platform(monkeypatch)
    monkeypatch.setattr(vllm.platforms, "current_platform", platform, raising=False)
    monkeypatch.setattr(policy, "_resolve_qualname", lambda path: f"class:{path}")
    target, mla_module, stock, calls = _install_draft_modules(monkeypatch)

    layer, vllm_config = _build_layer(target, run_dtype)

    assert layer.self_attn.attn_backend == (
        f"class:{AttentionBackendEnum[backend_name].get_path()}"
    )
    assert layer.self_attn.kv_cache_dtype == "auto"
    assert calls == []  # the cached stock selector was not consulted
    assert mla_module.get_attn_backend is stock
    assert vllm_config.cache_config.cache_dtype == run_dtype


def test_flashmla_pin_obeys_custom_ops_master_fallback(monkeypatch) -> None:
    import vllm.platforms
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv(BACKEND_ENV, "FLASHMLA")
    platform = _hcu_platform(monkeypatch)
    monkeypatch.setattr(vllm.platforms, "current_platform", platform, raising=False)
    monkeypatch.setattr(policy, "_resolve_qualname", lambda path: path)
    target, *_ = _install_draft_modules(monkeypatch)
    layer, _ = _build_layer(target, "fp8_ds_mla")
    assert layer.self_attn.attn_backend == AttentionBackendEnum.TRITON_MLA.get_path()


def test_pin_with_inherited_fp8_ds_mla_surfaces_platform_rejection(monkeypatch) -> None:
    import vllm.platforms

    monkeypatch.setenv(KV_DTYPE_ENV, "inherit")
    monkeypatch.setenv(BACKEND_ENV, "TRITON_MLA")
    platform = _hcu_platform(monkeypatch)
    monkeypatch.setattr(vllm.platforms, "current_platform", platform, raising=False)
    target, mla_module, stock, _calls = _install_draft_modules(monkeypatch)
    with pytest.raises(ValueError, match="kv_cache_dtype not supported"):
        _build_layer(target, "fp8_ds_mla")
    assert mla_module.get_attn_backend is stock


def test_adapter_rejects_foreign_attention_class(monkeypatch) -> None:
    from vllm_hcu.patch.worker.op_opt._common import PatchCompatibilityError

    target = ModuleType(adapter.TARGET_MODULE)

    class MultiHeadLatentAttention:
        pass

    class K3DSparkDecoderLayer:
        def __init__(self, *, vllm_config):
            pass

    target.MultiHeadLatentAttention = MultiHeadLatentAttention
    target.K3DSparkDecoderLayer = K3DSparkDecoderLayer
    with pytest.raises(PatchCompatibilityError, match="MultiHeadLatentAttention"):
        adapter.apply_to_module(target)
