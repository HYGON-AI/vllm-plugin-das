# SPDX-License-Identifier: Apache-2.0
"""CPU format/storage contracts. Public loader/device dispatch is a separate gate."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest


def api():
    name = "hcu_vllm_das.slimquant_packed"
    assert importlib.util.find_spec(name) is not None, "Native packed method is missing"
    return importlib.import_module(name)


def moe(**updates):
    import torch

    cfg = dict(
        moe_parallel_config=SimpleNamespace(
            tp_size=1,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            sp_size=1,
            use_ep=False,
            enable_eplb=False,
        ),
        activation=SimpleNamespace(value="silu"),
        is_act_and_mul=True,
        has_bias=False,
        is_lora_enabled=False,
        has_hash_routing=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        moe_backend="auto",
        num_experts=2,
        num_local_experts=2,
        in_dtype=torch.bfloat16,
    )
    cfg.update(updates)
    return SimpleNamespace(**cfg)


@pytest.mark.parametrize(
    "context,expected",
    [
        (None, "expanded-int8"),
        (SimpleNamespace(additional_config={}), "expanded-int8"),
        (SimpleNamespace(additional_config={"hcu_slimquant": {}}), "expanded-int8"),
        (
            SimpleNamespace(
                additional_config={"hcu_slimquant": {"moe_provider": "boltops"}}
            ),
            "boltops",
        ),
    ],
)
def test_provider_is_selected_only_by_public_hashed_context(context, expected):
    assert api().select_moe_provider(context) == expected


@pytest.mark.parametrize(
    "option",
    [
        "boltops",
        False,
        {"moe_provider": "unknown"},
        {"moe_provider": True},
        {"moe_provder": "boltops"},
    ],
)
def test_malformed_provider_configuration_refuses(option):
    with pytest.raises(ValueError, match="SlimQuant"):
        api().select_moe_provider(
            SimpleNamespace(additional_config={"hcu_slimquant": option})
        )


def test_provider_changes_real_full_vllm_configuration_hash():
    from vllm.config import VllmConfig

    expanded = VllmConfig(
        additional_config={"hcu_slimquant": {"moe_provider": "expanded-int8"}}
    )
    packed = VllmConfig(
        additional_config={"hcu_slimquant": {"moe_provider": "boltops"}}
    )
    assert expanded.compute_hash() != packed.compute_hash()


@pytest.mark.parametrize(
    "field,value",
    [
        ("has_bias", True),
        ("is_lora_enabled", True),
        ("has_hash_routing", True),
        ("swiglu_limit", 7.0),
        ("swiglu_alpha", 1.0),
        ("swiglu_beta", 1.0),
        ("activation_situ_beta", 1.0),
        ("moe_backend", "humming"),
    ],
)
def test_unverified_moe_modes_refuse_before_weights_or_vendor_prepare(field, value):
    with pytest.raises(ValueError, match="packed SlimQuant"):
        api().SlimQuantPackedExperts(moe(**{field: value}))


@pytest.mark.parametrize(
    "field", ["pcp_size", "dp_size", "ep_size", "sp_size", "use_ep", "enable_eplb"]
)
def test_unverified_distributed_modes_refuse(field):
    cfg = moe()
    setattr(cfg.moe_parallel_config, field, 2 if field.endswith("_size") else True)
    with pytest.raises(ValueError, match="packed SlimQuant"):
        api().SlimQuantPackedExperts(cfg)


def storage():
    import torch

    method = api().SlimQuantPackedExperts(moe())
    layer = torch.nn.Module()
    layer.apply_router_weight_on_input = False
    calls = []

    # Only a TP1 destination copier for CPU storage/validation tests. This does
    # not certify the Core RoutedExperts loader or TP slicing.
    def copy(
        param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        calls.append((weight_name, shard_id, expert_id))
        target = param.data[expert_id]
        if shard_id in ("w1", "w3"):
            target = target.chunk(2, dim=0)[0 if shard_id == "w1" else 1]
        target.copy_(loaded_weight)
        return True if return_success else None

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=8,
        intermediate_size_per_partition=4,
        params_dtype=torch.bfloat16,
        weight_loader=copy,
    )
    return method, layer, calls


def load_all(layer, scale=0.125, byte=0x12):
    import torch

    for expert in range(2):
        for attr, shard, shape in [
            ("w13", "w1", (4, 4)),
            ("w13", "w3", (4, 4)),
            ("w2", "w2", (8, 2)),
        ]:
            for suffix, tensor in [
                ("weight", torch.full(shape, byte, dtype=torch.int8)),
                ("weight_scale", torch.full((shape[0], 1), scale, dtype=torch.float32)),
            ]:
                name = f"{attr}_{suffix}"
                p = getattr(layer, name)
                p.weight_loader(p, tensor, name, shard, expert, True)


def test_packed_parameter_geometry_and_raw_scale_domain():
    import torch

    _, layer, _ = storage()
    assert tuple(layer.w13_weight.shape) == (2, 8, 4) and tuple(
        layer.w2_weight.shape
    ) == (2, 8, 2)
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.int8
    assert layer.w13_weight_scale.dtype == layer.w2_weight_scale.dtype == torch.float32
    load_all(layer)
    assert bool((layer.w13_weight == 0x12).all()) and bool(
        (layer.w13_weight_scale == 0.125).all()
    )


def test_postload_refreshes_x16_buffers_without_compounding_or_pointer_changes():
    method, layer, _ = storage()
    load_all(layer)
    method.process_weights_after_loading(layer)
    pointers = [
        getattr(layer, n).data_ptr()
        for n in [
            "w13_weight",
            "w2_weight",
            "w13_scale_for_kernel",
            "w2_scale_for_kernel",
        ]
    ]
    for _ in range(2):
        method.process_weights_after_loading(layer)
        assert bool((layer.w13_weight_scale == 0.125).all())
        assert bool((layer.w13_scale_for_kernel == 2.0).all())
    load_all(layer, scale=0.25, byte=0x34)
    method.process_weights_after_loading(layer)
    assert bool((layer.w13_weight == 0x34).all()) and bool(
        (layer.w13_scale_for_kernel == 4.0).all()
    )
    assert pointers == [
        getattr(layer, n).data_ptr()
        for n in [
            "w13_weight",
            "w2_weight",
            "w13_scale_for_kernel",
            "w2_scale_for_kernel",
        ]
    ]


def test_partial_initial_or_reload_generation_refuses_postload():
    import torch

    method, layer, _ = storage()
    with pytest.raises(ValueError, match="incomplete"):
        method.process_weights_after_loading(layer)
    load_all(layer)
    method.process_weights_after_loading(layer)
    p = layer.w13_weight
    p.weight_loader(
        p, torch.ones((4, 4), dtype=torch.int8), "w13_weight", "w1", 0, True
    )
    with pytest.raises(ValueError, match="incomplete"):
        method.process_weights_after_loading(layer)


def test_official_layerwise_reload_commits_derived_scales_to_capture_addresses():
    """Core must retain newly derived values when restoring stable storage."""
    import torch
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    method, layer, _ = storage()
    layer.quant_method = method
    record_metadata_for_reloading(layer)
    load_all(layer)
    method.process_weights_after_loading(layer)
    original = dict(layer.named_parameters()) | dict(layer.named_buffers())
    pointers = {name: tensor.data_ptr() for name, tensor in original.items()}
    for scale, byte in ((0.5, 0x34), (0.25, 0x12)):
        initialize_layerwise_reload(layer)
        load_all(layer, scale=scale, byte=byte)
        finalize_layerwise_reload(layer, model_config=None)
        assert layer._slimquant_ready
        for stem in ("w13", "w2"):
            canonical = getattr(layer, f"{stem}_weight_scale")
            derived = getattr(layer, f"{stem}_scale_for_kernel")
            assert torch.equal(canonical, torch.full_like(canonical, scale))
            assert torch.equal(derived, canonical * 16)
            assert f"{stem}_scale_for_kernel" not in layer.state_dict()
        for name, tensor in original.items():
            assert getattr(layer, name) is tensor
            assert tensor.data_ptr() == pointers[name]


@pytest.mark.parametrize("operation", ["kernel_reload", "weight_transfer"])
def test_loaded_native_policy_refuses_unverified_updates_without_changing_state(
    operation,
):
    import torch
    from vllm.model_executor.model_loader.reload.utils import (
        validate_model_weight_update,
    )

    method, layer, _ = storage()
    layer.quant_method = method
    load_all(layer)
    method.process_weights_after_loading(layer)
    tensors = dict(layer.named_parameters()) | dict(layer.named_buffers())
    before = {name: value.clone() for name, value in tensors.items()}
    with pytest.raises(NotImplementedError, match="packed SlimQuant"):
        validate_model_weight_update(layer, operation=operation)
    assert layer._slimquant_ready
    for name, value in before.items():
        assert getattr(layer, name) is tensors[name]
        assert torch.equal(getattr(layer, name), value)


@pytest.mark.parametrize(
    "kind,dtype", [("weight", "float32"), ("weight_scale", "float16")]
)
def test_wrong_dtype_is_rejected_before_destination_copy(kind, dtype):
    import torch

    _, layer, calls = storage()
    p = getattr(layer, "w13_" + kind)
    data = torch.ones(
        (4, 4) if kind == "weight" else (4, 1), dtype=getattr(torch, dtype)
    )
    with pytest.raises(ValueError, match="dtype"):
        p.weight_loader(p, data, "w13_" + kind, "w1", 0, True)
    assert not calls


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale_refuses_before_destination_copy(value):
    import torch

    _, layer, calls = storage()
    p = layer.w13_weight_scale
    with pytest.raises(ValueError, match="positive finite"):
        p.weight_loader(p, torch.full((4, 1), value), "w13_weight_scale", "w1", 0, True)
    assert not calls


def test_finite_raw_scale_that_overflows_compensation_is_refused():
    import torch

    method, layer, _ = storage()
    with pytest.raises(ValueError, match="finite"):
        load_all(layer, scale=torch.finfo(torch.float32).max / 2)
        method.process_weights_after_loading(layer)
    assert not layer._slimquant_ready


def test_official_opaque_additional_config_keeps_unrequested_native_disabled():
    from vllm.config import VllmConfig

    class OpaqueConfig:
        def compute_hash(self):
            return "opaque-platform-options"

    context = VllmConfig(additional_config=OpaqueConfig())
    assert isinstance(context.compute_hash(), str)
    assert api().select_moe_provider(context) == "expanded-int8"
