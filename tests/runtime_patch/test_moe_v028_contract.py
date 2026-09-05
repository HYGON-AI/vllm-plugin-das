import inspect

from vllm_hcu.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExperts,
    FusedMoEExpertsModular,
)


def test_hcu_fused_moe_experts_default_to_canonical_expert_map() -> None:
    assert FusedMoEExperts.consumes_expert_mask is False
    source = inspect.getsource(FusedMoEExperts.__init__)
    assert "ApplyMoEActivationConfig.from_configs" in source
    assert tuple(inspect.signature(FusedMoEExpertsModular.activation).parameters) == (
        "self",
        "activation",
        "output",
        "input",
        "topk_ids",
        "expert_map",
        "valid_rows",
    )
