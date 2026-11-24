"""Verify that expanded tuning remains local to a masked expert workspace."""
from hcu_vllm_das.moe.minimax31_ll import MaskedExperts


def test_expanded_verify_does_not_replace_a_class_method_or_second_workspace():
    original_call = MaskedExperts.__call__
    original_gemm, tuned_gemm = object(), object()
    first, second = (object.__new__(MaskedExperts) for _ in range(2))
    for workspace in (first, second):
        workspace.geometry = (16, 6144, 3072, 1024)
        workspace.gemm = original_gemm
        workspace.verify_gemm_config = {"original": True}
        workspace._verify_gemm_rows = (128,)
        workspace._activation_tuned_rows = (64, 128)

    def bind(gemm, config):
        assert gemm is original_gemm
        assert config == {"MODE": 1642, "PERSISTENT_MULTIPLIER": 5}
        return tuned_gemm

    first.configure_expanded_verify(80, bind)
    assert MaskedExperts.__call__ is original_call
    assert first.gemm is tuned_gemm
    assert first._verify_gemm_rows == first._activation_tuned_rows == (80,)
    assert second.gemm is original_gemm
    assert second._verify_gemm_rows == (128,)
    assert second._activation_tuned_rows == (64, 128)
    assert second.verify_gemm_config == {"original": True}
