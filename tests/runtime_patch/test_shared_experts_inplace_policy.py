# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch

from vllm_hcu.model_executor.layers.fused_moe.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


class _PolicyOnlySharedExperts(SharedExperts):
    def __init__(self, order: SharedExpertsOrder) -> None:
        torch.nn.Module.__init__(self)
        self._order = order

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        del hidden_states
        return self._order


def test_shared_experts_inplace_policy_preserves_overlapped_alias() -> None:
    routed_input = torch.zeros((2, 4))
    aliased_shared_input = routed_input.view_as(routed_input)
    independent_shared_input = routed_input.clone()

    overlapped = _PolicyOnlySharedExperts(
        SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    )
    serial = _PolicyOnlySharedExperts(SharedExpertsOrder.NO_OVERLAP)

    assert overlapped.requires_input_preservation(aliased_shared_input)
    assert not overlapped.allows_inplace_routed_output(
        routed_input,
        aliased_shared_input,
    )
    assert overlapped.allows_inplace_routed_output(
        routed_input,
        independent_shared_input,
    )
    assert serial.allows_inplace_routed_output(
        routed_input,
        aliased_shared_input,
    )
