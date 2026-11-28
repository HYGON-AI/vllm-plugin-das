# SPDX-License-Identifier: Apache-2.0
"""Explicit VL identity, never generic M-RoPE metadata, selects the new target."""

from hcu_vllm_das.models.hy4_standard.target_policy import uses_standard_target
from vllm.transformers_utils.configs.hy_v4 import HYV4Config


def test_plain_text_mrope_does_not_select_the_eager_only_target():
    text = HYV4Config(
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [12, 10, 10],
        }
    )
    assert uses_standard_target(text, method=None) is False
    assert uses_standard_target(text, method="mtp") is False
    assert uses_standard_target(text, method="dfly") is True


def test_explicit_owned_vl_text_marker_selects_standard_target():
    text = HYV4Config(hy4_standard_vl=True)
    assert uses_standard_target(text, method=None) is True
    assert (
        uses_standard_target(type(text).from_dict(text.to_dict()), method=None) is True
    )
