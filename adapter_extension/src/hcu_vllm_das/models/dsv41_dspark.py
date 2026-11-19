# SPDX-License-Identifier: Apache-2.0
"""Private AG-RS construction for the HCU V41 draft model.

The native loader and the compressed-tensors scale-alias load view belong to the
adapter (`hcu_adapter.dsv41_draft_weight_alias`): they are the naming bridge the
distinct channel-FP8 route creates, which is an adapter route this plugin does
not own. What stays here is the instance-owned AG-RS construction config, which
depends on this plugin's own communicator, so the class overrides `__init__`
only and inherits enumeration/loading unchanged.
"""
from __future__ import annotations

from ..adapter_api import adapter_module as _adapter_module

HcuDSV41DraftWeightAliasModel = _adapter_module('dsv41_draft_weight_alias').HcuDSV41DraftWeightAliasModel


class HcuDSparkDeepseekV41ForCausalLM(HcuDSV41DraftWeightAliasModel):
    """Keep native draft construction/loading; own CT DP communication policy."""

    def __init__(self, *, vllm_config, prefix=""):
        from vllm.config import set_current_vllm_config
        from hcu_vllm_das.dspark_agrs import _owned_quant_kind, draft_config
        from hcu_vllm_das.dspark_backend import native_draft_ll_requested
        from hcu_vllm_das.dspark_post import requested, initialize_owned

        native_ll = native_draft_ll_requested(vllm_config)
        constructor = super().__init__
        if requested(vllm_config):
            constructor = lambda **kwargs: initialize_owned(self, **kwargs)

        target_parallel = getattr(vllm_config.speculative_config, 'target_parallel_config', None)

        # CT W4A8 and Channel-FP8 drafts both take a private AG-RS construction
        # config so they do not allocate a second DeepEP/RocSHMEM LL buffer on
        # top of the target's already-resident weights.
        if (
            _owned_quant_kind(vllm_config) is not None
            and vllm_config.parallel_config.data_parallel_size > 1
            and not native_ll
        ):
            local = draft_config(vllm_config)
            # Native draft decoder construction reads current_config as well as
            # its explicit argument. Keep both references on the same private
            # copy and retain native compiler live tables for layer discovery.
            with set_current_vllm_config(local):
                constructor(vllm_config=local, prefix=prefix)
        elif (
            _owned_quant_kind(vllm_config) is not None
            and getattr(target_parallel, 'prefill_context_parallel_size', 1) > 1
        ):
            from ..pcp_dspark_draft import draft_config as pcp_draft_config

            local = pcp_draft_config(vllm_config)
            with set_current_vllm_config(local):
                constructor(vllm_config=local, prefix=prefix)
        else:
            constructor(vllm_config=vllm_config, prefix=prefix)
        if native_ll:
            from hcu_vllm_das.dspark_ll import install_model_entry_reset
            install_model_entry_reset(self, vllm_config, 128)
        self._hcu_markov_pointwise = native_ll

    def sample_sequential(self, speculator, num_reqs, head_hidden):
        from hcu_vllm_das.dspark_greedy_admission import sample_sequential
        if sample_sequential(self, speculator, num_reqs, head_hidden):
            return True
        return super().sample_sequential(speculator, num_reqs, head_hidden)
