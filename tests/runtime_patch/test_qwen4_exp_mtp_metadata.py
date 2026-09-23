# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.platform.framework_opt import (
    patch_qwen4_exp_mtp_kv_cache_groups as kv_groups_patch,
    patch_scheduler as scheduler_patch,
)
from vllm_hcu.patch.worker.framework_opt import (
    patch_qwen4_exp_qsa_metadata as qsa_metadata_patch,
)


def test_qsa_draft_metadata_refresh_reuses_official_builder_in_place():
    module = ModuleType(qsa_metadata_patch.TARGET_MODULE)

    class BaseBuilder:
        supports_draft_decode_metadata_update = False

        def update_draft_decode_metadata(self, metadata):
            raise NotImplementedError

    class QSAMetadataBuilder(BaseBuilder):
        def __init__(self):
            self.build_count = 0

        def build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build=False,
        ):
            self.build_count += 1
            return SimpleNamespace(
                generation=self.build_count,
                common=common_attn_metadata,
                fast_build=fast_build,
            )

    module.QSAMetadataBuilder = QSAMetadataBuilder

    assert qsa_metadata_patch.apply_to_module(module) is True
    assert qsa_metadata_patch.apply_to_module(module) is False
    assert QSAMetadataBuilder.supports_draft_decode_metadata_update is True

    builder = QSAMetadataBuilder()
    common = SimpleNamespace(seq_lens="persistent-seq-lens")
    metadata = builder.build(7, common)
    metadata_id = id(metadata)

    builder.update_draft_decode_metadata(metadata)

    assert id(metadata) == metadata_id
    assert metadata.generation == 2
    assert metadata.common is common
    assert metadata.fast_build is True


@pytest.mark.parametrize(
    "model_type",
    [
        "qwen4_exp",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    ],
)
def test_qwen_mtp_groups_are_annotated_without_target_mamba_groups(model_type):
    module = ModuleType(kv_groups_patch.TARGET_MODULE)
    original_calls = []

    def _annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        kv_cache_groups,
        use_deepseek_v4_fallback=False,
    ):
        original_calls.append(
            (vllm_config, kv_cache_spec, use_deepseek_v4_fallback)
        )

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        return []

    module._annotate_eagle_groups = _annotate_eagle_groups
    module.get_kv_cache_groups = get_kv_cache_groups
    assert kv_groups_patch.apply_to_module(module) is True
    assert kv_groups_patch.apply_to_module(module) is False

    spec_config = SimpleNamespace(use_eagle_block_drop=lambda: True)
    qwen_config = SimpleNamespace(
        speculative_config=spec_config,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
    )
    groups = [
        SimpleNamespace(
            layer_names=["model.layers.0.linear_attn"], is_eagle_group=False
        ),
        SimpleNamespace(
            layer_names=["model.layers.3.self_attn"], is_eagle_group=False
        ),
        SimpleNamespace(
            layer_names=["mtp.layers.48.self_attn"], is_eagle_group=False
        ),
        SimpleNamespace(
            layer_names=["mtp.layers.48.self_attn.indexer.raw_key_cache"],
            is_eagle_group=False,
        ),
    ]

    module._annotate_eagle_groups(qwen_config, {"layers": "spec"}, groups)

    assert [group.is_eagle_group for group in groups] == [False, False, True, True]
    assert len(original_calls) == 1

    other_config = SimpleNamespace(
        speculative_config=spec_config,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v4")
        ),
    )
    other_group = SimpleNamespace(
        layer_names=["mtp.layers.1.self_attn"], is_eagle_group=False
    )
    module._annotate_eagle_groups(other_config, {}, [other_group])
    assert other_group.is_eagle_group is False


def test_qwen4_exp_mtp_groups_are_annotated_after_upstream_early_return():
    module = ModuleType(kv_groups_patch.TARGET_MODULE)

    def _annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        kv_cache_groups,
        use_deepseek_v4_fallback=False,
    ):
        raise AssertionError("the upstream early-return path must bypass this hook")

    groups = [
        SimpleNamespace(
            layer_names=["model.layers.0.linear_attn"],
            kv_cache_spec=SimpleNamespace(),
            is_eagle_group=False,
        ),
        SimpleNamespace(
            layer_names=["mtp.layers.48.self_attn"],
            kv_cache_spec=SimpleNamespace(),
            is_eagle_group=False,
        ),
    ]

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        return groups

    module._annotate_eagle_groups = _annotate_eagle_groups
    module.get_kv_cache_groups = get_kv_cache_groups
    assert kv_groups_patch.apply_to_module(module) is True

    qwen_config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen4_exp")
        ),
    )
    returned_groups = module.get_kv_cache_groups(qwen_config, {"layers": "spec"})

    assert returned_groups is groups
    assert [group.is_eagle_group for group in groups] == [False, True]


def test_scheduler_uses_hybrid_block_size_only_inside_upstream_mamba_split():
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    module = ModuleType(scheduler_patch.TARGET_MODULE)
    observed_block_sizes = []

    class MambaSpec:
        def __init__(self, block_size):
            self.block_size = block_size

    class Scheduler:
        def update_draft_token_ids(self, draft_token_ids):
            del self, draft_token_ids

        def update_draft_token_ids_in_output(
            self, draft_token_ids, scheduler_output
        ):
            del self, draft_token_ids, scheduler_output

        def _update_after_schedule(self, scheduler_output):
            del self, scheduler_output

        def __init__(
            self,
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
        ):
            del vllm_config, kv_cache_config, structured_output_manager
            self.block_size = block_size
            self.cache_config = SimpleNamespace(block_size=64)

        def _mamba_block_aligned_split(self, request, num_new_tokens):
            del request
            observed_block_sizes.append(self.cache_config.block_size)
            return num_new_tokens - 1

        def _update_after_schedule(self, scheduler_output):
            del self, scheduler_output

    module.Scheduler = Scheduler
    module.MambaSpec = MambaSpec
    original_init = Scheduler.__init__
    assert scheduler_patch.apply_to_module(module) is True
    assert scheduler_patch.apply_to_module(module) is False
    assert Scheduler.__init__ is not original_init

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_moe_text")
        ),
    )
    groups = [
        SimpleNamespace(
            layer_names=["model.layers.0.linear_attn"],
            kv_cache_spec=SimpleNamespace(block_size=64),
            is_eagle_group=False,
        ),
        SimpleNamespace(
            layer_names=["mtp.layers.0.self_attn"],
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=832,
                kv_cache_specs={
                    "mtp.layers.0.self_attn": MambaSpec(block_size=832),
                    "mtp.layers.1.self_attn": MambaSpec(block_size=832),
                },
            ),
            is_eagle_group=False,
        ),
    ]
    scheduler = module.Scheduler(
        config,
        SimpleNamespace(kv_cache_groups=groups),
        None,
        1664,
    )

    assert [group.is_eagle_group for group in groups] == [False, False]
    assert scheduler._mamba_block_aligned_split(object(), 128) == 127
    assert scheduler.block_size == 1664
    assert observed_block_sizes == [832]
    assert scheduler.cache_config.block_size == 64

    no_mtp_config = SimpleNamespace(
        speculative_config=None,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_moe_text")
        ),
    )
    no_mtp_scheduler = module.Scheduler(
        no_mtp_config,
        SimpleNamespace(kv_cache_groups=groups),
        None,
        1664,
    )

    assert no_mtp_scheduler._mamba_block_aligned_split(object(), 128) == 127
    assert observed_block_sizes == [832, 832]
    assert no_mtp_scheduler.cache_config.block_size == 64
