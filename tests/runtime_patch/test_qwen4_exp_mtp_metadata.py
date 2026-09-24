# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.platform.framework_opt import (
    patch_qwen4_exp_mtp_kv_cache_groups as kv_groups_patch,
    patch_scheduler as scheduler_patch,
)
from vllm_hcu.patch.worker.framework_opt import (
    patch_qwen4_exp_qsa_metadata as qsa_metadata_patch,
)


def _get_kv_cache_config_from_groups(
    vllm_config,
    kv_cache_groups,
    available_memory,
):
    del vllm_config, available_memory
    return SimpleNamespace(
        kv_cache_groups=kv_cache_groups,
        kv_cache_tensors=[],
    )


class _DummyUniformTypeKVCacheSpecs:
    pass


def _patched_kv_config_module(get_config, uniform_type):
    module = ModuleType(kv_groups_patch.TARGET_MODULE)

    def _annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        kv_cache_groups,
        use_deepseek_v4_fallback=False,
    ):
        del (
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
            use_deepseek_v4_fallback,
        )

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        del vllm_config, kv_cache_spec
        return []

    def _warn_if_unannotated_eagle_mamba(vllm_config, kv_cache_groups):
        del vllm_config, kv_cache_groups

    module._annotate_eagle_groups = _annotate_eagle_groups
    module._warn_if_unannotated_eagle_mamba = (
        _warn_if_unannotated_eagle_mamba
    )
    module.get_kv_cache_groups = get_kv_cache_groups
    module.get_kv_cache_config_from_groups = get_config
    module.UniformTypeKVCacheSpecs = uniform_type
    assert kv_groups_patch.apply_to_module(module) is True
    return module


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

    def _warn_if_unannotated_eagle_mamba(vllm_config, kv_cache_groups):
        del vllm_config, kv_cache_groups

    module._annotate_eagle_groups = _annotate_eagle_groups
    module._warn_if_unannotated_eagle_mamba = (
        _warn_if_unannotated_eagle_mamba
    )
    module.get_kv_cache_groups = get_kv_cache_groups
    module.get_kv_cache_config_from_groups = _get_kv_cache_config_from_groups
    module.UniformTypeKVCacheSpecs = _DummyUniformTypeKVCacheSpecs
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
    warning_observations = []

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
            layer_names=[],
            kv_cache_spec=SimpleNamespace(
                kv_cache_specs={"mtp.layers.48.self_attn": SimpleNamespace()}
            ),
            is_eagle_group=False,
        ),
    ]

    def _warn_if_unannotated_eagle_mamba(vllm_config, kv_cache_groups):
        del vllm_config
        warning_observations.append(
            [group.is_eagle_group for group in kv_cache_groups]
        )

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        del kv_cache_spec
        module._warn_if_unannotated_eagle_mamba(vllm_config, groups)
        return groups

    module._annotate_eagle_groups = _annotate_eagle_groups
    module._warn_if_unannotated_eagle_mamba = (
        _warn_if_unannotated_eagle_mamba
    )
    module.get_kv_cache_groups = get_kv_cache_groups
    module.get_kv_cache_config_from_groups = _get_kv_cache_config_from_groups
    module.UniformTypeKVCacheSpecs = _DummyUniformTypeKVCacheSpecs
    assert kv_groups_patch.apply_to_module(module) is True

    qwen_config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen4_exp")
        ),
    )
    returned_groups = module.get_kv_cache_groups(qwen_config, {"layers": "spec"})

    assert returned_groups is groups
    assert warning_observations == [[False, True]]
    assert [group.is_eagle_group for group in groups] == [False, True]


def test_qwen_mtp_pp_drops_tensors_from_empty_projected_uniform_groups():
    """PP workers must not allocate tensors owned only by another stage."""

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        UniformTypeKVCacheSpecs,
    )

    module = ModuleType(kv_groups_patch.TARGET_MODULE)

    def _annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        kv_cache_groups,
        use_deepseek_v4_fallback=False,
    ):
        del (
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
            use_deepseek_v4_fallback,
        )

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        del vllm_config, kv_cache_spec
        return []

    def _warn_if_unannotated_eagle_mamba(vllm_config, kv_cache_groups):
        del vllm_config, kv_cache_groups

    module._annotate_eagle_groups = _annotate_eagle_groups
    module._warn_if_unannotated_eagle_mamba = (
        _warn_if_unannotated_eagle_mamba
    )
    module.get_kv_cache_groups = get_kv_cache_groups
    module.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs

    remote_layer = "model.layers.0.linear_attn"
    local_layer = "model.layers.24.linear_attn"
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    groups = [
        KVCacheGroupSpec(
            layer_names=[],
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs={remote_layer: spec},
            ),
        ),
        KVCacheGroupSpec(layer_names=[local_layer], kv_cache_spec=spec),
    ]

    def get_kv_cache_config_from_groups(
        vllm_config,
        kv_cache_groups,
        available_memory,
    ):
        del vllm_config, available_memory
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=4096,
                    layers=[remote_layer],
                    layer_stride=4096,
                    block_stride=512,
                ),
                KVCacheTensor(
                    size=4096,
                    layers=[local_layer],
                    layer_stride=4096,
                    block_stride=512,
                ),
            ],
            kv_cache_groups=kv_cache_groups,
        )

    module.get_kv_cache_config_from_groups = get_kv_cache_config_from_groups
    assert kv_groups_patch.apply_to_module(module) is True

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen4_exp")
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    result = module.get_kv_cache_config_from_groups(config, groups, 1 << 30)

    assert [tensor.layers for tensor in result.kv_cache_tensors] == [[local_layer]]
    assert result.kv_cache_groups is groups
    assert result.kv_cache_groups[0] is groups[0]
    assert result.kv_cache_groups[1] is groups[1]


def test_qwen_mtp_pp_restores_draft_marker_on_empty_projected_group():
    """The scheduler must still identify MTP when PP0 owns no draft layer."""

    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        UniformTypeKVCacheSpecs,
    )

    mtp_layer = "mtp.layers.48.self_attn"
    remote_target_layer = "model.layers.24.self_attn"
    local_target_layer = "model.layers.0.self_attn"
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    groups = [
        KVCacheGroupSpec(
            layer_names=[],
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs={mtp_layer: spec},
            ),
        ),
        KVCacheGroupSpec(
            layer_names=[],
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs={remote_target_layer: spec},
            ),
        ),
        KVCacheGroupSpec(layer_names=[local_target_layer], kv_cache_spec=spec),
    ]

    def get_config(vllm_config, kv_cache_groups, available_memory):
        del vllm_config, available_memory
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=4096,
                    layers=[layer_name],
                    layer_stride=4096,
                    block_stride=512,
                )
                for layer_name in (
                    mtp_layer,
                    remote_target_layer,
                    local_target_layer,
                )
            ],
            kv_cache_groups=kv_cache_groups,
        )

    module = _patched_kv_config_module(get_config, UniformTypeKVCacheSpecs)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen4_exp")
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )

    result = module.get_kv_cache_config_from_groups(config, groups, 1 << 30)
    scheduler_config = generate_scheduler_kv_cache_config([result])

    assert [group.is_eagle_group for group in scheduler_config.kv_cache_groups] == [
        True,
        False,
        False,
    ]
    assert [tensor.layers for tensor in result.kv_cache_tensors] == [
        [local_target_layer]
    ]


def test_qwen_mtp_pp_rejects_unclassified_unowned_tensors():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        UniformTypeKVCacheSpecs,
    )

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    groups = [KVCacheGroupSpec(layer_names=[], kv_cache_spec=spec)]

    def get_config(vllm_config, kv_cache_groups, available_memory):
        del vllm_config, available_memory
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=4096,
                    layers=["unknown.remote.layer"],
                    layer_stride=4096,
                    block_stride=512,
                )
            ],
            kv_cache_groups=kv_cache_groups,
        )

    module = _patched_kv_config_module(get_config, UniformTypeKVCacheSpecs)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen4_exp")
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )

    with pytest.raises(RuntimeError, match="unowned KV tensor layers"):
        module.get_kv_cache_config_from_groups(config, groups, 1 << 30)


def test_qwen_mtp_pp_rejects_tensor_mixing_local_and_remote_layers():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        UniformTypeKVCacheSpecs,
    )

    remote_layer = "model.layers.0.linear_attn"
    local_layer = "model.layers.24.linear_attn"
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    groups = [
        KVCacheGroupSpec(
            layer_names=[],
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs={remote_layer: spec},
            ),
        ),
        KVCacheGroupSpec(layer_names=[local_layer], kv_cache_spec=spec),
    ]

    def get_config(vllm_config, kv_cache_groups, available_memory):
        del vllm_config, available_memory
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=4096,
                    layers=[remote_layer, local_layer],
                    layer_stride=2048,
                    block_stride=512,
                )
            ],
            kv_cache_groups=kv_cache_groups,
        )

    module = _patched_kv_config_module(get_config, UniformTypeKVCacheSpecs)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen4_exp")
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )

    with pytest.raises(RuntimeError, match="mixes local and remote"):
        module.get_kv_cache_config_from_groups(config, groups, 1 << 30)


@pytest.mark.parametrize(
    ("pipeline_parallel_size", "use_block_drop", "model_type"),
    [
        (1, True, "qwen4_exp"),
        (2, False, "qwen4_exp"),
        (2, True, "deepseek_v4"),
    ],
)
def test_kv_config_filter_is_exact_noop_outside_qwen_mtp_pp(
    pipeline_parallel_size,
    use_block_drop,
    model_type,
):
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    sentinel_tensor = SimpleNamespace(layers=["unclassified.layer"])
    original_config = SimpleNamespace(
        kv_cache_groups=[],
        kv_cache_tensors=[sentinel_tensor],
    )

    def get_config(vllm_config, kv_cache_groups, available_memory):
        del vllm_config, kv_cache_groups, available_memory
        return original_config

    module = _patched_kv_config_module(get_config, UniformTypeKVCacheSpecs)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            use_eagle_block_drop=lambda: use_block_drop
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type)
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size
        ),
    )

    result = module.get_kv_cache_config_from_groups(config, [], 1 << 30)

    assert result is original_config
    assert result.kv_cache_tensors == [sentinel_tensor]


def test_qwen_mtp_kv_config_patch_rejects_stale_wrapper_marker():
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    module = _patched_kv_config_module(
        _get_kv_cache_config_from_groups,
        UniformTypeKVCacheSpecs,
    )
    module.get_kv_cache_config_from_groups = _get_kv_cache_config_from_groups

    with pytest.raises(kv_groups_patch.PatchCompatibilityError, match="stale"):
        kv_groups_patch.apply_to_module(module)


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
