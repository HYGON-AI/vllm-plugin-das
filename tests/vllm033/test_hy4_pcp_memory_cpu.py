# SPDX-License-Identifier: Apache-2.0
"""Exercise installed constructor/proposer bodies without native GPU imports."""

import __future__

import ast
from importlib.metadata import distribution
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def installed_class(filename, name, scope, *, method="propose"):
    path = distribution("hcu-vllm-das-plugin").locate_file(
        "hcu_vllm_das/models/hy_v4_flash/" + filename
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if name == "HYV4DFlySpeculator":
        cls.bases = (
            [ast.Name(id="DraftModelSpeculator", ctx=ast.Load())]
            if method == "__init__"
            else []
        )
        cls.body = [
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method
        ]
        cls.body[0].decorator_list = []
    scope["__package__"] = "hcu_vllm_das.models.hy_v4_flash"
    exec(  # noqa: S102
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    return scope[name]


def reuse_inputs():
    class Attention:
        def __init__(self, name):
            self.indexer = object()
            self.skip_topk = False
            self.topk_indices_buffer = torch.zeros(8, 4, dtype=torch.int32)
            self.topk_snapshot = None
            self.num_local_heads = 2
            self.learnable_sink = False
            self.rotary_emb = object()
            self.mla_attn = NS(
                layer_name=name,
                get_kv_cache_spec=lambda _: spec,
                get_attn_backend=lambda: object(),
            )

    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    target = NS(
        model=NS(layers=[NS(self_attn=Attention(f"target.{i}")) for i in range(5)])
    )
    draft = NS(
        config=NS(
            target_index_layer=[1, 3],
            target_kv_layers=[2, 4],
            target_index_topk=4,
            context_local_window=None,
        ),
        model=NS(layers=[NS(self_attn=Attention(f"draft.{i}")) for i in range(2)]),
    )
    scope = {
        "torch": torch,
        "HYV4MLAAttention": Attention,
        "MLAAttentionSpec": MLAAttentionSpec,
        "customize_attention_spec": lambda _, value: value,
        "validate_dfly_rotary_pair": lambda *_: None,
        "replace": __import__("dataclasses").replace,
    }
    cls = installed_class("dfly_target_reuse.py", "HYV4DFlyTargetReuse", scope)
    return cls, target, draft, NS(cache_config=NS(cache_dtype="auto"))


def test_default_target_reuse_still_captures_every_index_layer():
    cls, target, draft, config = reuse_inputs()
    reuse = cls(target, draft, config)
    assert reuse.snapshots.shape == (2, 8, 4)
    assert all(
        target.model.layers[i].self_attn.topk_snapshot is not None for i in (1, 3)
    )
    assert (
        draft.model.layers[0].self_attn.mla_attn.kv_sharing_target_layer_name
        == "target.2"
    )


def test_prefill_only_reuse_keeps_aliases_without_allocating_or_publishing_snapshots():
    cls, target, draft, config = reuse_inputs()
    reuse = cls(target, draft, config, capture_target_indices=False)
    assert reuse.snapshots.numel() == 0
    assert all(target.model.layers[i].self_attn.topk_snapshot is None for i in (1, 3))
    assert reuse.aliases == {"draft.0": "target.2", "draft.1": "target.4"}
    assert (
        draft.model.layers[1].self_attn.mla_attn.kv_sharing_target_layer_name
        == "target.4"
    )
    with pytest.raises(ValueError, match="prefill-only"):
        reuse.update_indices(None, None, None, 0)


@pytest.mark.parametrize("is_profile", [False, True])
def test_real_producer_dummy_proposer_returns_before_unused_draft_execution(is_profile):
    cls = installed_class("dfly_speculator.py", "HYV4DFlySpeculator", {})
    owner = cls.__new__(cls)
    owner._hy4_pd_role = "producer"
    owner.max_num_reqs = 4
    owner.num_query_per_req = 7
    owner.max_num_tokens = 28
    owner.dp_size, owner.dp_rank = 1, 0
    owner.draft_tokens = torch.zeros(4, 6, dtype=torch.int64)
    owner.reuse = None  # Entering the actual query path would fail here.
    result = owner.propose(
        NS(num_reqs=2),
        *([None] * 10),
        dummy_run=True,
        is_profile=is_profile,
        num_speculative_tokens=6,
    )
    assert result.shape == (4 if is_profile else 2, 6)
    assert torch.all(result == -1)


@pytest.mark.parametrize(
    "role, target_budget, expected",
    [
        ("producer", 98304, 1792),
        ("producer", 32768, 1792),
        ("decode", 256, 256),
        ("decode", 32, 56),
    ],
)
def test_real_speculator_constructor_sizes_private_token_capacity(
    role, target_budget, expected
):
    from copy import copy

    captured = []

    class ReachedCoreConstructor(Exception):
        pass

    class Base:
        def __init__(self, config, device):
            captured.append(config)
            raise ReachedCoreConstructor

    scope = {
        "DraftModelSpeculator": Base,
        "copy": copy,
        "_admits_hisparse_graph": lambda _: False,
    }
    cls = installed_class(
        "dfly_speculator.py", "HYV4DFlySpeculator", scope, method="__init__"
    )
    producer = role == "producer"
    config = NS(
        attention_config=NS(hisparse_config=None),
        cache_config=NS(cache_dtype="auto"),
        speculative_config=NS(kv_cache_dtype=None, num_speculative_tokens=6),
        model_config=NS(
            enforce_eager=True,
            hf_config=NS(
                model_type="hy_v4_vl",
                architectures=["HYV4VLForConditionalGeneration"],
                text_config=NS(model_type="hy_v4"),
            ),
        ),
        kv_transfer_config=NS(
            kv_connector="MooncakeConnector",
            kv_connector_module_path="hcu_vllm_das.hy4_pcp_mooncake",
            kv_role="kv_producer" if producer else "kv_consumer",
        ),
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=16 if producer else 1,
            decode_context_parallel_size=1,
            data_parallel_size=1 if producer else 16,
            data_parallel_rank=0,
            enable_expert_parallel=True,
            all2all_backend="deepep_high_throughput",
            enable_dbo=False,
            enable_eplb=False,
            enable_elastic_ep=False,
            use_ubatching=False,
        ),
        lora_config=None,
        compilation_config=NS(cudagraph_mode=0),
        scheduler_config=NS(
            max_num_batched_tokens=target_budget, max_num_seqs=256 if producer else 8
        ),
    )
    with pytest.raises(ReachedCoreConstructor):
        cls(config, torch.device("cpu"))
    assert config.scheduler_config.max_num_batched_tokens == target_budget
    assert captured[0].scheduler_config is not config.scheduler_config
    assert captured[0].scheduler_config.max_num_batched_tokens == expected
