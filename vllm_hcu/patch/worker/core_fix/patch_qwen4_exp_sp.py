# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Support Qwen4Exp HC layers under sequence-parallel MoE.

``use_sequence_parallel_moe`` (DP>1, TP>1, EP, AGRS-class backends) makes the
AMD Qwen4Exp constructors raise.  This patch always removes those guards; the
MoE block then shards TP-replicated tokens itself.  With
``VLLM_HCU_QWEN4_EXP_HC_SP=1`` it additionally ports the HC sequence-parallel
layout of upstream vLLM PR #56322 to the AMD model:

* the embedding is sharded across TP ranks right after lookup, so every HC
  mix/combine runs on the local ``ceil(T / tp)`` rows only;
* attention all-gathers its block input and reduce-scatters its partial
  output (``reduce_results=False``) instead of all-reducing;
* MoE consumes the local rows directly (``already_sequence_parallel=True``);
* the PLE layer runs unchanged on the all-gathered state and its output is
  sharded back (PLE is a single early layer, so the extra gather is cheap);
* the final mixer outputs are all-gathered once (packed with the MTP state).

The MTP drafter counterpart lives in ``patch_qwen4_exp_sp_mtp``.  Every
replaced function is a fingerprint-checked copy of the audited source with
only the SP branches added; with HC SP disabled it executes the original code
path, so other parallel layouts are unaffected.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import textwrap
from itertools import islice
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_exact_signature,
)
from .patch_qwen4_exp_ple_prefetch import _require_model_init_compatible

TARGET_MODULE = "vllm.models.qwen4_exp.amd.model"
PATCH_ID = "worker.core_fix.qwen4_exp.sequence_parallel_moe"
TARGETS = (
    f"{TARGET_MODULE}.Qwen4ExpSparseMoeBlock.__init__",
    f"{TARGET_MODULE}.Qwen4ExpDecoderLayer.__init__",
    f"{TARGET_MODULE}.Qwen4ExpDecoderLayer.forward",
    f"{TARGET_MODULE}.Qwen4ExpModel.__init__",
    f"{TARGET_MODULE}.Qwen4ExpModel.forward",
)
HC_SP_ATTR = "_vllm_hcu_hc_sp"
_MARKER = "_vllm_hcu_qwen4_exp_sp_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_sp_wrapper"
_SOURCE_SHA256 = {
    "Qwen4ExpSparseMoeBlock.__init__": (
        "9e84d4d730d14443c55886ea7d75fd76d16ce61a59f92774c0ca4b20048af404"
    ),
    "Qwen4ExpDecoderLayer.__init__": (
        "074a90fb40855a585a08d7b125b12df63ff3ddc6759248e45d628bc9b5572c19"
    ),
    "Qwen4ExpDecoderLayer.forward": (
        "ad9b8c646d8e4d4b53f561763d6534c9bfd1da2c50ead50c6225fc493d2a5337"
    ),
    "Qwen4ExpModel.forward": (
        "c930baedccbae6fb9fadcc70a351a56a6b7bcb7130362e7ab4d9652ed70002a1"
    ),
}

# Runtime names used by the replacement forwards.  They are bound when the
# patch is applied so that registering this adapter imports nothing from vLLM.
envs = None
get_forward_context = None
is_forward_context_available = None
get_tensor_model_parallel_rank = None
get_tensor_model_parallel_world_size = None
tensor_model_parallel_reduce_scatter = None
sp_all_gather = None
_custom_collective = None
# The patched module; names the original forwards resolve from it
# (``get_pp_group``, ``IntermediateTensors``, classes) are looked up there.
_target: ModuleType | None = None


def _bind_runtime_names(model_module: ModuleType) -> None:
    from vllm import envs as vllm_envs
    from vllm.forward_context import (
        get_forward_context as forward_context,
        is_forward_context_available as forward_context_available,
    )
    from vllm import distributed
    from vllm.models.common.ops import sequence_parallel

    globals().update(
        envs=vllm_envs,
        get_forward_context=forward_context,
        is_forward_context_available=forward_context_available,
        get_tensor_model_parallel_rank=distributed.get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size=(
            distributed.get_tensor_model_parallel_world_size
        ),
        tensor_model_parallel_reduce_scatter=(
            distributed.tensor_model_parallel_reduce_scatter
        ),
        sp_all_gather=sequence_parallel.sp_all_gather,
        _custom_collective=sequence_parallel._custom_collective,
        _target=model_module,
    )


def hc_sequence_parallel_enabled(vllm_config) -> bool:
    from vllm_hcu.patch.config import get_hcu_config

    # Read the platform-resolved value of VLLM_HCU_QWEN4_EXP_HC_SP, which is
    # part of the compilation cache key. use_sequence_parallel_moe already
    # implies TP>1, DP>1 and EP.
    return bool(
        get_hcu_config(vllm_config).qwen4_exp_hc_sp
        and vllm_config.parallel_config.use_sequence_parallel_moe
    )


# vLLM's sp_shard/sp_reduce_scatter/sp_padding_mask pad only under
# ``if sp_pad > 0``.  torch.compile specializes that branch on the traced token
# count and vLLM drops the guard, so a graph traced with an even count silently
# loses a row for odd counts.  These variants always pad (possibly by zero).
def _pad_rows(x: torch.Tensor, value=0) -> torch.Tensor:
    sp_pad = (-x.shape[0]) % get_tensor_model_parallel_world_size()
    return torch.nn.functional.pad(x, (0, 0) * (x.ndim - 1) + (0, sp_pad), value=value)


def _local_rows(x: torch.Tensor) -> torch.Tensor:
    chunk = x.shape[0] // get_tensor_model_parallel_world_size()
    start = get_tensor_model_parallel_rank() * chunk
    return x[start : start + chunk]


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    return _local_rows(_pad_rows(x))


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    x = _pad_rows(x)
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_padding_mask(
    is_padding: torch.Tensor | None, hidden_states: torch.Tensor
) -> torch.Tensor:
    if is_padding is None:
        is_padding = hidden_states.new_zeros(hidden_states.shape[0], dtype=torch.bool)
    return _local_rows(_pad_rows(is_padding, value=True))


def shard_tokens_for_hc_sp(hidden_states: torch.Tensor) -> torch.Tensor:
    """Keep this rank's token rows and align the MoE padding mask with them."""
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        forward_context = get_forward_context()
        forward_context.is_padding = sp_padding_mask(
            forward_context.is_padding, hidden_states
        )
    return sp_shard(hidden_states)


def gather_packed(
    first: torch.Tensor, second: torch.Tensor, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather two row-aligned tensors with a single collective."""
    packed = sp_all_gather(torch.cat([first, second], dim=-1))[:num_tokens]
    first, second = packed.split([first.shape[-1], second.shape[-1]], dim=-1)
    return first.contiguous(), second.contiguous()


def _checked_source(function, key: str, target: str) -> str:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} source fingerprint could not be "
            "computed"
        ) from exc
    expected = _SOURCE_SHA256[key]
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != expected:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} source fingerprint mismatch: "
            f"expected sha256={expected}, actual sha256={actual}"
        )
    return source


def _is_sp_moe_guard(node: ast.stmt) -> bool:
    """Match ``if <...>.use_sequence_parallel_moe: raise NotImplementedError``."""
    if not isinstance(node, ast.If) or node.orelse or len(node.body) != 1:
        return False
    test, body = node.test, node.body[0]
    if not (
        isinstance(test, ast.Attribute) and test.attr == "use_sequence_parallel_moe"
    ):
        return False
    if not isinstance(body, ast.Raise) or not isinstance(body.exc, ast.Call):
        return False
    func = body.exc.func
    return isinstance(func, ast.Name) and func.id == "NotImplementedError"


def _rebuild_init_without_guard(owner: type, target: str):
    init = vars(owner).get("__init__")
    if not callable(init):
        raise PatchCompatibilityError(f"required HCU patch target {target} is missing")
    source = _checked_source(init, f"{owner.__name__}.__init__", target)

    function_def = ast.parse(source).body[0]
    assert isinstance(function_def, ast.FunctionDef)
    kept = [stmt for stmt in function_def.body if not _is_sp_moe_guard(stmt)]
    if len(function_def.body) - len(kept) != 1:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has "
            f"{len(function_def.body) - len(kept)} sequence-parallel MoE guards; "
            "expected 1"
        )
    function_def.body = kept

    # Zero-argument super() needs a ``__class__`` cell; a factory parameter of
    # that name provides it outside the class body.
    factory = ast.FunctionDef(
        name="_factory",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="__class__")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[function_def, ast.Return(value=ast.Name(id="__init__", ctx=ast.Load()))],
        decorator_list=[],
    )
    module_ast = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    namespace: dict[str, object] = {}
    filename = inspect.getsourcefile(init) or init.__code__.co_filename
    exec(compile(module_ast, filename, "exec"), init.__globals__, namespace)
    rebuilt = namespace["_factory"](owner)
    functools.update_wrapper(rebuilt, init)
    return init, rebuilt


def _attention_out_projection(layer) -> torch.nn.Module:
    if layer.layer_type == "linear_attention":
        return layer.linear_attn.out_proj
    return layer.self_attn.o_proj


def _make_decoder_init(guardless_init):
    @functools.wraps(guardless_init)
    def hcu_decoder_init(self, vllm_config, layer_type, prefix=""):
        guardless_init(self, vllm_config, layer_type, prefix)
        hc_sp = hc_sequence_parallel_enabled(vllm_config)
        setattr(self, HC_SP_ATTR, hc_sp)
        if hc_sp:
            # Partial attention outputs are reduce-scattered by the layer.
            _attention_out_projection(self).reduce_results = False

    return hcu_decoder_init


def _make_model_init(init):
    @functools.wraps(init)
    def hcu_model_init(self, *args, vllm_config=None, prefix="", **kwargs):
        init(self, *args, vllm_config=vllm_config, prefix=prefix, **kwargs)
        hc_sp = hc_sequence_parallel_enabled(vllm_config)
        if hc_sp and vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Qwen4Exp HC sequence parallelism requires pipeline_parallel_size=1"
            )
        setattr(self, HC_SP_ATTR, hc_sp)

    return hcu_model_init


def decoder_forward(
    self,
    hidden_states: torch.Tensor,
    prev_block_output: torch.Tensor | None,
    prev_injection: torch.Tensor | None,
    positions: torch.Tensor,
    *,
    input_ids: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    ngram_context: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_sp = getattr(self, HC_SP_ATTR, False)
    # Positions keep the full token axis for both text and MRoPE inputs.
    num_tokens = positions.shape[-1]
    attn_hc = self.attn_hyper_connection
    if self.ple is not None:
        # PLE adds directly to the multi-stream state, so pending HC state
        # must be materialized before the addition.
        if prev_block_output is not None and prev_injection is not None:
            hidden_states = attn_hc.combine(
                hidden_states, prev_block_output, prev_injection
            )
            prev_block_output = prev_injection = None

        if input_ids is None or query_start_loc is None or ngram_context is None:
            raise RuntimeError("PLE inputs were not prepared")
        if hc_sp:
            # PLE's short convolution spans tokens: run it on the full state.
            full_hidden_states = sp_all_gather(hidden_states)[:num_tokens]
            hidden_states = hidden_states + sp_shard(
                self.ple(
                    full_hidden_states,
                    input_ids,
                    query_start_loc,
                    ngram_context,
                )
            )
        else:
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

    # Fuse a pending combine with this HC module's mix when possible.
    if prev_block_output is not None and prev_injection is not None:
        hidden_states, block_input, injection = attn_hc.combine_and_mix(
            hidden_states, prev_block_output, prev_injection
        )
    else:
        hidden_states, block_input, injection = attn_hc.mix(hidden_states)

    if hc_sp:
        block_input = sp_all_gather(block_input)[:num_tokens]
    if self.layer_type == "linear_attention":
        attn_out = self.linear_attn(hidden_states=block_input)
    elif self.layer_type == "full_attention":
        attn_out = self.self_attn(
            hidden_states=block_input,
            positions=positions,
        )
    else:
        raise ValueError("Invalid layer_type")
    if hc_sp:
        attn_out = sp_reduce_scatter(attn_out)

    mlp_hc = self.mlp_hyper_connection
    hidden_states, block_input, injection = mlp_hc.combine_and_mix(
        hidden_states, attn_out, injection
    )
    if not hc_sp:
        mlp_out = self.mlp(block_input)
    elif isinstance(self.mlp, _target.Qwen4ExpSparseMoeBlock):
        # MoE consumes local tokens and returns complete local outputs.
        mlp_out = self.mlp(block_input, already_sequence_parallel=True)
    else:
        # Dense MLPs all-reduce their output; keep only the local rows.
        mlp_out = sp_shard(self.mlp(sp_all_gather(block_input)[:num_tokens]))
    return hidden_states, mlp_out, injection


def model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    ngram_context: torch.Tensor | None = None,
    deepstack_input_embeds=None,
):
    hc_sp = getattr(self, HC_SP_ATTR, False)
    num_tokens = positions.shape[-1]
    if _target.get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self.embed_input_ids(input_ids)
        if hc_sp:
            hidden_states = shard_tokens_for_hc_sp(hidden_states)
        # Expand only the local token rows when HC SP is enabled.
        hidden_states = hidden_states.repeat(1, self.config.hc_count)
    else:
        if intermediate_tensors is None:
            raise ValueError("pipeline stage requires intermediate tensors")
        hidden_states = intermediate_tensors["hidden_states"]

    block_output = None
    injection = None
    last_layer = None
    for layer_idx, layer in islice(
        enumerate(self.layers), self.start_layer, self.end_layer
    ):
        last_layer = layer
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=block_output,
            prev_injection=injection,
            positions=positions,
            input_ids=input_ids,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        if deepstack_input_embeds is not None and layer_idx < len(
            deepstack_input_embeds
        ):
            deepstack_embed = deepstack_input_embeds[
                f"deepstack_input_embeds_{layer_idx}"
            ]
            if hc_sp:
                deepstack_embed = sp_shard(deepstack_embed)
            deepstack_embed = (
                deepstack_embed.unsqueeze(-2)
                .expand(
                    *deepstack_embed.shape[:-1],
                    self.config.hc_count,
                    self.config.hidden_size,
                )
                .flatten(-2)
            )
            # Deepstack is an external addition to the materialized
            # multi-stream state and therefore terminates delayed combine.
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
            block_output = None
            injection = None
            hidden_states = hidden_states + deepstack_embed

    if not _target.get_pp_group().is_last_rank:
        # PP transports one tensor, not the delayed HC tuple. Materialize
        # with the HC module that produced the pending injection.
        if last_layer is not None and block_output is not None:
            hidden_states = last_layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
        return _target.IntermediateTensors({"hidden_states": hidden_states})

    # The final mixer consumes the last pending combine and returns both
    # the sampled single stream and the materialized multi-stream state.
    final_mixer = self.hyper_connection_mixer
    assert final_mixer is not None
    multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
        hidden_states, block_output, injection
    )
    if hc_sp:
        if self._mtp_hidden_buffer is not None:
            # Gather LM-head and MTP states together when both are needed.
            sample_hidden_states, multi_hidden = gather_packed(
                sample_hidden_states, multi_hidden, num_tokens
            )
        else:
            sample_hidden_states = sp_all_gather(sample_hidden_states)[:num_tokens]
    if self._mtp_hidden_buffer is not None:
        # Capture the pre-final-mixer multi-stream hidden state
        # [T, hc_count*H] for the MTP drafter (zero extra compute:
        # this tensor is needed by the final mixer regardless).
        num_tokens = multi_hidden.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
    return sample_hidden_states


def same_parameters(a, b) -> bool:
    """Compare names, kinds and defaults, ignoring annotation spelling."""
    pa = tuple(inspect.signature(a).parameters.values())
    pb = tuple(inspect.signature(b).parameters.values())
    return [(p.name, p.kind, p.default) for p in pa] == [
        (p.name, p.kind, p.default) for p in pb
    ]


def _replacement_forward(owner: type, key: str, target: str, replacement):
    original = vars(owner).get("forward")
    if not callable(original):
        raise PatchCompatibilityError(f"required HCU patch target {target} is missing")
    _checked_source(original, key, target)
    if not same_parameters(original, replacement):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature "
            f"{inspect.signature(original)}"
        )
    functools.update_wrapper(replacement, original)
    return original


def apply_to_module(module: ModuleType) -> bool:
    model_module = load_exact_module(TARGET_MODULE, module)
    moe_class, decoder_class, model_class = (
        require_class(model_module, name, f"{TARGET_MODULE}.{name}")
        for name in ("Qwen4ExpSparseMoeBlock", "Qwen4ExpDecoderLayer", "Qwen4ExpModel")
    )
    slots = (
        (moe_class, "__init__"),
        (decoder_class, "__init__"),
        (decoder_class, "forward"),
        (model_class, "__init__"),
        (model_class, "forward"),
    )
    current = [vars(owner).get(name) for owner, name in slots]
    if getattr(model_module, _MARKER, False):
        if not all(getattr(function, _WRAPPER, False) for function in current):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False
    if any(getattr(function, _WRAPPER, False) for function in current):
        raise PatchCompatibilityError(
            "refusing a partial Qwen4Exp sequence-parallel patch"
        )

    # Validate and build everything before installing anything.
    _, moe_init = _rebuild_init_without_guard(moe_class, TARGETS[0])
    _, guardless_decoder_init = _rebuild_init_without_guard(decoder_class, TARGETS[1])
    require_exact_signature(
        guardless_decoder_init,
        TARGETS[1],
        positional=("self", "vllm_config", "layer_type", "prefix"),
        defaults={"prefix": ""},
    )
    decoder_init = _make_decoder_init(guardless_decoder_init)
    _replacement_forward(
        decoder_class, "Qwen4ExpDecoderLayer.forward", TARGETS[2], decoder_forward
    )
    model_init = _make_model_init(
        _require_model_init_compatible(model_class, TARGETS[3])
    )
    _replacement_forward(
        model_class, "Qwen4ExpModel.forward", TARGETS[4], model_forward
    )
    _bind_runtime_names(model_module)

    replacements = (moe_init, decoder_init, decoder_forward, model_init, model_forward)
    for function in replacements:
        setattr(function, _WRAPPER, True)
    try:
        for (owner, name), function in zip(slots, replacements):
            setattr(owner, name, function)
        setattr(model_module, _MARKER, True)
    except BaseException:
        for (owner, name), function in zip(slots, current):
            setattr(owner, name, function)
        if hasattr(model_module, _MARKER):
            delattr(model_module, _MARKER)
        raise
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "HC_SP_ATTR",
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "gather_packed",
    "hc_sequence_parallel_enabled",
    "same_parameters",
    "sp_reduce_scatter",
    "sp_shard",
    "shard_tokens_for_hc_sp",
]
