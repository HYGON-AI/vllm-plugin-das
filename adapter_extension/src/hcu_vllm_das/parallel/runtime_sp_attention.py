# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Projection layout adapters for the official GLM4/HY3 attention modules.

Construction is explicit and scoped. No model or extension is enabled here:
model entry/exit, dense/shared MLP and early MoE policy must be composed first.
Official linear methods own allocation, loading, quantization and GEMM.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import inspect
import os

import torch
from hcu_adapter.extensions import ExtensionCommitError
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear,
)
from vllm.utils.torch_utils import direct_register_custom_op
from .runtime_sp_batch import current_sp_batch

_construction = ContextVar("hcu_runtime_sp_attention_construction", default=False)
_mlp_construction = ContextVar("hcu_runtime_sp_mlp_construction", default=False)
_qkv_signature = inspect.signature(QKVParallelLinear.__init__)
_row_signature = inspect.signature(RowParallelLinear.__init__)
_merged_signature = inspect.signature(MergedColumnParallelLinear.__init__)
_PUBLICATION_ACTION = "runtime-sp-attention-publication"
_publication_lifecycle = ProcessLocalLifecycle(
    runtime_action=_PUBLICATION_ACTION, error_type=ExtensionCommitError,
)


@contextmanager
def runtime_sp_attention_construction():
    """Construct attention projections with replicated weights/local tokens."""
    token = _construction.set(True)
    try:
        yield
    finally:
        _construction.reset(token)


@contextmanager
def runtime_sp_model_construction():
    """Prepare attention, TP dense MLP and replicated shared projections.

    This construction scope does not enable a model. Entry/exit token layouts
    and early routed-expert policy still belong to the composing model owner.
    """
    token = _mlp_construction.set(True)
    try:
        with runtime_sp_attention_construction():
            yield
    finally:
        _mlp_construction.reset(token)


def _mlp_projection(prefix, name):
    return _mlp_construction.get() and any(prefix.endswith(f".mlp.{path}{name}")
        for path in ("", "shared_experts.", "shared_mlp."))


def _dense_mlp_projection(prefix, name):
    return _mlp_construction.get() and prefix.endswith(f".mlp.{name}")


def model_construction_active():
    return _mlp_construction.get()


@dataclass(frozen=True)
class AttentionLayout:
    size: int
    rank: int
    q_heads: int
    kv_heads: int
    head_dim: int
    v_head_dim: int

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (
            self.size, self.q_heads, self.kv_heads, self.head_dim, self.v_head_dim
        )):
            raise ValueError("runtime SP attention requires positive integer geometry")
        if type(self.rank) is not int or not 0 <= self.rank < self.size:
            raise ValueError("runtime SP attention rank is outside its TP group")
        if self.q_heads % self.size or self.q_heads % self.kv_heads:
            raise ValueError("runtime SP attention requires divisible query heads")
        if self.kv_heads >= self.size:
            if self.kv_heads % self.size:
                raise ValueError("runtime SP KV heads must divide the TP group")
        elif self.size % self.kv_heads:
            raise ValueError("runtime SP KV replica groups must divide TP")

    @property
    def q_width(self):
        return self.q_heads * self.head_dim

    @property
    def k_width(self):
        return self.kv_heads * self.head_dim

    @property
    def v_width(self):
        return self.kv_heads * self.v_head_dim

    def pack_qkv(self, qkv):
        if qkv.ndim != 2 or qkv.shape[-1] != self.q_width + self.k_width + self.v_width:
            raise ValueError("runtime SP QKV output does not match its head geometry")
        q, k, v = qkv.split([self.q_width, self.k_width, self.v_width], dim=-1)
        rows = qkv.shape[0]
        q = q.reshape(rows, self.size, self.q_width // self.size)
        kv_per_rank = max(1, self.kv_heads // self.size)
        # Core maps rank -> KV head by rank // num_kv_head_replicas.
        # Every source must send the chosen head to EVERY replica destination;
        # selecting the local rank's head before all_gather mixes KV heads.
        if self.kv_heads < self.size:
            repeats = self.size // self.kv_heads
            k = k.reshape(rows, self.kv_heads, self.head_dim).repeat_interleave(repeats, dim=1)
            v = v.reshape(rows, self.kv_heads, self.v_head_dim).repeat_interleave(repeats, dim=1)
        k = k.reshape(rows, self.size, kv_per_rank * self.head_dim)
        v = v.reshape(rows, self.size, kv_per_rank * self.v_head_dim)
        return torch.cat((q, k, v), dim=-1).permute(1, 0, 2).contiguous()

    def local_bias(self, bias):
        return self.pack_qkv(bias.reshape(1, -1))[self.rank, 0].contiguous()


def _exchange(group, send):
    if group.world_size == 1:
        return send
    recv = torch.empty_like(send)
    group.device_communicator.all_to_all_single(recv, send)
    return recv


def _check_group(layer):
    group = _check_group_fields(*_owner_fields(layer))
    if group is not layer._runtime_sp_group:
        raise RuntimeError("runtime SP attention TP group ownership changed")
    return group


def _owner_fields(layer):
    return (layer._runtime_sp_pid, layer._runtime_sp_owner_id,
            layer._runtime_sp_size, layer._runtime_sp_rank)


def _check_group_fields(pid, owner_id, size, rank):
    if os.getpid() != pid:
        raise RuntimeError("runtime SP attention inherited across fork; use spawn")
    group = get_tp_group()
    if os.getpid() != pid:
        raise RuntimeError("runtime SP attention inherited across fork; use spawn")
    if id(group) != owner_id:
        raise RuntimeError("runtime SP attention TP group ownership changed")
    if group.world_size != size or group.rank_in_group != rank:
        raise RuntimeError("runtime SP attention TP geometry changed")
    return group


def _masked_input(input_, padding):
    if padding is None:
        return input_.clone(memory_format=torch.contiguous_format)
    if input_.ndim != 2 or input_.shape[0] != padding.shape[0]:
        raise ValueError("runtime SP projection received an invalid token layout")
    return input_.masked_fill(padding[:, None], 0)


def _prepare_input(input_: torch.Tensor, padding: torch.Tensor | None,
                   pid: int, owner_id: int, size: int, rank: int) -> torch.Tensor:
    # This data-producing op precedes the official GEMM. Its guard cannot be
    # removed as an unused, output-free Python side effect during compilation.
    _check_group_fields(pid, owner_id, size, rank)
    return _masked_input(input_, padding)


def _prepare_input_fake(input_: torch.Tensor, padding: torch.Tensor | None,
                        pid: int, owner_id: int, size: int, rank: int) -> torch.Tensor:
    return input_.new_empty(input_.shape)


def _collective(input_: torch.Tensor, padding: torch.Tensor | None, pid: int, owner_id: int,
                size: int, rank: int, kind: str) -> torch.Tensor:
    group = _check_group_fields(pid, owner_id, size, rank)
    if kind == 'exchange':
        return _exchange(group, input_) if size > 1 else input_.clone()
    if kind == 'gather':
        return group.all_gather(input_, dim=0) if size > 1 else input_.clone()
    if kind == 'scatter':
        result = group.reduce_scatter(input_, dim=0) if size > 1 else input_
        return _masked_input(result, padding)
    raise ValueError("unknown runtime SP collective")


def _collective_fake(input_: torch.Tensor, padding: torch.Tensor | None, pid: int, owner_id: int,
                     size: int, rank: int, kind: str) -> torch.Tensor:
    shape = list(input_.shape)
    if kind == 'gather':
        shape[0] *= size
    elif kind == 'scatter':
        shape[0] //= size
    elif kind != 'exchange':
        raise ValueError("unknown runtime SP collective")
    return input_.new_empty(shape)


# Keep only layout/ownership/collective Python behind the public Core custom-op
# seam. Core still compiles the original quantization method and GEMM normally.
_sp_lib = torch.library.Library('hcu_vllm_das_sp', 'FRAGMENT')
for _name, _impl, _fake in (
    ('prepare_input', _prepare_input, _prepare_input_fake),
    ('collective', _collective, _collective_fake),
):
    direct_register_custom_op(_name, _impl, fake_impl=_fake,
                              target_lib=_sp_lib, dispatch_key='CUDA')
    _sp_lib.impl(_name, _impl, dispatch_key='CPU')


def _prepared(layer, input_, global_layout=False):
    # Masks must be graph tensor inputs, not hidden context reads inside an
    # opaque op: Core's graph wrapper must track/copy them on later replays.
    batch = current_sp_batch()
    padding = None if batch is None else (
        batch.global_padding if global_layout else batch.local_padding)
    return torch.ops.hcu_vllm_das_sp.prepare_input(
        input_, padding, *_owner_fields(layer))


def _communicate(layer, input_, kind):
    batch = current_sp_batch() if kind == 'scatter' else None
    padding = None if batch is None else batch.local_padding
    return torch.ops.hcu_vllm_das_sp.collective(input_, padding, *_owner_fields(layer), kind)


def _bind(signature, self, args, kwargs):
    bound = signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    return {k: v for k, v in bound.arguments.items() if k != "self"}


def _capture_group(layer):
    group = get_tp_group()
    layer._runtime_sp_pid = os.getpid()
    layer._runtime_sp_group = group
    layer._runtime_sp_owner_id = id(group)
    layer._runtime_sp_size = group.world_size
    layer._runtime_sp_rank = group.rank_in_group
    return group


class RuntimeSPQKVParallelLinear(QKVParallelLinear):
    def __init__(self, *args, **kwargs):
        params = _bind(_qkv_signature, self, args, kwargs)
        self._runtime_sp = _construction.get() and params["prefix"].endswith(".self_attn.qkv_proj")
        if self._runtime_sp:
            group = _capture_group(self)
            self._runtime_sp_layout = AttentionLayout(
                group.world_size, group.rank_in_group, params["total_num_heads"],
                params["total_num_kv_heads"] or params["total_num_heads"],
                params["head_size"], params["v_head_size"] or params["head_size"],
            )
            # Core sets tp_rank=0/tp_size=1 BEFORE creating quantized weights
            # and their loaders. Never concatenate/replace initialized weights.
            params["disable_tp"] = True
        super().__init__(**params)

    def forward(self, input_):
        if not self._runtime_sp:
            return super().forward(input_)
        input_ = _prepared(self, input_)
        if input_.ndim != 2:
            raise ValueError("runtime SP QKV expects a two-dimensional local token tensor")
        result = super().forward(input_)
        output, bias = result if self.return_bias else (result, None)
        send = self._runtime_sp_layout.pack_qkv(output)
        recv = _communicate(self, send, 'exchange')
        output = recv.reshape(recv.shape[0] * recv.shape[1], recv.shape[2])
        if bias is not None:
            bias = self._runtime_sp_layout.local_bias(bias)
        return (output, bias) if self.return_bias else output


class RuntimeSPRowParallelLinear(RowParallelLinear):
    def __init__(self, *args, **kwargs):
        params = _bind(_row_signature, self, args, kwargs)
        self._runtime_sp = _construction.get() and params["prefix"].endswith(".self_attn.o_proj")
        self._runtime_sp_mlp = _mlp_projection(params["prefix"], "down_proj")
        self._runtime_sp_mlp_gather = _dense_mlp_projection(params["prefix"], "down_proj")
        if self._runtime_sp:
            group = _capture_group(self)
            if params["input_size"] % group.world_size:
                raise ValueError("runtime SP output projection width must divide TP")
            params["disable_tp"] = True
        elif self._runtime_sp_mlp:
            _capture_group(self)
            if self._runtime_sp_mlp_gather:
                if params["disable_tp"]:
                    raise ValueError("runtime SP dense MLP requires TP weight allocation")
                # Dense gate/up sees gathered tokens and retains official TP
                # weights. Scatter the summed partial outputs instead of
                # all-reducing them onto every token shard.
                params["reduce_results"] = False
            else:
                params["disable_tp"] = True
        super().__init__(**params)

    def forward(self, input_):
        if self._runtime_sp_mlp:
            if not self._runtime_sp_mlp_gather:
                return super().forward(_prepared(self, input_))
            input_ = _prepared(self, input_, global_layout=True)
            if input_.ndim != 2 or input_.shape[0] % self._runtime_sp_size:
                raise ValueError("runtime SP dense MLP requires gathered padded tokens")
            result = super().forward(input_)
            output, bias = result if self.return_bias else (result, None)
            output = _communicate(self, output, 'scatter')
            return (output, bias) if self.return_bias else output
        if not self._runtime_sp:
            return super().forward(input_)
        input_ = _prepared(self, input_, global_layout=True)
        if input_.ndim != 2 or input_.shape[-1] * self._runtime_sp_size != self.input_size:
            raise ValueError("runtime SP output projection expects head-sharded input")
        if input_.shape[0] % self._runtime_sp_size:
            raise ValueError("runtime SP attention tokens must be padded to TP size")
        # Clear unwritten NaN padding before inverse exchange/quantization.
        rows = input_.shape[0] // self._runtime_sp_size
        send = input_.reshape(self._runtime_sp_size, rows, input_.shape[-1]).contiguous()
        recv = _communicate(self, send, 'exchange')
        tokens = recv.permute(1, 0, 2).contiguous().reshape(rows, self.input_size)
        return super().forward(tokens)


class RuntimeSPMergedColumnParallelLinear(MergedColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        params = _bind(_merged_signature, self, args, kwargs)
        self._runtime_sp_mlp = _mlp_projection(params["prefix"], "gate_up_proj")
        self._runtime_sp_mlp_gather = _dense_mlp_projection(params["prefix"], "gate_up_proj")
        if self._runtime_sp_mlp:
            _capture_group(self)
            # Shared experts remain local/replicated; ordinary dense MLPs keep
            # TP allocation/loading and gather token rows before their GEMM.
            if self._runtime_sp_mlp_gather and params["disable_tp"]:
                raise ValueError("runtime SP dense MLP requires TP weight allocation")
            if not self._runtime_sp_mlp_gather:
                params["disable_tp"] = True
        super().__init__(**params)

    def forward(self, input_):
        if self._runtime_sp_mlp:
            input_ = _prepared(self, input_)
            if self._runtime_sp_mlp_gather and self._runtime_sp_size > 1:
                input_ = _communicate(self, input_.contiguous(), 'gather')
        return super().forward(input_)


_BINDINGS = ((QKVParallelLinear, RuntimeSPQKVParallelLinear),
             (RowParallelLinear, RuntimeSPRowParallelLinear),
             (MergedColumnParallelLinear, RuntimeSPMergedColumnParallelLinear))


def _publish_attention_layers():
    from vllm.model_executor.custom_op import op_registry_oot
    for base, candidate in _BINDINGS:
        owner = op_registry_oot.get(base.__name__)
        if owner is not None and owner is not candidate:
            raise RuntimeError(f"foreign OOT owner for {base.__name__}")
    try:
        for base, candidate in _BINDINGS:
            if op_registry_oot.get(base.__name__) is None:
                base.register_oot(candidate)
    except BaseException as error:
        # No assumption that a partially published registry is reusable.
        raise ExtensionCommitError("runtime SP attention publication incomplete") from error
    check_attention_layers()


def install_attention_layers():
    """Serialize concrete OOT publication and refuse inherited partial state."""
    _publication_lifecycle.run_once(_PUBLICATION_ACTION, _publish_attention_layers)
    check_attention_layers()


def check_attention_layers():
    _publication_lifecycle.require_runtime_healthy()
    from vllm.model_executor.custom_op import op_registry_oot
    if any(op_registry_oot.get(base.__name__) is not candidate for base, candidate in _BINDINGS):
        raise ExtensionCommitError("runtime SP attention OOT ownership changed")
