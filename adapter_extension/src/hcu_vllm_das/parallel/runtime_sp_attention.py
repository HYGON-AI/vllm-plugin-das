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
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear

_construction = ContextVar("hcu_runtime_sp_attention_construction", default=False)
_qkv_signature = inspect.signature(QKVParallelLinear.__init__)
_row_signature = inspect.signature(RowParallelLinear.__init__)
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
    if os.getpid() != layer._runtime_sp_pid:
        raise RuntimeError("runtime SP attention inherited across fork; use spawn")
    group = get_tp_group()
    if group is not layer._runtime_sp_group:
        raise RuntimeError("runtime SP attention TP group ownership changed")
    if group.world_size != layer._runtime_sp_size or group.rank_in_group != layer._runtime_sp_rank:
        raise RuntimeError("runtime SP attention TP geometry changed")
    return group


def _bind(signature, self, args, kwargs):
    bound = signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    return {k: v for k, v in bound.arguments.items() if k != "self"}


def _capture_group(layer):
    group = get_tp_group()
    layer._runtime_sp_pid = os.getpid()
    layer._runtime_sp_group = group
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
        group = _check_group(self)
        if input_.ndim != 2:
            raise ValueError("runtime SP QKV expects a two-dimensional local token tensor")
        result = super().forward(input_)
        output, bias = result if self.return_bias else (result, None)
        send = self._runtime_sp_layout.pack_qkv(output)
        recv = _exchange(group, send)
        output = recv.reshape(recv.shape[0] * recv.shape[1], recv.shape[2])
        if bias is not None:
            bias = self._runtime_sp_layout.local_bias(bias)
        return (output, bias) if self.return_bias else output


class RuntimeSPRowParallelLinear(RowParallelLinear):
    def __init__(self, *args, **kwargs):
        params = _bind(_row_signature, self, args, kwargs)
        self._runtime_sp = _construction.get() and params["prefix"].endswith(".self_attn.o_proj")
        if self._runtime_sp:
            group = _capture_group(self)
            if params["input_size"] % group.world_size:
                raise ValueError("runtime SP output projection width must divide TP")
            params["disable_tp"] = True
        super().__init__(**params)

    def forward(self, input_):
        if not self._runtime_sp:
            return super().forward(input_)
        group = _check_group(self)
        if input_.ndim != 2 or input_.shape[-1] * group.world_size != self.input_size:
            raise ValueError("runtime SP output projection expects head-sharded input")
        if input_.shape[0] % group.world_size:
            raise ValueError("runtime SP attention tokens must be padded to TP size")
        rows = input_.shape[0] // group.world_size
        send = input_.reshape(group.world_size, rows, input_.shape[-1]).contiguous()
        recv = _exchange(group, send)
        tokens = recv.permute(1, 0, 2).contiguous().reshape(rows, self.input_size)
        return super().forward(tokens)


_BINDINGS = ((QKVParallelLinear, RuntimeSPQKVParallelLinear),
             (RowParallelLinear, RuntimeSPRowParallelLinear))


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
