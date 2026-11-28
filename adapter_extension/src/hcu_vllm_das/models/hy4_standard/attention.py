# SPDX-License-Identifier: Apache-2.0
"""Owned stock-c440 MLA construction; native forward bodies remain intact."""

import ast
import hashlib
import inspect
import textwrap

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.hy_v4.nvidia import attention as native

from ...hy4_constructor import bind_constructor
from ..hy_v4_flash.dfly_rotary import HYV4StandardRotaryEmbedding


def raw_constructor(function):
    while function.__module__ == "vllm.compilation.decorators":
        if "old_init" not in function.__code__.co_freevars:
            raise RuntimeError("HY4 stock compile constructor anchor changed")
        position = function.__code__.co_freevars.index("old_init")
        function = function.__closure__[position].cell_contents
    return function


def verify_stock_anchor(function, digest):
    body = ast.parse(textwrap.dedent(inspect.getsource(raw_constructor(function))))
    actual = hashlib.sha256(
        ast.dump(body.body[0], include_attributes=False).encode()
    ).hexdigest()
    if actual != digest:
        raise RuntimeError(
            f"HY4 reviewed stock Core source changed: {function.__qualname__}"
        )


def bind_stock_constructor(function, digest, substitutions):
    verify_stock_anchor(function, digest)
    return bind_constructor(function, substitutions)


_STOCK_GET_ROPE = native.get_rope
_STOCK_SKIP = native.compute_skip_topk_layers
_STOCK_INDEXER = native.Indexer
_STOCK_ATTENTION = native.HYV4MLAAttention
_STOCK_INIT = _STOCK_ATTENTION.__init__
_STOCK_INDEX_FORWARD = _STOCK_INDEXER.forward
_STOCK_FORWARD = _STOCK_ATTENTION.forward
_STOCK_INDEX_ATTN = _STOCK_ATTENTION._indexer_and_attn
_INIT_DIGEST = "afb6c0e49644fc65ee24bee24aad226fcaec98e852aff79ea0ab64f6b2635e59"
_FORWARD_DIGEST = "a1316a8c8b8be906d2a75ec2d932af468ee8691f5e1e6d8a40a3c8cb6ebdd924"
_INDEX_DIGEST = "6a7fc73e00814b378de2a404aef423fcba9bfdd1ee8e0c58286cca38eea0345e"


def check_stock_attention_anchors():
    if (
        native.HYV4MLAAttention is not _STOCK_ATTENTION
        or _STOCK_ATTENTION.__init__ is not _STOCK_INIT
        or _STOCK_ATTENTION.forward is not _STOCK_FORWARD
        or _STOCK_ATTENTION._indexer_and_attn is not _STOCK_INDEX_ATTN
        or native.Indexer is not _STOCK_INDEXER
        or _STOCK_INDEXER.forward is not _STOCK_INDEX_FORWARD
        or _STOCK_INIT.__globals__["Indexer"] is not _STOCK_INDEXER
        or _STOCK_INIT.__globals__["get_rope"] is not _STOCK_GET_ROPE
        or _STOCK_INIT.__globals__["compute_skip_topk_layers"] is not _STOCK_SKIP
    ):
        raise RuntimeError("HY4 stock attention code/global owner changed")


def standard_rope(*args, **kwargs):
    return HYV4StandardRotaryEmbedding(_STOCK_GET_ROPE(*args, **kwargs))


def standard_skip_topk(config):
    if getattr(config, "dfly_external_topk", False):
        # Every draft layer is a backbone layer (not an MTP layer); stock
        # construction therefore never creates an indexer or its KV cache.
        return set(range(config.num_hidden_layers))
    return _STOCK_SKIP(config)


class HYV4SnapshotIndexer(_STOCK_INDEXER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk_snapshot = None

    def forward(self, hidden_states, qr, positions, rotary_emb):
        result = super().forward(hidden_states, qr, positions, rotary_emb)
        if self.topk_snapshot is not None:
            rows = hidden_states.shape[0]
            if self.topk_indices_buffer is None:
                raise RuntimeError("HY4 producer snapshot has no logical top-k buffer")
            self.topk_snapshot[:rows].copy_(self.topk_indices_buffer[:rows])
        return result


verify_stock_anchor(
    _STOCK_INDEX_ATTN,
    "08a3306474644f2271c986268896b5d8a89cd78963c32bce44e8df9295ca8479",
)
verify_stock_anchor(_STOCK_FORWARD, _FORWARD_DIGEST)
verify_stock_anchor(_STOCK_INDEX_FORWARD, _INDEX_DIGEST)
_BOUND_INIT = bind_stock_constructor(
    _STOCK_INIT,
    _INIT_DIGEST,
    {
        "get_rope": standard_rope,
        "Indexer": HYV4SnapshotIndexer,
        "compute_skip_topk_layers": standard_skip_topk,
    },
)


class HYV4StandardAttention(_STOCK_ATTENTION):
    def __init__(self, *args, **kwargs):
        check_stock_attention_anchors()
        config = kwargs.get("config", args[1] if len(args) > 1 else None)
        if config is None:
            raise ValueError("Standard HY4 attention requires its real config")
        self.external_topk = bool(getattr(config, "dfly_external_topk", False))
        if self.external_topk:
            layer = kwargs.get("layer_idx", 0)
            prefix = kwargs.get("prefix", "")
            if (
                not 0 <= layer < config.num_hidden_layers
                or int(prefix.split(".")[-2]) != layer
            ):
                raise ValueError(
                    "External-topk draft must use a valid backbone layer id"
                )
        _BOUND_INIT(self, *args, **kwargs)
        self._topk_snapshot = None
        if self.external_topk and (self.indexer is not None or self.create_indexer):
            raise RuntimeError("External-topk draft registered an unused indexer")
        self.post_proj_norm = (
            RMSNorm(self.hidden_size, config.rms_norm_eps)
            if getattr(config, "attn_post_proj_norm", False)
            else None
        )

    @property
    def topk_snapshot(self):
        return self._topk_snapshot

    @topk_snapshot.setter
    def topk_snapshot(self, value):
        self._topk_snapshot = value
        if self.indexer is not None:
            self.indexer.topk_snapshot = value

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        output = super().forward(positions, hidden_states, llama_4_scaling)
        return (
            self.post_proj_norm(output) if self.post_proj_norm is not None else output
        )
