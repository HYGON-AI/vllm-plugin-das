# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit Core033 HCU HiSparse connector/config -> native planner boundary.

Import is inert. Explicit installation owns the lazy builder resolver and the
worker spec resolver with its pinned imported alias. Cross-worker convergence,
strides, null-block accounting and memory-fit remain Core-owned. IndexPool uses
real separate native HMA groups and allocation-only physical LCM alignment. Public KVTransferConfig must explicitly
select HCUHiSparseConnector in THIS module, kv_both, and finite host_pool_gib.
This is a distinct external connector, not a spoof of Core's connector-name test.
Its native SupportsHMA and required-layout interfaces run during real post-init.
Neither KVTransferConfig nor speculative_config is hidden from post-init.

The external connector retains the runtime admission guard, then delegates to
the REAL native scheduler/worker constructor (not a metadata-only substitute).
This patch qualifies planner metadata, not HIP mapping, transfer operations,
DFly storage aliases or joint execution. Runtime and DFly joint guards remain
in force. No implicit adapter/model activation occurs.
"""
import ast
from copy import deepcopy
from dataclasses import replace
from functools import wraps
import hashlib
from importlib import import_module
import inspect
import math
import os
from threading import RLock

_MODULE = __name__
_CONNECTOR = "HCUHiSparseConnector"
# Read-only AST digests, Core 4ba787ec9959e566d3b587a796dff07673324da5.
_ABI = {
    'vllm.v1.core.kv_cache_config_builder': '9d550f3ca5218dc409e9c3ec7dceac6358310fe0b82058cffee06c1581c09695',
    'vllm.v1.core.kv_cache_planning': 'b0d7a9c829310aab55c71be2244b57e3790e72123a5b6c357be9281ae6fecd8c',
    'vllm.v1.hisparse.layout': 'c5d674731fd330ab23a9fbecc55b0bb93cde6def8909fa8b8d62abb714c6ae51',
    'vllm.v1.hisparse.runtime': '54584b842a883683e407d6b05f27425b3b06d386552beaddd70d22741e518002',
    'vllm.v1.hisparse.binding': '35f20e9df2fe6c618d6c858e8a923c0c8423e5f123ea46e46409ec8b8b414ea2',
    'vllm.v1.kv_cache_interface': '4fd1acf03c4e9c8f2fef2a701b2ba17c2679874fc78f40909746f54daa9b20f7',
    'vllm.config.kv_transfer': '4511e352608649a188f7e9d10e7579ba1b4c42c12a4058323a265a3f90d64c35',
    'vllm.distributed.kv_transfer.kv_connector.factory': 'f810566b1dc0624162af7c38cf306c4eb0e50d0d8a49f46374c5ace38a3ace4a',
    'vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector': 'c77c03b3208bb3fba99dde9dd569ec53e4916996a408c25e8f2f65af6f6c63dd',
    'vllm.v1.attention.backend': '16f15dcd7beffd4787c3b7fd74b689dc4d3060202c693319413311614545728d',
    'vllm.v1.kv_cache_layout': '68949ee5b96e3de08029513914eb55da3ba73ef748bddb55df5d4f74727a8035',
    'vllm.v1.attention.backends.utils': 'cdaacd4b63f0e0d215db8b261ed320f33ef351d2bd8dc1934e2d758d11981c5c',
    'vllm.v1.worker.utils': '8577c2dc843834984a56b36c0f7c9763c1bf16504e4ac0eb0eab3ba77716fca0',
    'vllm.v1.worker.gpu.attn_utils': 'bcf208ac8151fe4790c00d7691a9a5734d1e8b4f3e8ba866fcc5550544487826',
    'hcu_vllm_das.models.hy_v4_flash.indexpool_cache': '3107cf1b034405f371b925c766629d41f7f6e7eed0ec7ed5164c6609dc6c184f',
}
_INDEXPOOL_BASE_ABI = 'c3db5f4fbf1f18837eea3a85576d337aeb6682c7e3455d094d5718a6fc04c7f8'
_lock = RLock()
_state = None
_failure = None
_publishing = False


def _after_fork():
    global _lock, _failure
    _lock = RLock()
    if _state is not None or _publishing:
        _failure = "HiSparse planner publication inherited across fork; use spawn"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _fingerprint(function):
    return (function.__code__, repr(function.__defaults__),
            repr(function.__kwdefaults__), str(inspect.signature(function)),
            tuple(function.__dict__.items()))


def _members(module):
    """Own definitions, including class methods/properties, not imported aliases."""
    result = []
    for name, obj in vars(module).items():
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj):
            result.append((module, name, obj, _fingerprint(obj)))
        elif inspect.isclass(obj):
            result.append((module, name, obj, None))
            for key, value in vars(obj).items():
                if isinstance(value, (staticmethod, classmethod)):
                    fn = value.__func__
                elif isinstance(value, property):
                    fn = value.fget
                else:
                    fn = value
                if inspect.isfunction(fn):
                    result.append((obj, key, value, (fn, _fingerprint(fn))))
    return result


def _healthy():
    if _failure is not None:
        raise RuntimeError(_failure)
    if _state is None:
        raise RuntimeError("HiSparse planner is not explicitly installed")
    if os.getpid() != _state["pid"]:
        raise RuntimeError("HiSparse planner PID changed; use spawn")
    s = _state
    for owner, key, _, replacement in s['publications']:
        if vars(owner).get(key) is not replacement:
            raise RuntimeError('HiSparse planner published entry-point ownership changed')
    if _fingerprint(s['spec_original']) != s['spec_original_fp']:
        raise RuntimeError('HiSparse worker resolver original signature/code changed')
    if s["resolver_module"].get_kv_cache_config_builder is not s["wrapper"]:
        raise RuntimeError("HiSparse planner resolver ownership changed")
    if _fingerprint(s["original"]) != s["original_fp"]:
        raise RuntimeError("HiSparse planner original resolver signature/code changed")
    if _fingerprint(s["wrapper"]) != s["wrapper_fp"]:
        raise RuntimeError("HiSparse planner wrapper signature/code changed")
    for name, function, fingerprint in s["helpers"]:
        if globals().get(name) is not function or _fingerprint(function) != fingerprint:
            raise RuntimeError(f"HiSparse planner helper ownership changed: {name}")
    for owner, key, expected, fingerprint in s["members"]:
        actual = vars(owner).get(key)
        if actual is not expected:
            raise RuntimeError(f"HiSparse planner Core ownership drift: {key}")
        if fingerprint is None:
            continue
        if inspect.ismodule(owner):
            okay = _fingerprint(expected) == fingerprint
        else:
            fn, fp = fingerprint
            okay = _fingerprint(fn) == fp
        if not okay:
            raise RuntimeError(f"HiSparse planner Core signature/code drift: {key}")
    from vllm.platforms import current_platform
    if current_platform is not s["platform"]:
        raise RuntimeError("HiSparse planner platform owner changed")
    for module in s["modules"]:
        import sys
        if sys.modules.get(module.__name__) is not module:
            raise RuntimeError("HiSparse planner Core module owner changed")
    runtime = import_module("vllm.v1.hisparse.runtime")
    if runtime.current_platform is not current_platform:
        raise RuntimeError("HiSparse planner runtime platform alias changed")
    for module, key, value in s["aliases"]:
        if vars(module).get(key) is not value:
            raise RuntimeError(f"HiSparse planner Core alias owner changed: {key}")
    for cls, name, value, fp in s["plugin_members"]:
        if vars(cls).get(name) is not value or _fingerprint(value) != fp:
            raise RuntimeError("HiSparse planner plugin class ownership changed")


def check_health() -> dict:
    """Publication/ABI ownership, NOT a runtime acceptance certificate."""
    global _failure
    with _lock:
        try:
            _healthy()
        except Exception as exc:
            if _state is not None:
                _failure = f"HiSparse planner health failed: {exc}"
            raise
        return {"installed": True, "pid": _state["pid"],
                "connector": _CONNECTOR, "module": _MODULE,
                "planner_metadata_only": True, "runtime_admitted": False}


def _is_connector(kv):
    return getattr(kv, "kv_connector", None) == _CONNECTOR


def _host_bytes(kv):
    from vllm.config.kv_transfer import KVTransferConfig
    if (type(kv) is not KVTransferConfig or not _is_connector(kv)
            or kv.kv_connector_module_path != _MODULE or kv.kv_role != "kv_both"):
        raise ValueError("HiSparse planner requires its explicit external kv_both connector")
    if (kv.enable_permute_local_kv or kv.kv_load_failure_policy != "fail"
            or kv.kv_buffer_device != "cuda"):
        raise ValueError("Unsupported HiSparse connector buffer/failure/permutation scope")
    extra = kv.kv_connector_extra_config
    if type(extra) is not dict or set(extra) != {"host_pool_gib"}:
        raise ValueError("HiSparse connector requires only explicit host_pool_gib")
    gib = extra["host_pool_gib"]
    if (type(gib) not in (int, float) or gib <= 0
            or gib > (2**63 - 1) / 2**30 or not math.isfinite(gib)):
        raise ValueError("host_pool_gib must be a finite positive real number")
    budget = int(gib * 2**30)
    if budget <= 0 or budget > 2**63 - 1:
        raise ValueError("host_pool_gib is outside the supported byte range")
    return budget


def validate_config_scope(value, *, final=False) -> None:
    """Internal config-owner handshake. Does not install or admit execution.

    Only explicit plugin connector enables the additional planner/DFly5 scope;
    arbitrary/native/MultiConnector KV-transfer configurations remain rejected.
    """
    check_health()
    _host_bytes(value.kv_transfer_config)
    if final:
        if value.model_config.original_max_model_len == -1:
            raise ValueError("HiSparse planner does not admit auto-fit model length")
        spec = value.speculative_config
        if spec is not None:
            from . import dfly_compat
            from .dfly_config import HCUDFlySpeculativeConfig
            dfly_compat.check_health()
            if (type(spec) is not HCUDFlySpeculativeConfig or spec.method != "dfly"
                    or type(spec.num_speculative_tokens) is not int
                    or spec.num_speculative_tokens != 5
                    or spec.target_model_config is not value.model_config
                    or spec.target_parallel_config is not value.parallel_config
                    or value.num_lookahead_tokens != 6
                    or spec.parallel_drafting
                    or spec.draft_model_config.dtype != value.model_config.dtype
                    or spec.draft_model_config.enforce_eager is not True
                    or spec.kv_cache_dtype not in (None, "auto", "bfloat16")):
                raise ValueError("HiSparse planner admits only intact owned BF16 DFly5")
    else:
        if value.max_model_len == -1:
            raise ValueError("HiSparse planner does not admit auto-fit model length")
        if any(getattr(value, key, None) is not None
               for key in ("spec_method", "spec_model", "spec_tokens")):
            raise ValueError("HiSparse planner requires canonical speculative_config, not CLI aliases")
        spec = value.speculative_config
        if spec is not None and (type(spec) is not dict
                or set(spec) != {"method", "num_speculative_tokens"}
                or spec.get("method") != "dfly"
                or type(spec.get("num_speculative_tokens")) is not int
                or spec["num_speculative_tokens"] != 5):
            raise ValueError("HiSparse planner supports explicit DFly5 only")


def _validate_owned(config):
    from . import hisparse_config_compat
    hisparse_config_compat._validate_planner_config(config)
    validate_config_scope(config, final=True)



def _is_indexpool_spec(spec):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole
    return (isinstance(spec, MLAAttentionSpec)
            and spec.cache_role is SparseCacheRole.INDEXER
            and spec.tokens_per_state > 1)


def _indexpool_groups(config, specs):
    """Native group objects, preserving independent logical 64/1024/ring blocks.

    The LCM is a PHYSICAL byte stride, not a fabricated common scheduler or
    kernel block. HiSparse source/hot stays 64; IndexPool remains 64 states per
    ratio*64-token kernel page; tail rings remain CircularBufferSpec. Separate
    HMA groups have separate block tables and account for their own admission.
    This pure placement routine does not certify HCU/config ownership.
    """
    from vllm.v1.core import kv_cache_planning as planning
    from vllm.v1.hisparse.layout import HISPARSE_HOT_SUFFIX, HISPARSE_RESIDENT_SUFFIX
    from vllm.v1.hisparse.runtime import ResolvedHiSparseConfig
    from vllm.v1.kv_cache_interface import (MLAAttentionSpec, SparseCacheRole,
        KVCacheGroupSpec, KVCacheGroupRole, UniformTypeKVCacheSpecs,
        HiSparseHotSpec, HiSparseResidentSpec)
    source = {n: v for n, v in specs.items() if isinstance(v, MLAAttentionSpec)
              and v.cache_role is SparseCacheRole.SPARSE}
    indexers = {n: v for n, v in specs.items() if isinstance(v, MLAAttentionSpec)
                and v.cache_role is SparseCacheRole.INDEXER}
    if not source or not indexers or {v.block_size for v in source.values()} != {64}:
        raise ValueError('IndexPool HiSparse requires real 64-token sparse sources and indexers')
    if any(v.tokens_per_state != 1 for v in source.values()):
        raise ValueError('Compressed HiSparse source states are not supported')
    source_spec = UniformTypeKVCacheSpecs.from_specs(source)
    if source_spec is None:
        raise ValueError('IndexPool source specs are not uniform')
    groups = [KVCacheGroupSpec(list(source), source_spec, host_resident=True,
        enable_kv_transfer=True, role=KVCacheGroupRole.HISPARSE_SOURCE)]
    for spec in indexers.values():
        if _is_indexpool_spec(spec):
            kernel_page = spec.copy_with_new_block_size(int(spec.tokens_per_state)*64).page_size_bytes
            if spec.get_block_stride_alignment() % kernel_page:
                raise ValueError('IndexPool specs must be worker-resolved with physical kernel-page alignment')
    # One real indexer spec per group avoids inventing a common kernel ABI.
    groups.extend(KVCacheGroupSpec([n], v, enable_kv_transfer=True,
        role=KVCacheGroupRole.HISPARSE_INDEXER) for n, v in indexers.items())
    topk = max(v.topk_buffer_width if v.topk_buffer_width is not None else
               config.model_config.hf_text_config.index_topk for v in source.values())
    resolved = ResolvedHiSparseConfig.from_vllm_config(config, topk, 64)
    units = []
    for name, spec in source.items():
        if spec.is_index_group_leader or not units:
            units.append([])
        units[-1].append((name, spec))
    for unit in units:
        pages = {v.page_size_bytes for _, v in unit}
        if len(pages) != 1:
            raise ValueError('Index-sharing HiSparse layers need equal physical pages')
        page = pages.pop()
        groups.append(KVCacheGroupSpec([n+HISPARSE_RESIDENT_SUFFIX for n, _ in unit],
            HiSparseResidentSpec(block_size=64, page_size=page), enable_kv_transfer=False))
        groups.append(KVCacheGroupSpec([n+HISPARSE_HOT_SUFFIX for n, _ in unit],
            HiSparseHotSpec(block_size=64, page_size=page,
                blocks_per_request=(resolved.device_buffer_size+63)//64), enable_kv_transfer=False))
    regular = {n: v for n, v in specs.items()
        if not isinstance(v, (MLAAttentionSpec, HiSparseResidentSpec, HiSparseHotSpec))}
    # No MLA in this subset: native regular SWA/ring grouping, not a hidden
    # attention_config or config view with HiSparse removed.
    if regular:
        groups.extend(planning.get_kv_cache_groups(config, regular))
    return groups


def _resolve_indexpool_specs(config, specs, attn_layers):
    """Pure real-spec transformation; no Core mutation/admission bypass.

    Only allocation alignment changes for actual IndexPoolCache specs. Logical
    block size, tokens_per_state, bytes/page, dtype and backend remain intact.
    Derived resident/hot objects are real Core specs. Caller inputs are immutable.
    """
    from vllm.v1.worker.utils import select_common_block_size
    from vllm.v1.kv_cache_interface import (MLAAttentionSpec, SparseCacheRole,
        HiSparseHotSpec, HiSparseResidentSpec, CircularBufferSpec)
    from .indexpool_cache import IndexPoolCache, IndexPoolBackend, POOL_PAGE_SIZE, tail_ring_size
    result = dict(specs)
    for name, spec in specs.items():
        if _is_indexpool_spec(spec):
            layer = attn_layers[name]
            ratio = spec.tokens_per_state
            expected = math.lcm(config.cache_config.block_size, ratio*POOL_PAGE_SIZE,
                tail_ring_size(ratio, config.num_speculative_tokens))
            if (type(layer) is not IndexPoolCache or layer.tail
                    or layer.get_attn_backend() is not IndexPoolBackend
                    or type(ratio) is not int or ratio not in (2, 4, 8, 16) or layer.ratio != ratio
                    or spec.block_size != expected or spec.head_size != 132
                    or str(spec.dtype) != 'torch.uint8' or spec.num_kv_heads != 1
                    or spec.dcp_sharded):
                raise ValueError('Unreviewed IndexPool state/kernel ABI')
            tail_name = name.removesuffix('.k_cache') + '.tail_cache'
            tail = specs.get(tail_name)
            tail_layer = attn_layers.get(tail_name)
            if (not name.endswith('.k_cache') or type(tail_layer) is not IndexPoolCache
                    or not tail_layer.tail or tail_layer.ratio != ratio
                    or not isinstance(tail, CircularBufferSpec)
                    or tail.block_size != tail_ring_size(ratio, config.num_speculative_tokens)
                    or tail.head_size != 131 or str(tail.dtype) != 'torch.float32'):
                raise ValueError('IndexPool requires its intact paired speculative tail ring')
            kernel = select_common_block_size(spec.block_size, [IndexPoolBackend], [spec])
            if kernel != ratio*64 or spec.page_size_bytes != (spec.block_size//ratio)*132:
                raise ValueError('IndexPool must retain 64-state kernel pages')
            alignment = math.lcm(spec.get_block_stride_alignment(),
                spec.copy_with_new_block_size(kernel).page_size_bytes)
            result[name] = replace(spec, block_stride_alignment=alignment)
        elif isinstance(spec, MLAAttentionSpec):
            backend = attn_layers[name].get_attn_backend()
            if (spec.block_size != 64 or spec.tokens_per_state != 1
                    or select_common_block_size(64, [backend], [spec]) != 64):
                raise ValueError('HiSparse token source/full indexer must retain its 64-token ABI')
            if spec.cache_role not in (SparseCacheRole.SPARSE, SparseCacheRole.INDEXER):
                raise ValueError('Unreviewed sparse cache role')
    groups = _indexpool_groups(config, result)
    result.update({n: g.kv_cache_spec for g in groups
        if isinstance(g.kv_cache_spec, (HiSparseHotSpec, HiSparseResidentSpec))
        for n in g.layer_names})
    return result


def resolve_hisparse_specs(vllm_config, kv_cache_spec, attn_layers):
    """Owned worker entry point. Native non-target delegation is exact once."""
    check_health()
    original = _state['spec_original']
    if not _is_connector(vllm_config.kv_transfer_config):
        return original(vllm_config, kv_cache_spec, attn_layers)
    _validate_owned(vllm_config)
    if not any(_is_indexpool_spec(v) for v in kv_cache_spec.values()):
        return original(vllm_config, kv_cache_spec, attn_layers)
    return _resolve_indexpool_specs(vllm_config, kv_cache_spec, attn_layers)


def _make_types(planning, layout, native_connector):
    class HCUHiSparseConnector(native_connector.HiSparseConnector):
        def __init__(self, vllm_config, role, kv_cache_config):
            # Native interfaces above are used for real HMA/layout validation.
            # Never enter its scheduler/worker constructor before qualification.
            from .hisparse_config_compat import validate_hisparse_runtime
            validate_hisparse_runtime(vllm_config)
            super().__init__(vllm_config, role, kv_cache_config)

    class HCUHiSparsePlanner(planning.DefaultKVCacheConfigBuilder):
        def get_kv_cache_groups(self, vllm_config, kv_cache_spec):
            _validate_owned(vllm_config)
            # Preserve Core's model grouping conflict and exact partition checks.
            if any(_is_indexpool_spec(v) for v in kv_cache_spec.values()):
                if vllm_config.model_config.supports_kv_cache_grouping:
                    raise ValueError('HiSparse refuses model-owned grouping')
                return _indexpool_groups(vllm_config, kv_cache_spec)
            return super().get_kv_cache_groups(vllm_config, kv_cache_spec)

        def get_kv_cache_config_from_groups(self, vllm_config, kv_cache_groups, num_blocks):
            _validate_owned(vllm_config)
            if type(num_blocks) is not int or num_blocks <= 0:
                raise ValueError("HiSparse planner requires a positive block count")
            if not kv_cache_groups:
                raise ValueError("HiSparse planner requires real sparse/indexer specs")
            pool_bytes = planning._pool_bytes_per_block(vllm_config, kv_cache_groups)
            result = layout.get_hisparse_kv_cache_config(
                vllm_config, kv_cache_groups, num_blocks * pool_bytes,
                _host_bytes(vllm_config.kv_transfer_config))
            if result.num_blocks != num_blocks:
                raise ValueError("HiSparse planner num_blocks/override contract conflict")
            return result

    return HCUHiSparseConnector, HCUHiSparsePlanner


def __getattr__(name):
    # Native external-connector resolution uses this real class, not a registry
    # mutation or a has_connector override. Import alone constructs nothing.
    if name == _CONNECTOR:
        check_health()
        return _state["connector_cls"]
    raise AttributeError(name)


def install_compat() -> bool:
    """Transactional builder/spec resolver publication; exact non-target delegation."""
    global _state, _failure, _publishing
    with _lock:
        if _publishing:
            _failure = "Recursive HiSparse planner publication"
            raise RuntimeError(_failure)
        if _state is not None:
            check_health()
            return False
        if _failure is not None:
            raise RuntimeError(_failure)
        _publishing = True
        module = wrapper = original = None
        publications = []
        try:
            from vllm.platforms import current_platform
            from hcu_adapter.bridges.vllm import HcuPlatform
            if type(current_platform) is not HcuPlatform:
                raise RuntimeError("HiSparse planner requires canonical HCU platform")
            modules = []
            members = []
            for name, expected in _ABI.items():
                mod = import_module(name)
                digest = hashlib.sha256(ast.dump(ast.parse(inspect.getsource(mod)),
                    include_attributes=False).encode()).hexdigest()
                if digest != expected:
                    raise RuntimeError(f"HiSparse planner reviewed Core ABI changed: {name}")
                modules.append(mod)
                members.extend(_members(mod))
            module, planning, layout = modules[:3]
            original = module.get_kv_cache_config_builder
            if (tuple(inspect.signature(original).parameters) != ("vllm_config",)
                    or hasattr(original, "__wrapped__")):
                raise RuntimeError("HiSparse planner resolver signature/ownership changed")
            binding, worker = modules[4], modules[13]
            cache_module = modules[14]
            base = cache_module.DeepseekV32IndexerCache
            if hashlib.sha256(ast.dump(ast.parse(inspect.getsource(base)), include_attributes=False).encode()).hexdigest() != _INDEXPOOL_BASE_ABI:
                raise RuntimeError("IndexPool native cache consumer ABI changed")
            members.append((cache_module, "DeepseekV32IndexerCache", base, None))
            for name, fn in vars(base).items():
                if inspect.isfunction(fn):
                    members.append((base, name, fn, (fn, _fingerprint(fn))))
            spec_original = binding.resolve_hisparse_specs
            if (worker.resolve_hisparse_specs is not spec_original
                    or tuple(inspect.signature(spec_original).parameters) != ('vllm_config', 'kv_cache_spec', 'attn_layers')
                    or hasattr(spec_original, '__wrapped__')):
                raise RuntimeError('HiSparse worker resolver signature/alias ownership changed')
            connector_cls, builder_cls = _make_types(planning, layout, modules[8])

            @wraps(original)
            def resolve(vllm_config):
                check_health()
                if not _is_connector(vllm_config.kv_transfer_config):
                    return original(vllm_config)
                _validate_owned(vllm_config)
                native = original(vllm_config)
                if type(native) is not planning.DefaultKVCacheConfigBuilder:
                    raise RuntimeError("HiSparse refuses another platform builder owner")
                return builder_cls()

            # Intentional owned function slots, never original __globals__ edits.
            slots = ((module, 'get_kv_cache_config_builder', original, resolve),
                     (binding, 'resolve_hisparse_specs', spec_original, resolve_hisparse_specs),
                     (worker, 'resolve_hisparse_specs', spec_original, resolve_hisparse_specs))
            members = [entry for entry in members
                       if not any(entry[0] is owner and entry[1] == key for owner, key, _, _ in slots)]
            plugin_members = [(cls, name, fn, _fingerprint(fn))
                              for cls in (connector_cls, builder_cls)
                              for name, fn in vars(cls).items() if inspect.isfunction(fn)]
            _state = {"pid": os.getpid(), "resolver_module": module,
                      "publications": slots, "spec_original": spec_original,
                      "spec_original_fp": _fingerprint(spec_original),
                      "wrapper": resolve, "wrapper_fp": _fingerprint(resolve),
                      "original": original, "original_fp": _fingerprint(original),
                      "members": members,
                      "plugin_members": plugin_members, "modules": modules,
                      "platform": current_platform, "connector_cls": connector_cls,
                      "builder_cls": builder_cls}
            _state["helpers"] = [(name, fn, _fingerprint(fn))
                for name, fn in globals().copy().items()
                if inspect.isfunction(fn) and fn.__module__ == __name__]
            _state["aliases"] = [(mod, key, value) for mod in modules
                for key, value in vars(mod).items()
                if not any(mod is owner and key == slot for owner, slot, _, _ in slots)
                and getattr(value, "__module__", None) in _ABI
                and getattr(value, "__module__", None) != mod.__name__
                and (inspect.isfunction(value) or inspect.isclass(value))]
            wrapper = resolve
            for owner, key, before, replacement in slots:
                publications.append((owner, key, before, replacement))
                setattr(owner, key, replacement)
            check_health()
            return True
        except BaseException as exc:
            for owner, key, before, replacement in reversed(publications):
                if vars(owner).get(key) is replacement:
                    setattr(owner, key, before)
            _state = None
            _failure = f"HiSparse planner install failed: {exc}"
            raise
        finally:
            _publishing = False


def plan_hisparse(vllm_config, kv_cache_specs, available_memory):
    """Run REAL Core global grouping + cross-worker planner, allocating no tensors.

    The real engine must first resolve the cache layout from worker backends.
    Specs must already be worker-resolved (including native resident/hot entries,
    from resolve_hisparse_specs). No synthetic draft cache, omitted KV-transfer,
    spec replacement, automatic activation, or runtime certificate is supplied.
    Per-worker specs are copied because native grouping may normalize them.
    Returned objects are actual Core KVCacheConfig metadata. Core retains native
    num_gpu_blocks_override and memory-fit/null-block/cross-rank semantics.
    """
    _validate_owned(vllm_config)
    if (not kv_cache_specs or len(kv_cache_specs) != len(available_memory)
            or any(type(n) is not int or n <= 0 for n in available_memory)):
        raise ValueError("One positive byte budget is required per worker spec map")
    planning = _state["modules"][1]
    specs = deepcopy(kv_cache_specs)
    groups = planning.get_global_kv_cache_groups(vllm_config, specs)
    result = planning.get_kv_cache_configs(vllm_config, specs, list(available_memory), groups)
    _validate_owned(vllm_config)
    return result


def describe_capacity(vllm_config, kv_cache_config) -> dict:
    """Read native group/page/tensor capacity; not a throughput/engine guarantee."""
    _validate_owned(vllm_config)
    from vllm.v1.attention.backend import max_decode_query_len
    from vllm.v1.hisparse.runtime import ResolvedHiSparseConfig
    from vllm.v1.kv_cache_interface import HiSparseHotSpec, UniformTypeKVCacheSpecs
    source, = [g for g in kv_cache_config.kv_cache_groups if g.host_resident]
    source_spec = source.kv_cache_spec
    if not isinstance(source_spec, UniformTypeKVCacheSpecs):
        raise ValueError("HiSparse capacity requires the native uniform source spec")
    # Native grouping sizes the union using the padded per-layer buffer width,
    # NOT necessarily the HF model's unpadded index_topk (especially for VL).
    model_topk = vllm_config.model_config.hf_text_config.index_topk
    topk = max(s.topk_buffer_width if s.topk_buffer_width is not None else model_topk
               for s in source_spec.kv_cache_specs.values())
    block = source_spec.block_size
    resolved = ResolvedHiSparseConfig.from_vllm_config(vllm_config, topk, block)
    cdiv = lambda n, d: (n + d - 1) // d
    rows = []
    admission = 0
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        pages = (spec.max_memory_usage_pages(vllm_config)
                 if isinstance(spec, UniformTypeKVCacheSpecs) else
                 cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes))
        if not group.host_resident:
            admission += pages
        rows.append({"layers": list(group.layer_names), "host": group.host_resident,
                     "page_bytes": spec.page_size_bytes, "admission_blocks": pages,
                     "hot": isinstance(spec, HiSparseHotSpec)})
    pages_per_full_request = cdiv(vllm_config.model_config.max_model_len, block)
    seqs = vllm_config.scheduler_config.max_num_seqs
    return {"runtime_admitted": False, "decode_query_len": max_decode_query_len(vllm_config),
            "lookahead_tokens": vllm_config.num_lookahead_tokens,
            "max_in_flight_tokens": vllm_config.max_in_flight_tokens,
            "model_index_topk": model_topk, "planner_topk_buffer_width": topk,
            "hot_minimum_rows": resolved.max_union_rows,
            "hot_buffer_rows": resolved.device_buffer_size,
            "device_layout": vllm_config.cache_config.get_resolved_kv_cache_layout().name,
            "host_layout": "LBNHC",  # Fixed by the pinned native placement helper.
            "host_budget_bytes": _host_bytes(vllm_config.kv_transfer_config),
            "host_block_stride": kv_cache_config.hisparse_host_block_stride,
            "host_num_blocks": kv_cache_config.hisparse_host_num_blocks,
            "native_min_host_blocks": pages_per_full_request + 2,
            # Conservative envelope: distinct full contexts, null + one COW each.
            # Not a new admission policy and not guaranteed concurrency.
            "distinct_context_host_envelope_blocks": seqs * (pages_per_full_request + 1) + 1,
            "device_num_blocks": kv_cache_config.num_blocks,
            "device_admission_blocks_per_request": admission,
            "groups": rows,
            "tensors": [dict(size=t.size, layers=list(t.layers), offset=t.offset,
                             layer_stride=t.layer_stride, block_stride=t.block_stride,
                             host_resident=t.host_resident)
                        for t in kv_cache_config.kv_cache_tensors]}
