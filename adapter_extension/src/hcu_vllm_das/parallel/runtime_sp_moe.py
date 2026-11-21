# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Early SP argument bridge for the unchanged official GLM4/HY3 factories.

The target exposes is_sequence_parallel on its factory, but these two models
do not pass it. runner_cls is too late: the factory already built the parallel
configuration, router and weights. Bind only the two model-local aliases, with
explicit construction scope; never replace the shared Core factory itself.
Retire this bridge when those models expose an equivalent early policy seam.
"""
import functools
import inspect
from importlib import import_module

from hcu_adapter.extensions import ExtensionCommitError
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

from .runtime_sp_attention import model_construction_active

_ACTION = "runtime-sp-moe-factory-publication"
_lifecycle = ProcessLocalLifecycle(runtime_action=_ACTION, error_type=ExtensionCommitError)
_targets = ()
_factory = None
_official_module = None
_original_factory = None


def check_moe_factory_policy():
    _lifecycle.require_runtime_healthy()
    if (not _targets or any(module.FusedMoEFactory is not _factory for module in _targets)
            or _official_module.FusedMoEFactory is not _original_factory):
        raise ExtensionCommitError("runtime SP MoE factory ownership changed")


def _publish():
    global _targets, _factory, _official_module, _original_factory
    official_module = import_module("vllm.model_executor.layers.fused_moe.layer")
    original = official_module.FusedMoEFactory
    targets = tuple(import_module("vllm.model_executor.models." + name)
                    for name in ("glm4_moe", "hy_v3"))
    signature = inspect.signature(original)
    for name in ("prefix", "is_sequence_parallel", "runner_cls"):
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise RuntimeError("Core lacks the early MoE SP factory argument")
    if any(module.FusedMoEFactory is not original for module in targets):
        raise RuntimeError("foreign model-local MoE factory owner")

    @functools.wraps(original)
    def scoped_factory(*args, **kwargs):
        check_moe_factory_policy()
        if not model_construction_active():
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        if bound.arguments.get("prefix", "").endswith(".mlp.experts"):
            from .runtime_sp_moe_runner import RuntimeSPMoERunner
            if bound.arguments.get("runner_cls") not in (None, RuntimeSPMoERunner):
                raise RuntimeError("foreign runtime SP MoE runner policy")
            bound.arguments["is_sequence_parallel"] = True
            bound.arguments["runner_cls"] = RuntimeSPMoERunner
            return original(*bound.args, **bound.kwargs)
        return original(*args, **kwargs)

    # Publish prepared aliases only after all owners and the ABI admit them.
    # A partial write poisons this lifecycle even if someone repairs an alias.
    _targets, _factory = targets, scoped_factory
    _official_module, _original_factory = official_module, original
    try:
        for module in targets:
            module.FusedMoEFactory = scoped_factory
    except BaseException as error:
        raise ExtensionCommitError("runtime SP MoE factory publication incomplete") from error
    check_moe_factory_policy()


def install_moe_factory_policy():
    """Install an inactive-by-default, scoped pre-construction argument bridge."""
    _lifecycle.run_once(_ACTION, _publish)
    check_moe_factory_policy()
