"""Explicit communication subfeature; discovery does not import Torch/Core."""
from packaging.version import Version
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

from .runtime_sp_policy import custom_sp_requested

ENTRY = "das_hcu_runtime_sp_communication"
CAPABILITY = "runtime-sp-communication"
_ACTION = "runtime-sp-communication-publication"
_PATH = "hcu_vllm_das.parallel.runtime_sp_communicator.RuntimeSPCommunicator"
_platform = None
_lifecycle = ProcessLocalLifecycle(runtime_action=_ACTION, error_type=ExtensionCommitError)


def check_health():
    _lifecycle.require_runtime_healthy()
    if (_platform is None
            or not _platform.owns_device_communicator_selector(select_communicator)):
        raise ExtensionCommitError("runtime SP communicator selector ownership changed")


def select_communicator(default):
    from hcu_adapter.bridges.vllm import require_extension_healthy
    require_extension_healthy(CAPABILITY)
    from vllm.config import get_current_vllm_config_or_none
    return _PATH if custom_sp_requested(get_current_vllm_config_or_none()) else default


def activate(context):
    version = Version(context.vllm_version or "0")
    if version.epoch != 0 or version.release[:2] != (0, 33):
        raise RuntimeError("runtime SP communication requires vLLM 0.33")
    if (ENTRY not in context.requested_extensions
            and CAPABILITY not in context.requested_capabilities):
        raise RuntimeError("runtime SP communication requires an explicit request")

    def publish():
        global _platform
        import inspect
        from hcu_adapter.bridges.vllm import HcuPlatform
        from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

        expected = (
            ("self", inspect.Parameter.empty), ("cpu_group", inspect.Parameter.empty),
            ("device", None), ("device_group", None), ("unique_name", ""),
            ("global_ranks", None), ("global_world_size", None),
            ("tcp_store_group", None), ("use_all2all", False),
        )
        parameters = tuple(inspect.signature(CudaCommunicator.__init__).parameters.values())
        if (tuple((p.name, p.default) for p in parameters) != expected
                or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters)):
            raise RuntimeError("runtime SP communicator constructor ABI changed")
        register = getattr(HcuPlatform, "register_device_communicator_selector", None)
        ownership = getattr(HcuPlatform, "owns_device_communicator_selector", None)
        if not callable(register) or not callable(ownership):
            raise RuntimeError("Adapter lacks the communicator-selector capability")
        try:
            register(select_communicator)
            _platform = HcuPlatform
        except BaseException as error:
            raise ExtensionCommitError("runtime SP selector publication failed") from error

    _lifecycle.run_once(_ACTION, publish)
    check_health()


def extension_spec():
    return ExtensionSpec(
        name=ENTRY, api_version=1, priority=60,
        capabilities=frozenset({CAPABILITY}), activate=activate,
        check_health=check_health,
    )
