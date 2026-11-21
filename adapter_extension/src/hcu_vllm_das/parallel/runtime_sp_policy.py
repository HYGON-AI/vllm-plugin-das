"""Read the existing custom-SP sidecar without importing legacy patches."""
from collections.abc import Mapping


def custom_sp_requested(config):
    if config is None:
        return False
    extra = getattr(config, "additional_config", None)
    if extra is None:
        return False
    if not isinstance(extra, Mapping):
        raise TypeError("runtime SP requires a mapping additional_config")
    hcu = extra.get("hcu", {})
    if not isinstance(hcu, Mapping):
        raise TypeError("additional_config['hcu'] must be a mapping")
    enabled = hcu.get("enable_custom_sp", False)
    if type(enabled) is not bool:
        raise TypeError("enable_custom_sp must be bool")
    return enabled
