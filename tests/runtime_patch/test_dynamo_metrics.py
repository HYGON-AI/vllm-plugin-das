from __future__ import annotations

import json

from vllm_hcu.runtime_compat import dynamo_metrics


def test_json_safe_filters_callable_config_values() -> None:
    safe = dynamo_metrics._json_safe(
        {"keep": {1, 2}, "drop": {lambda: None}}
    )
    assert safe == {"keep": [1, 2]}


def test_dynamo_metrics_wrapper_handles_callable_config(monkeypatch) -> None:
    from torch._dynamo import utils

    monkeypatch.delattr(utils, dynamo_metrics._PATCH_MARKER, raising=False)
    monkeypatch.setattr(
        utils,
        "_get_dynamo_config_for_logging",
        staticmethod(lambda: (_ for _ in ()).throw(
            TypeError("function is not JSON serializable")
        )),
    )
    assert dynamo_metrics.install_dynamo_metrics_compat()
    assert "dynamic_shapes" in json.loads(utils._get_dynamo_config_for_logging())
