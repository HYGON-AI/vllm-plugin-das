# Public INT8 publication and process boundaries

`tests/vllm033/test_aiter_int8_publication.py` exercises the independent INT8
owner through the installed extension entry point, Adapter's public activation
API and health sink, and Core's real INT8 backend registry. Hardware discovery
and native acquisition use explicit CPU fixtures; no vendor kernel runs.

The matrix covers preparation failure, failure after a genuine registry write,
and the first published health check. Each boundary receives a regular error,
`KeyboardInterrupt`, and an exception whose string conversion raises. Failed
publication must stop later owners, preserve fatal typing and reject discovery,
direct reactivation, health and cache-factor retries without another write.
The injected fault is removed before retries; neither preparation nor
registration may run again, including after a partial publication.

The success controls check idempotence and two synchronized owner callers.
Actual bounded CPU forks verify that an in-flight or failed owner refuses the
child, while a healthy idle child must explicitly publish its own registration.
The parent's descriptor, reports and health remain valid. Forks are injected
only by the original parent, so a broken child guard cannot recursively fork.

Run against compatible installed Core, Adapter and Plugin wheels:

```bash
HCU_ADAPTER_HOST=none VLLM_PLUGINS=hcu_adapter \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 \
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
python -I -B -m pytest -q tests/vllm033/test_aiter_int8_publication.py
```

The checked-in portable CI tier also collects these tests. This matrix validates
publication and process ownership; native ABI, numerical/Graph behavior,
full engine startup, models, collectives and shutdown require their own gates.
Framework versions are selected by the actual Core/provider/deployment stack.
