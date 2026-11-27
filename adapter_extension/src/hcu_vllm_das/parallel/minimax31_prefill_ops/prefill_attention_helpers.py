"""Runtime helpers for the qualified MiniMax paged Prefill path."""

from contextlib import contextmanager
import importlib.util
import os

import torch


def load(path, tag):
    spec = importlib.util.spec_from_file_location(f"{tag}._native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_outputs(ref, output, name):
    # v2 does not expose its internal maximum through max_logits.
    for index, label in ((0, "attention"), (2, "LSE")):
        if not torch.equal(ref[index], output[index]):
            raise AssertionError(f"{name} changed {label}")


@contextmanager
def scoped_knobs(plan):
    """The isolated replay provider reads these variables during the launch."""
    settings = {
        "MSA128_FP8_PREFILL_KERNEL": "v2",
        "MSA128_FP8_PREFILL_SLOTS": "2",
        "MSA128_FP8_CTA_TOKENS": None,
        "MSA128_FP8_CTA_LDS_KB": None,
    }
    if plan is not None:
        tokens, slots, lds = plan
        settings.update(MSA128_FP8_CTA_TOKENS=str(tokens),
                        MSA128_FP8_PREFILL_SLOTS=str(slots),
                        MSA128_FP8_CTA_LDS_KB=str(lds))
    previous = {name: os.environ.get(name) for name in settings}

    def apply(values):
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    try:
        apply(settings)
        yield
    finally:
        apply(previous)
