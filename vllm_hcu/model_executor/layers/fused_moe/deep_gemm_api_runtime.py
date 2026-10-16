# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU DeepGEMM package/API compatibility for upstream lazy utilities."""

import functools
import importlib
import inspect


@functools.cache
def accepts_ue8m0(impl):
    try:
        params = inspect.signature(impl).parameters
    except (TypeError, ValueError):
        return True
    return "disable_ue8m0_cast" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def install_api(target):
    original_import = target._import_deep_gemm
    original_get = target.get_mk_alignment_for_contiguous_layout
    original_set = target.set_mk_alignment_for_contiguous_layout

    @functools.wraps(original_import)
    def import_deep_gemm():
        module = original_import()
        if module is not None:
            return module
        try:
            return importlib.import_module("deepgemm")
        except ImportError:
            return None

    @functools.wraps(original_get)
    def get_alignment():
        target._lazy_init()
        if (
            getattr(target, "_get_mk_alignment_for_contiguous_layout_impl", None)
            is None
        ):
            return [16, 16]
        return original_get()

    @functools.wraps(original_set)
    def set_alignment(value):
        dg = import_deep_gemm()
        if dg is not None and hasattr(dg, "set_mk_alignment_for_contiguous_layout"):
            return original_set(value)

    def make_grouped(name, impl_name):
        original = getattr(target, name)

        @functools.wraps(original)
        def grouped(*args, **kwargs):
            target._lazy_init()
            impl = getattr(target, impl_name)
            if impl is None:
                return original(*args, **kwargs)
            if accepts_ue8m0(impl):
                kwargs["disable_ue8m0_cast"] = not target.is_deep_gemm_e8m0_used()
            return impl(*args, **kwargs)

        return grouped

    target._import_deep_gemm = import_deep_gemm
    target.get_mk_alignment_for_contiguous_layout = get_alignment
    target.set_mk_alignment_for_contiguous_layout = set_alignment
    for name, impl in (
        ("m_grouped_fp8_gemm_nt_contiguous", "_grouped_impl"),
        ("m_grouped_fp8_fp4_gemm_nt_contiguous", "_grouped_fp4_impl"),
        ("fp8_m_grouped_gemm_nt_masked", "_grouped_masked_impl"),
    ):
        setattr(target, name, make_grouped(name, impl))
