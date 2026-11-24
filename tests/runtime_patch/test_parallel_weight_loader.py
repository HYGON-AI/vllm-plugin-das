# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from pathlib import Path

from vllm_hcu.runtime_compat.parallel_weight_loader import call_parallel_weight_loader


class _GGUFParam:
    def load_column_parallel_weight(self, loaded_weight):
        self.loaded = loaded_weight

    def load_row_parallel_weight(self, loaded_weight):
        self.loaded = loaded_weight


class _HCUParam:
    def load_column_parallel_weight(self, loaded_weight, is_quantization=False):
        self.loaded = loaded_weight
        self.is_quantization = is_quantization

    def load_row_parallel_weight(self, loaded_weight, is_quantization=False):
        self.loaded = loaded_weight
        self.is_quantization = is_quantization


def test_gguf_loader_ignores_is_quantization_and_hcu_loader_keeps_it():
    gguf_column = _GGUFParam()
    hcu_column = _HCUParam()
    call_parallel_weight_loader(
        gguf_column.load_column_parallel_weight,
        "column-weight",
        is_quantization=True,
    )
    call_parallel_weight_loader(
        hcu_column.load_column_parallel_weight,
        "column-weight",
        is_quantization=True,
    )

    gguf_row = _GGUFParam()
    hcu_row = _HCUParam()
    call_parallel_weight_loader(
        gguf_row.load_row_parallel_weight,
        "row-weight",
        is_quantization=False,
    )
    call_parallel_weight_loader(
        hcu_row.load_row_parallel_weight,
        "row-weight",
        is_quantization=False,
    )

    assert gguf_column.loaded == "column-weight"
    assert hcu_column.loaded == "column-weight"
    assert hcu_column.is_quantization is True
    assert gguf_row.loaded == "row-weight"
    assert hcu_row.loaded == "row-weight"
    assert hcu_row.is_quantization is False


def test_var_keyword_loader_still_receives_is_quantization():
    class VarKeywordParam:
        def load_column_parallel_weight(self, loaded_weight, **kwargs):
            self.loaded = loaded_weight
            self.kwargs = kwargs

    param = VarKeywordParam()
    call_parallel_weight_loader(
        param.load_column_parallel_weight,
        "weight",
        is_quantization=True,
    )

    assert param.loaded == "weight"
    assert param.kwargs == {"is_quantization": True}


def test_hcu_column_and_row_loaders_use_the_signature_check():
    source = Path(__file__).parents[2].joinpath(
        "vllm_hcu/model_executor/layers/linear.py"
    ).read_text()
    assert source.count("call_parallel_weight_loader(") == 2
    assert "load_column_parallel_weight(loaded_weight=loaded_weight, is_quantization=" not in source
    assert "load_row_parallel_weight(loaded_weight=loaded_weight, is_quantization=" not in source
