# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
    HcuFlashMLASparseBackend,
)


def test_hcu_flashmla_sparse_supports_glm5next_d512_and_official_d576() -> None:
    assert HcuFlashMLASparseBackend.get_supported_head_sizes() == [512, 576]
