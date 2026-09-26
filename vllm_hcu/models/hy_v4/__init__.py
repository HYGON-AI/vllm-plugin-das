# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from typing import TYPE_CHECKING, Any

from vllm.transformers_utils.configs.hy_v4 import HYV4Config

if TYPE_CHECKING:
    from .model import HYV4ForCausalLM
    from .mtp import HYV4MTP


def __getattr__(name: str) -> Any:
    if name == "HYV4ForCausalLM":
        from .model import HYV4ForCausalLM

        return HYV4ForCausalLM
    if name == "HYV4MTP":
        from .mtp import HYV4MTP

        return HYV4MTP
    raise AttributeError(name)


__all__ = ["HYV4Config", "HYV4ForCausalLM", "HYV4MTP"]
