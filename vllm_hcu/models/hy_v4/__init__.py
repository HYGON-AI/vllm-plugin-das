# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from typing import TYPE_CHECKING, Any

from .config import HYV4Config, register_hy_v4_config

if TYPE_CHECKING:
    from .model import HYV4ForCausalLM


def __getattr__(name: str) -> Any:
    if name == "HYV4ForCausalLM":
        from .model import HYV4ForCausalLM

        return HYV4ForCausalLM
    raise AttributeError(name)

__all__ = ["HYV4Config", "HYV4ForCausalLM", "register_hy_v4_config"]
