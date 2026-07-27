# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU worker-side operator optimization runtime adapters.

Importing this package has no effect on vLLM.  Dispatchers explicitly arm the
exact target module for each adapter and call its ``apply_to_module`` callback.
"""

__all__: list[str] = []
