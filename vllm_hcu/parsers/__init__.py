# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Parser implementations owned by the HCU plugin.

Modules in this package import their vLLM parent classes lazily: the runtime
patch coordinator imports them only after the corresponding vLLM registry has
finished importing.
"""

__all__: list[str] = []
