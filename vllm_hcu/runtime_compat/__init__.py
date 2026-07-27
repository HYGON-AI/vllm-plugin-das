# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned runtime compatibility implementations.

The package is intentionally side-effect free. Exact-target callbacks import
individual modules only after the corresponding vLLM module is available.
"""
