# SPDX-License-Identifier: Apache-2.0
"""HCU-owned runtime compatibility implementations.

The package is intentionally side-effect free. Exact-target callbacks import
individual modules only after the corresponding vLLM module is available.
"""
