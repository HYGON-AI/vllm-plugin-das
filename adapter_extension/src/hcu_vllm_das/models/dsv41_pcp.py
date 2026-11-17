# SPDX-License-Identifier: Apache-2.0
"""Compatibility path for the single plugin-published PCP model.

The registration owner remains v41_pcp_activation. This alias must not construct
an independent model or bypass the adapter's shared mHC/indexer policies.
"""
from .v41_pcp import HcuV41PcpForCausalLM as HcuDSV41PcpForCausalLM

__all__ = ["HcuDSV41PcpForCausalLM"]
