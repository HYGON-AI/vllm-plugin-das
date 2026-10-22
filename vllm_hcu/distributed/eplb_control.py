# SPDX-License-Identifier: Apache-2.0
"""Process-local EPLB recording scope shared by policy and runtime layers."""
from contextvars import ContextVar

RECORD_ONLY = ContextVar("hcu_eplb_record_only", default=False)
