# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.engine.core
"""

PATCHES = [
    (
"""
from vllm.version import __version__ as VLLM_VERSION
""",
"""
from vllm.version import __version__ as VLLM_VERSION
from vllm_hcu.platforms import envs as henvs
""",              
    ),
    
    (
"""
                    self.input_queue.put_nowait((request_type, request))
""",
"""
                    self.input_queue.put_nowait((request_type, request))
                    if henvs.VLLM_HCU_USE_DP_CONNECTOR and \
                    isinstance(request, tuple) and self.scheduler.connector is not None:
                        req, _ = request
                        if request_type == EngineCoreRequestType.ADD:
                            self.scheduler.connector.register_req(req.request_id)
""",
    )
]
