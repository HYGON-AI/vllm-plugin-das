# SPDX-License-Identifier: Apache-2.0
"""Read-only deployment identity for the two unified Step5 profiles."""
import hashlib
from importlib.metadata import version
from pathlib import Path


class Step5SourceIdentityExtension:
    def step5_source_identity(self):
        import vllm_hcu
        import vllm_hcu.hcu_ops

        root=Path(vllm_hcu.__file__).resolve().parent
        files=['models/step5_text.py',
               'model_executor/layers/fused_moe/step5_wasp128.py',
               'model_executor/layers/fused_moe/deepep_runtime.py',
               'model_executor/layers/fused_moe/step5_wide_fp8_gemm.py',
               'model_executor/layers/fused_moe/dp_pipe_output_pool.py',
               'ops/step5/decode_producers_runtime.py']
        native=Path(vllm_hcu.hcu_ops.__file__).resolve()
        return {'version':version('vllm-hcu'),'root':str(root),
                'native_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),
                'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                 for name in files}}
