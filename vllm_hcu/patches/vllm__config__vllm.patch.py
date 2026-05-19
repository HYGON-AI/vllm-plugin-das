# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.config.vllm
"""

PATCHES = [
(
'''
        self.compilation_config.pass_config.log_enabled_passes()
''',
'''
        self.compilation_config.pass_config.log_enabled_passes()

        if self.parallel_config.enable_lightly_cp and not self.model_config.enforce_eager:
            raise ValueError(
                "Lightly context parallel currently only supports the eager mode."
            )

        if self.parallel_config.enable_lightly_cp and self.parallel_config.decode_context_parallel_size > 1:
            raise ValueError(
                "Lightly context parallel and DCP cannot be enabled simultaneously."
            )


'''
),
]
