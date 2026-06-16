# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.engine.arg_utils
"""

PATCHES = [
(
'''
        model_kwargs = get_kwargs(ModelConfig)
''',
'''
        model_kwargs = get_kwargs(ModelConfig)
        model_kwargs["disable_cascade_attn"]["default"] = True
'''
),

(
'''
    fail_on_environ_validation: bool = False
''',
'''
    fail_on_environ_validation: bool = False

    enable_lightly_cp: bool = ParallelConfig.enable_lightly_cp
    enable_lightly_cplb: bool = ParallelConfig.enable_lightly_cplb
    enable_custom_sp: bool = ParallelConfig.enable_custom_sp
'''
),

(
'''
            _api_process_rank=self._api_process_rank,
''',
'''
            _api_process_rank=self._api_process_rank,
            enable_lightly_cp=self.enable_lightly_cp,
            enable_lightly_cplb=self.enable_lightly_cplb,
            enable_custom_sp=self.enable_custom_sp,
'''
),

(
'''
        parallel_group.add_argument(
            "--data-parallel-backend",
            "-dpb",
            type=str,
            default="mp",
            help='Backend for data parallel, either "mp" or "ray".',
        )
''',
'''
        parallel_group.add_argument(
            "--data-parallel-backend",
            "-dpb",
            type=str,
            default="mp",
            help='Backend for data parallel, either "mp" or "ray".',
        )

        parallel_group.add_argument(
            "--enable-lightly-cp",
            action="store_true",
        )
        parallel_group.add_argument(
            "--enable-lightly-cplb",
            action="store_true",
        )
        parallel_group.add_argument(
            "--enable-custom-sp",
            action="store_true",
        )
'''
),
]
