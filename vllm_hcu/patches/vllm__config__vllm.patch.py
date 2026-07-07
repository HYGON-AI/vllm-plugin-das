# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.config.vllm
"""

PATCHES = [
(
'''
import vllm.envs as envs
''',
'''
import vllm.envs as envs
import vllm_hcu.platforms.envs as henvs
''',
),
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
(
'''
        model_config.hf_config = hf_config
        model_config.model_arch_config = model_config.get_model_arch_config()
''',
'''
        model_config.hf_config = hf_config
        model_config.hf_text_config = model_config.hf_config.get_text_config()
        model_config.model_arch_config = model_config.get_model_arch_config()
''',
),
(
'''
            # determine the initial max_cudagraph_capture_size
            max_cudagraph_capture_size = (
                self.compilation_config.max_cudagraph_capture_size
            )
            if max_cudagraph_capture_size is None:
                decode_query_len = 1
                if (
                    self.speculative_config
                    and self.speculative_config.num_speculative_tokens
                ):
                    decode_query_len += self.speculative_config.num_speculative_tokens
                max_cudagraph_capture_size = min(
                    self.scheduler_config.max_num_seqs * decode_query_len * 2, 512
                )
''',
'''
            decode_query_len = 1
            if (
                self.speculative_config
                and self.speculative_config.num_speculative_tokens
            ):
                decode_query_len += self.speculative_config.num_speculative_tokens

            # determine the initial max_cudagraph_capture_size
            max_cudagraph_capture_size = (
                self.compilation_config.max_cudagraph_capture_size
            )
            if max_cudagraph_capture_size is None:
                max_cudagraph_capture_size = min(
                    self.scheduler_config.max_num_seqs * decode_query_len * 2, 512
                )
''',
),
(
'''
            else:
                if self.performance_mode == "interactivity":
                    # Fine-grained CUDA graphs at small batch sizes
                    # for minimal padding overhead
                    interactivity_max = min(max_cudagraph_capture_size, 32)
                    cudagraph_capture_sizes = list(range(1, interactivity_max + 1))
                else:
                    cudagraph_capture_sizes = [
                        i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
                    ]
                if max_cudagraph_capture_size >= 8:
                    # Step size 8 for small batch sizes, up to 256(not included)
                    cudagraph_capture_sizes += list(
                        range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                    )
                if max_cudagraph_capture_size >= 256:
                    # Step size 16 for larger batch sizes
                    cudagraph_capture_sizes += list(
                        range(256, max_cudagraph_capture_size + 1, 16)
                    )
''',
'''
            else:
                use_generic_capture_strides = True
                if self.performance_mode == "interactivity":
                    # Fine-grained CUDA graphs at small batch sizes
                    # for minimal padding overhead
                    interactivity_max = min(max_cudagraph_capture_size, 32)
                    cudagraph_capture_sizes = list(range(1, interactivity_max + 1))
                elif (
                    henvs.VLLM_HCU_USE_CUSTOM_OPS
                    and henvs.VLLM_HCU_ENABLE_REQUEST_CUDAGRAPH_BUCKETS
                ):
                    # Use request-count-oriented CUDA graph batch buckets,
                    # converted to vLLM's token counts. For spec decode, each
                    # request contributes decode_query_len tokens.
                    request_capture_sizes = (
                        list(range(1, 9, 1))
                        + list(range(10, 33, 2))
                        + list(range(40, 65, 4))
                        + list(range(72, 257, 8))
                    )
                    cudagraph_capture_sizes = [
                        req_size * decode_query_len
                        for req_size in request_capture_sizes
                        if req_size * decode_query_len <= max_cudagraph_capture_size
                    ]
                    use_generic_capture_strides = False
                else:
                    cudagraph_capture_sizes = [
                        i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
                    ]
                if use_generic_capture_strides and max_cudagraph_capture_size >= 8:
                    # Step size 8 for small batch sizes, up to 256(not included)
                    cudagraph_capture_sizes += list(
                        range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                    )
                if use_generic_capture_strides and max_cudagraph_capture_size >= 256:
                    # Step size 16 for larger batch sizes
                    cudagraph_capture_sizes += list(
                        range(256, max_cudagraph_capture_size + 1, 16)
                    )
''',
),
]
