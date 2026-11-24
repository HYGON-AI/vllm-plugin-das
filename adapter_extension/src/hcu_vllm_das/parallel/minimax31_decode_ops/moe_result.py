"""Bind the released LightOP result merge to the MiniMax Decode runner."""
import inspect
import textwrap

from lightop.minimax31_decode_result import merge, supports


def install(block):
    from hcu_vllm_das.parallel.minimax31_decode import MiniMax31DecodeMoERunner

    original = MiniMax31DecodeMoERunner.forward
    source = textwrap.dedent(inspect.getsource(original))
    needle = """    shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
        shared_output, fused_output
    )
"""
    if source.count(needle) != 1:
        raise RuntimeError("unexpected MiniMax routed epilogue")
    source = source.replace(needle, """    if (self.routed_output_transform is None and fused_output_is_reduced
            and og_hidden_dim_post_xform in (None, 6144)
            and m31_supports_result_merge(fused_output, shared_output)):
        zero = self.router.zero_expert_output if isinstance(self.router, ZeroExpertRouter) else None
        combined = m31_merge_result(fused_output, shared_output, zero, self.routed_scaling_factor)
        return self._maybe_reduce_final_output(
            combined, og_hidden_dim_post_xform, fused_output_is_reduced
        )
    m31_log_merge_fallback(self, fused_output, shared_output, fused_output_is_reduced,
                           og_hidden_dim_post_xform)

""" + needle)
    emitted = set()

    def print_once(event, *values):
        if event not in emitted:
            print(event, *values, flush=True)
            emitted.add(event)

    def dispatch(routed, shared, zero, scale):
        print_once("M31_MOE_RESULT_ACTIVE", routed.stride(), None if zero is None else zero.stride(), scale)
        return merge(routed, shared, zero, scale, block=block)

    def fallback(runner, routed, shared, reduced, truncation):
        print_once("M31_MOE_RESULT_FALLBACK", type(runner.router).__name__, reduced,
                   runner.moe_config.tp_size, truncation, type(runner.routed_output_transform).__name__,
                   (tuple(routed.shape), routed.dtype, routed.stride()),
                   None if shared is None else (tuple(shared.shape), shared.dtype, shared.stride()))

    namespace = {}
    globals_for_forward = dict(original.__globals__)
    globals_for_forward.update(
        m31_supports_result_merge=supports,
        m31_merge_result=dispatch,
        m31_log_merge_fallback=fallback,
    )
    exec(compile(source, original.__code__.co_filename,"exec"), globals_for_forward, namespace)
    MiniMax31DecodeMoERunner.forward = namespace["forward"]
    print("M31_FUSED_MOE_RESULT", block, flush=True)
