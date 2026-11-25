"""Opt the isolated BS128 replay process into fused route preparation."""
import inspect
import textwrap

from lightop.minimax31_decode_route import prepare_route


def replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"unexpected MiniMax route method layout for {old!r}")
    return source.replace(old,new)


def install(*, shared_communication=False, assert_inline=False):
    from hcu_vllm_das.moe.minimax31_ll import LowLatencyMoE
    from hcu_vllm_das.parallel.minimax31_decode import MiniMax31DecodeMoERunner

    def apply_patches(cls, name, changes, extra=None):
        original=getattr(cls,name)
        text=textwrap.dedent(inspect.getsource(original))
        for old,new in changes:
            text=replace(text,old,new)
        # Reuse the live module globals, including the stream-owned executor.
        if extra:
            original.__globals__.update(extra)
        namespace={}
        exec(compile(text,original.__code__.co_filename,"exec"),original.__globals__,namespace)
        setattr(cls,name,namespace[name])

    low_latency_overlap_patches = ()
    runner_overlap_patches = ()
    if shared_communication:
        low_latency_overlap_patches = (
            ("route_check=None):", "route_check=None, shared_stream=None):"),
            ("        received, counts, handle, event, hook = self.buffer.low_latency_dispatch(",
             """        main_stream = None
        if shared_stream is not None and before_dispatch_recv is not None:
            main_stream = torch.cuda.current_stream(self.device)
            shared_stream.wait_stream(main_stream)
        received, counts, handle, event, hook = self.buffer.low_latency_dispatch("""),
            ("                before_dispatch_recv()",
             """                if main_stream is None:
                    before_dispatch_recv()
                else:
                    with torch.cuda.stream(shared_stream):
                        before_dispatch_recv()"""),
            ("        if progress:progress.moe(slot,2)",
             """        if main_stream is not None:
            main_stream.wait_stream(shared_stream)
        if progress:progress.moe(slot,2)"""),
            ("        combined, event, hook = self.buffer.low_latency_combine(",
             """        if shared_stream is not None and before_combine_recv is not None:
            main_stream = torch.cuda.current_stream(self.device)
            shared_stream.wait_stream(main_stream)
        combined, event, hook = self.buffer.low_latency_combine("""),
            ("                before_combine_recv()",
             """                if main_stream is None:
                    before_combine_recv()
                else:
                    with torch.cuda.stream(shared_stream):
                        before_combine_recv()"""),
            ("        if progress:progress.moe(slot,7)",
             """        if main_stream is not None:
            main_stream.wait_stream(shared_stream)
        if progress:progress.moe(slot,7)"""),
        )
        runner_overlap_patches = (
            ("route_check=route_check)", "route_check=route_check, shared_stream=self._shared_aux_stream)"),
            ("        return shared_output, routed",
             """        if self._shared_aux_stream is not None:
            shared_output.record_stream(torch.cuda.current_stream(hidden_states.device))
        return shared_output, routed"""),
        )

    apply_patches(LowLatencyMoE,"__call__",(
        ("before_combine_recv: Callable[[], None] | None = None):",
         "before_combine_recv: Callable[[], None] | None = None, route_check=None):"),
        ('''        self.route_checks(ids,route_weights,self.route_check_flags,self.physical_experts)
        torch._assert_async(self.route_check_scalars[0],
                            "invalid physical expert ID")
        torch._assert_async(self.route_check_scalars[1],
                            "invalid route/padding weights")''',
         '''        if route_check is None:
            self.route_checks(ids,route_weights,self.route_check_flags,self.physical_experts)
            torch._assert_async(self.route_check_scalars[0],
                                "invalid physical expert ID")
            torch._assert_async(self.route_check_scalars[1],
                                "invalid route/padding weights")
        elif route_check is not True:
            torch._assert_async(route_check, "invalid physical expert ID or route/padding weights")'''),
        *low_latency_overlap_patches,
    ))
    apply_patches(MiniMax31DecodeMoERunner,"_forward_impl",(
        ("weights = weights.masked_fill(ids<0,0).contiguous()", "route_check = None"),
        ("        from triton_tmp_dir.minimax31_decode_remap import map_experts\n",""),
        ('''        ids = map_experts(ids, layer.decode_gpu_permutation,
                          counters=getattr(layer, "decode_eplb_counters", None))''',
         '''        ids, weights, route_check = prepare_route(
            ids, weights, layer.decode_gpu_permutation,
            counters=getattr(layer, "decode_eplb_counters", None),
            experts=layer.global_num_experts, assert_inline=route_inline_assert,
        )
    else:
        weights = weights.masked_fill(ids<0,0).contiguous()'''),
        ('''before_dispatch_recv=before_dispatch_recv, before_combine_recv=before_combine_recv)''',
         '''before_dispatch_recv=before_dispatch_recv, before_combine_recv=before_combine_recv,
            route_check=route_check)'''),
        *runner_overlap_patches,
    ),{"prepare_route":prepare_route,"route_inline_assert":assert_inline})
    print("M31_FUSED_ROUTE_PREP enabled",flush=True)
