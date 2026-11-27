# SPDX-License-Identifier: Apache-2.0
"""INT8 DeepEP low-latency -> masked HIPC W4A8 -> weighted combine.

No framework registration or permanent weight transformation here. The caller
owns shared experts and the model's outer routed scaling. All count-dependent
work stays on device, including activation and changed-input graph replay.
"""
from __future__ import annotations
from collections.abc import Callable
import threading

import torch


def _verify_gemm_config(device, geometry):
    if geometry != (16, 6144, 3072, 2048):
        return None
    gpu = torch.cuda.get_device_properties(device)
    if gpu.gcnArchName.split(":", 1)[0] != "gfx938" or gpu.multi_processor_count != 64:
        return None
    try:
        from deepgemm.configs.w4a8_masked_hipc import select_w4a8_hipc_masked_config
    except ImportError:
        return None
    # These MiniMax projection shapes use the vendor's generic config.
    # Use its M56 tiles for VERIFY/M128 to shorten the send-arrival tail.
    experts, hidden, intermediate, _ = geometry
    gate = select_w4a8_hipc_masked_config(56, 2 * intermediate, hidden, experts)
    down = select_w4a8_hipc_masked_config(56, hidden, intermediate, experts)
    tested = {"MODE": 1642, "PERSISTENT_MULTIPLIER": 4}
    return tested if gate == tested and down == tested else None


class MaskedExperts:
    def __init__(self, *, device, experts, hidden, intermediate, capacity,
                 hipc_verify_tuned=True, lightop_moe_tuned=False):
        from deepgemm import m_grouped_w4a8_gemm_nt_masked_hipc
        from lightop.activation.swiglu_oai_int8_quant import swiglu_oai_int8_quant_masked_flat
        if experts not in (16, 17) or hidden % 128 or intermediate % 128 or capacity <= 0:
            raise ValueError("unsupported MiniMax masked workspace")
        self.geometry = experts, hidden, intermediate, capacity
        self.gemm = m_grouped_w4a8_gemm_nt_masked_hipc
        self.verify_gemm_config = (
            _verify_gemm_config(device, self.geometry) if hipc_verify_tuned else None
        )
        self._verify_gemm_rows = (128,)
        self._activation_tuned_rows = (64, 128)
        self.activation = swiglu_oai_int8_quant_masked_flat
        arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
        self.tune_activation_grid = (
            lightop_moe_tuned
            and arch.startswith("gfx938")
            and (experts, intermediate, capacity) == (16, 3072, 2048)
        )
        self.gateup = torch.empty((experts, capacity, 2 * intermediate), device=device,
                                  dtype=torch.bfloat16)
        self.quant = torch.empty((experts, capacity, intermediate), device=device, dtype=torch.int8)
        self.scale = torch.empty((experts, capacity, 1), device=device, dtype=torch.float32)
        # OAI has finished reading gateup before GEMM2 begins. MiniMax has
        # 2*intermediate == hidden, so both BF16 temporaries can share storage.
        # The stream-owned executor serializes combine before the next layer.
        self.down = (self.gateup if hidden==2*intermediate else
                     torch.empty((experts, capacity, hidden),device=device,dtype=torch.bfloat16))

    def configure_expanded_verify(self, rows, bind_gemm):
        """Bind the offline tuned kernels to this workspace only."""
        if rows not in (64, 80, 96, 144):
            raise ValueError(f"Unsupported expanded Verify8 rows {rows}")
        capacity = 2048 if rows == 144 else 1024
        if self.geometry != (16, 6144, 3072, capacity):
            raise ValueError(f"Unsupported expert geometry {self.geometry}")
        config = {"MODE": 1642, "PERSISTENT_MULTIPLIER": 5}
        self.gemm = bind_gemm(self.gemm, config)
        self.verify_gemm_config = config
        self._verify_gemm_rows = (rows,)
        self._activation_tuned_rows = (rows,)
        self.tune_activation_grid = True

    def __call__(self, x, counts, weights, expected_m, *, local_tokens=None):
        q, scale = x
        e, h, i, cap = self.geometry
        w13, w2, s13, s2 = weights
        if (q.shape != (e, cap, h) or q.dtype != torch.int8
                or scale.shape != (e, cap, 1) or scale.dtype != torch.float32
                or counts.shape != (e,) or counts.dtype != torch.int32
                or w13.shape != (e, 2*i, h//2) or w2.shape != (e, h, i//2)
                or s13.shape != (e, 2*i, 1) or s2.shape != (e, h, 1)):
            raise ValueError("masked activation/weight/count geometry mismatch")
        if any(t.device != q.device for t in (scale, counts, w13, w2, s13, s2, self.down)):
            raise ValueError("masked tensors must share one device")
        # Padding is neither read by GEMM2 nor reduced by combine. The vendor
        # activation supports live GPU counts and may leave tail rows undefined.
        gemm_options = {}
        if local_tokens in self._verify_gemm_rows and self.verify_gemm_config is not None:
            gemm_options["config"] = self.verify_gemm_config
        self.gemm((q, scale), (w13, s13), self.gateup, counts, expected_m, **gemm_options)
        progress=getattr(self,'gpu_progress',None)
        if progress:progress.moe(progress.moe_index-1,3)
        launch = {"grid_blocks": 512} if (
            self.tune_activation_grid and local_tokens in self._activation_tuned_rows
        ) else {}
        self.activation(self.gateup, counts, alpha=1.702, beta=1.0, limit=7.0,
                        output=self.quant, scales=self.scale, **launch)
        if progress:progress.moe(progress.moe_index-1,4)
        self.gemm((self.quant, self.scale), (w2, s2), self.down, counts, expected_m, **gemm_options)
        if progress:progress.moe(progress.moe_index-1,5)
        return self.down


class LowLatencyMoE:
    """One stream-ordered transport/workspace per rank, shared by serial layers."""
    def __init__(self, group, *, device, hidden=6144, intermediate=3072,
                 physical_experts=128, max_tokens=256,
                 hipc_verify_tuned=True, lightop_moe_tuned=False):
        import deep_ep
        if group.size() != 8 or physical_experts not in (128, 136) or max_tokens <= 0:
            raise ValueError("MiniMax low-latency requires EP8")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("initialize low-latency workspace before graph capture")
        self.physical_experts = physical_experts
        self.max_tokens = max_tokens
        self.hidden = hidden
        self.device = torch.device(device)
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self.lock = threading.Lock()
        self.poisoned = False
        from triton_tmp_dir.minimax31_route_checks import route_checks
        self.route_check_backend='fused_triton_v1'
        self.route_checks=route_checks
        self.route_check_flags=torch.empty(2,device=self.device,dtype=torch.bool)
        self.route_check_scalars=(self.route_check_flags[0],self.route_check_flags[1])
        rdma = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            max_tokens, hidden, 8, physical_experts, num_topk=4)
        self.buffer = deep_ep.Buffer(group, 0, rdma, low_latency_mode=True,
                                     num_qps_per_rank=physical_experts//8)
        self.experts = MaskedExperts(device=device, experts=physical_experts//8,
            hidden=hidden, intermediate=intermediate, capacity=8*max_tokens,
            hipc_verify_tuned=hipc_verify_tuned, lightop_moe_tuned=lightop_moe_tuned)

    def __call__(self, hidden, ids, route_weights, weights, *,
                 before_dispatch_recv: Callable[[], None] | None = None,
                 before_combine_recv: Callable[[], None] | None = None):
        """Queue optional shared work on this stream between each send and receive."""
        if self.poisoned:
            raise RuntimeError("failed low-latency executor; restart consumer")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("low-latency workspace is non-reentrant")
        try:
            if (hidden.ndim != 2 or hidden.shape[1] != self.hidden
                    or hidden.shape[0] > self.max_tokens or hidden.dtype != torch.bfloat16
                    or hidden.device != self.device or ids.shape != (hidden.shape[0], 4)
                    or ids.dtype != torch.int64 or route_weights.shape != ids.shape
                    or route_weights.dtype != torch.float32
                    or any(not t.is_contiguous() for t in (hidden, ids, route_weights))
                    or torch.cuda.current_stream(hidden.device).cuda_stream != self.stream):
                raise ValueError("unsupported low-latency inputs/capacity/stream")
            progress=getattr(self,'gpu_progress',None)
            slot=progress.begin_moe(ids,route_weights,self.physical_experts) if progress else None
            self.route_checks(ids,route_weights,self.route_check_flags,self.physical_experts)
            torch._assert_async(self.route_check_scalars[0],
                                "invalid physical expert ID")
            torch._assert_async(self.route_check_scalars[1],
                                "invalid route/padding weights")
            received, counts, handle, event, hook = self.buffer.low_latency_dispatch(
                hidden, ids, route_weights, self.max_tokens, self.physical_experts,
                quant_type=1, quant_group_size=0, async_finish=False, return_recv_hook=True)
            if progress:progress.moe(slot,1)
            try:
                if before_dispatch_recv is not None:
                    before_dispatch_recv()
            finally:
                hook()
            if progress:progress.moe(slot,2)
            # Expected occupancy only selects launch geometry; actual GPU counts
            # always mask GEMM and activation, including zero-token experts.
            expected_m = max(1, (hidden.shape[0]*8*4 + self.physical_experts-1)//self.physical_experts)
            output = self.experts(
                received, counts, weights, expected_m, local_tokens=hidden.shape[0],
            )
            combined, event, hook = self.buffer.low_latency_combine(
                output, ids, route_weights, handle, async_finish=False,
                return_recv_hook=True)
            if progress:progress.moe(slot,6)
            try:
                if before_combine_recv is not None:
                    before_combine_recv()
            finally:
                # Once send is submitted, enqueue receive even if the work failed.
                hook()
            if progress:progress.moe(slot,7)
            return combined
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.lock.release()
