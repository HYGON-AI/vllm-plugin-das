"""Opt-in FP8 metadata and bandwidth changes on the pinned Prefill service."""
import inspect
import textwrap

import torch
import triton
import triton.language as tl

from hcu_vllm_das.parallel import minimax31_deepep_runner as runner
from hcu_vllm_das.parallel import minimax31_msa as msa
from hcu_vllm_das.parallel import minimax31_pcp as pcp
from hcu_vllm_das.parallel import minimax31_static_runtime as static_runtime

_original_cache_guard = pcp.validate_fp8_sparse_cache
_original_index = msa.TeacherMSAPrefill.index
_original_attend = msa.TeacherMSAPrefill.attend
_original_sparse = pcp.MiniMax31PCPSparseAttention.forward
_mode = "baseline"
_hits = {"scale_cache": 0, "scale_read": 0, "local_pages": 0, "unit_q": 0}


@triton.jit(do_not_specialize=["Q", "PAGES", "STRIDE_H", "STRIDE_P"])
def _mark_local_pages(score, ends, Q, PAGES,
                      STRIDE_H, STRIDE_P, STRIDE_Q: tl.constexpr,
                      BLOCK: tl.constexpr):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = tl.program_id(1)
    age = tl.program_id(2)
    active = q < Q
    end = tl.load(ends + q, active, other=0)
    last = (end - 1) // 128
    page = last - age
    tl.store(score + head * STRIDE_H + page * STRIDE_P + q * STRIDE_Q, 1e30,
             active & (end > 0) & (page >= 0) & (page < PAGES))


def mark_local_pages(score, ends, local):
    if (score.dtype != torch.float32 or score.ndim != 3 or score.shape[0] != 4
            or not ends.is_contiguous() or ends.dtype != torch.int32 or ends.ndim != 1
            or score.size(2) != ends.numel() or score.device != ends.device or local <= 0):
        raise ValueError("Unsupported MSA local page metadata")
    _mark_local_pages[(triton.cdiv(ends.numel(), 128), 4, local)](
        score, ends, ends.numel(), score.size(1), *score.stride(), 128,
    )
    _hits["local_pages"] += 1


def scale_values(layer, cache, scales):
    if any(t.is_inference() for t in scales):
        _hits["scale_read"] += 1
        return tuple(float(t.item()) for t in scales)
    values = getattr(layer, "_prefill_scale_values", None)
    key = (cache.data_ptr(), tuple((t.data_ptr(), t._version) for t in scales))
    same_objects = values is not None and all(old is cur for old, cur in zip(values[1], scales))
    if same_objects and key == values[0]:
        _hits["scale_cache"] += 1
        return values[2]
    result = tuple(float(t.item()) for t in scales)
    layer._prefill_scale_values = key, tuple(scales), result
    _hits["scale_read"] += 1
    return result


def quant_query(q, scale):
    if msa._UNIT.get(q.device) is scale and scale._version == msa._UNIT_VERSION.get(q.device):
        _hits["unit_q"] += 1
        return q.to(torch.float8_e4m3fn)
    return (q.float() / scale).to(torch.float8_e4m3fn)


def _patch(fn, old, new, **bindings):
    source = textwrap.dedent(inspect.getsource(fn))
    if source.count(old) != 1:
        raise RuntimeError(f"Unrecognized source for {fn.__qualname__}")
    namespace = {}
    exec(compile(source.replace(old, new), fn.__code__.co_filename, "exec"),
         dict(fn.__globals__, **bindings), namespace)
    return namespace[fn.__name__]


def select_variant(model, mode):
    global _mode
    if mode not in ("baseline", "metadata", "fp8_gather", "status"):
        raise ValueError(f"Unknown Prefill variant {mode}")
    if mode == "status":
        return {"rank": torch.cuda.current_device(), "variant": _mode, "hits": dict(_hits)}
    if runner._buffer_inflight.locked() or static_runtime._FRAME_LOCK.locked():
        raise RuntimeError("Wait for active Prefill work to drain before switching variant")
    torch.cuda.synchronize()
    sparse = _original_sparse
    qualification = None
    if mode == "fp8_gather":
        from .prefill_fp8_transport import gather_fp8, validate_pack
        qualification = validate_pack()
        sparse = _replace_sparse(gather_fp8)
    if mode == "baseline":
        pcp.validate_fp8_sparse_cache = _original_cache_guard
        msa.TeacherMSAPrefill.index = _original_index
        msa.TeacherMSAPrefill.attend = _original_attend
    else:
        cache = _patch(_original_cache_guard,
                       "values = tuple(float(scale.item()) for scale in scales)",
                       "values = scale_values(layer, cache, scales)", scale_values=scale_values)
        from .prefill_stage2_runtime import score_and_select_pages
        index = _patch(_original_index,
                       '    score=self.native.msa128_stage1_page_max(index_q,index_cache.unsqueeze(1),meta.k_end,True,-1,\n        meta.q_lens_cpu,meta.kv_lens_cpu,None,meta.packed_kv_pages,meta.max_kv_len,index_q_scale,index_k_scale,True,False,False)\n    if local_blocks:\n        last=(meta.k_end-1)//128\n        pages=torch.arange(score.shape[1],device=score.device).view(1,-1,1)\n        mask=(pages<=last.view(1,1,-1))&(pages>=(last-local_blocks+1).clamp_min(0).view(1,1,-1))\n        score.masked_fill_(mask,1.e30) # same FP32 sentinel as reference 1.e30-offset\n    # Return native [Q,4,16]; the stock vLLM representation is permute(1,0,2).\n    return self.native.msa128_stage2_topk_select(score,16,(meta.max_kv_len+127)//128,\n                                                 init_blocks,0,None,None)\n',
                       '    return score_and_select_pages(\n        self.native, index_q, index_cache, meta, init_blocks, local_blocks,\n        index_q_scale, index_k_scale, mark_local_pages)\n',
                       score_and_select_pages=score_and_select_pages,
                       mark_local_pages=mark_local_pages)
        attend = _patch(_original_attend, "q8=(q.float()/q_scale).to(torch.float8_e4m3fn)",
                        "q8=quant_query(q, q_scale)", quant_query=quant_query)
        pcp.validate_fp8_sparse_cache = cache
        msa.TeacherMSAPrefill.index = index
        msa.TeacherMSAPrefill.attend = attend
    if sparse is not _original_sparse:
        sparse.__globals__["validate_fp8_sparse_cache"] = pcp.validate_fp8_sparse_cache
    pcp.MiniMax31PCPSparseAttention.forward = sparse
    _mode = mode
    return {"rank": torch.cuda.current_device(), "variant": mode, "hits": dict(_hits),
            "qualification": qualification}


def _replace_sparse(gather_fp8, code=None, **extra):
    old = """        kv_end = self.q_size + 2 * self.kv_size
        rows = qkv[:, self.q_size:kv_end]
        if not self.skip_index_topk:
            index_k_start = kv_end + self.index_q_size
            rows = torch.cat((rows, qkv[:, index_k_start:]), dim=-1)
        gathered = gather_cache_rows(rows, slots[self.layer_name])"""
    new = """        if (_EXPERIMENTAL_MSA and _EXPERIMENTAL_FP8_INDEX and
                self.kv_cache.numel() and not self.use_aiter_sparse_pa):
            gathered = gather_fp8(self, qkv, slots[self.layer_name], gather_cache_rows)
        else:
""" + "\n".join("    " + line for line in old.splitlines())
    clamp_guard = "                if _EXPERIMENTAL_FP8_INDEX:\n"
    # Apply both source substitutions before dedenting, preserving guards.
    if code is None:
        code = inspect.getsource(_original_sparse)
    if code.count(old) != 1 or code.count(clamp_guard) != 1:
        raise ValueError("Unrecognized sparse attention cache writer")
    code = code.replace(old, new).replace(
        clamp_guard, "                if _EXPERIMENTAL_FP8_INDEX and index_k.dtype == torch.bfloat16:\n"
    )
    namespace = {}
    exec(compile(textwrap.dedent(code), _original_sparse.__code__.co_filename, "exec"),
         dict(_original_sparse.__globals__, gather_fp8=gather_fp8, **extra), namespace)
    return namespace[_original_sparse.__name__]
