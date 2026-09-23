"""GPU parity checks for DeepSeek-V4 ROCm FlashMLA sparse prefill."""

import torch
from flash_mla import flash_mla_sparse_fwd
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

DEVICE = "cuda:0"
HEADS = 64
D_QK = 512
D_V = 512
SCALE = D_QK**-0.5


def run_case(name: str, num_q: int, compressed_len: int, swa_len: int) -> None:
    torch.manual_seed(20260923 + num_q + compressed_len)
    total_kv = compressed_len + swa_len
    q = torch.randn(num_q, HEADS, D_QK, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(total_kv, 1, D_QK, device=DEVICE, dtype=torch.bfloat16)
    width = total_kv
    indices = torch.arange(width, device=DEVICE, dtype=torch.int32)
    indices = indices.view(1, 1, width).expand(num_q, 1, width).clone()
    lengths = torch.full((num_q,), width, device=DEVICE, dtype=torch.int32)
    sink = torch.randn(HEADS, device=DEVICE, dtype=torch.float32)
    reference = torch.empty(num_q, HEADS, D_V, device=DEVICE, dtype=torch.bfloat16)
    rocm_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=lengths,
        scale=SCALE,
        head_dim=D_QK,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=sink,
        output=reference,
    )
    actual, _, _ = flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=SCALE,
        d_v=D_V,
        attn_sink=sink,
        topk_length=lengths,
    )
    torch.cuda.synchronize()
    error = (actual.float() - reference.float()).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    print(name, num_q, compressed_len, swa_len, max_error, mean_error, flush=True)
    assert max_error <= 0.015625, (name, max_error, mean_error)


for tokens in (17, 257):
    run_case("swa_only", tokens, compressed_len=0, swa_len=128)
    run_case("c4a", tokens, compressed_len=64, swa_len=128)
    run_case("c128a", tokens, compressed_len=32, swa_len=128)
