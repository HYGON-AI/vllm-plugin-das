import json
import os
import pytest
import torch
from boltops.kimi_k31 import prepare_w4a8_experts, w8a8_linear
from boltops.fused_moe import fused_experts

MODEL = os.environ["KIMI_K31_CHECKPOINT"]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def checkpoint_tensors(keys):
    safetensors = pytest.importorskip("safetensors")
    with open(os.path.join(MODEL, "model.safetensors.index.json")) as f:
        mapping = json.load(f)["weight_map"]
    result = {}
    for shard in set(mapping[k] for k in keys):
        with safetensors.safe_open(os.path.join(MODEL, shard), framework="pt", device="cpu") as f:
            for k in keys:
                if mapping[k] == shard:
                    result[k] = f.get_tensor(k)
    return result


def unpack(w):
    b = w.view(torch.uint8).to(torch.int16)
    v = torch.stack((b >> 4, b & 15), -1).flatten(-2)
    return torch.where(v >= 8, v - 16, v).float()


def qdq(x):
    scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
    q = (x.float() / scale).round().clamp(-127, 127)
    return q * scale

@GPU
def test_real_checkpoint_w8a8_linear():
    prefix = "language_model.model.layers.3.self_attn.q_b_proj."
    t = checkpoint_tensors([prefix + "weight", prefix + "weight_scale"])
    w = t[prefix+"weight"].contiguous().cuda()
    s = t[prefix+"weight_scale"].contiguous().cuda()
    torch.manual_seed(19)
    x = torch.randn(3, w.shape[1], device="cuda", dtype=torch.bfloat16)
    actual = w8a8_linear(x, w, s)
    expected = (qdq(x) @ (w.float() * s).T).bfloat16()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=.03, atol=.08)

@GPU
def test_real_checkpoint_w4a8_situ_expert():
    prefix = "language_model.model.layers.1.block_sparse_moe.experts.0."
    keys = [prefix + w + suffix for w in ("w1", "w2", "w3") for suffix in (".weight", ".weight_scale")]
    t = checkpoint_tensors(keys)
    # Use full checkpoint expert dimensions and signed packed bytes.
    w1 = t[prefix+"w1.weight"][:192].contiguous().cuda().unsqueeze(0)
    w3 = t[prefix+"w3.weight"][:192].contiguous().cuda().unsqueeze(0)
    w2 = t[prefix+"w2.weight"][:,:96].contiguous().cuda().unsqueeze(0)
    s1 = t[prefix+"w1.weight_scale"][:192].contiguous().cuda().unsqueeze(0)
    s3 = t[prefix+"w3.weight_scale"][:192].contiguous().cuda().unsqueeze(0)
    s2 = t[prefix+"w2.weight_scale"].contiguous().cuda().unsqueeze(0)
    a,b,sa,sb = prepare_w4a8_experts(w1,w3,w2,s1,s3,s2)
    torch.manual_seed(9)
    x = torch.randn(2,3584,device="cuda",dtype=torch.bfloat16)
    ids = torch.zeros((2,1),device="cuda",dtype=torch.int32)
    weights = torch.ones((2,1),device="cuda",dtype=torch.float32)
    actual = fused_experts(x,a,b,weights,ids,activation="situ",use_int4_w4a8=True,per_channel_quant=True,w1_scale=sa,w2_scale=sb,activation_situ_beta=4.0,activation_situ_linear_beta=25.0)
    gate = (qdq(x) @ (unpack(w1[0]) * s1[0]).T).bfloat16().float()
    up = (qdq(x) @ (unpack(w3[0]) * s3[0]).T).bfloat16().float()
    gate = 4.0 * torch.tanh(gate / 4.0)
    hidden = (torch.nn.functional.silu(gate) * 25.0 * torch.tanh(up / 25.0)).bfloat16()
    expected = (qdq(hidden) @ (unpack(w2[0]) * s2[0]).T).bfloat16()
    torch.testing.assert_close(actual.float(),expected.float(),rtol=.08,atol=.05)
