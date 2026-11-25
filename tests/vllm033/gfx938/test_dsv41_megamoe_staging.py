"""Compare W4A8 IPC staging with Core's real token quantizer on HCU."""
from types import SimpleNamespace

import pytest
import torch

from hcu_vllm_das.moe.dsv41_w4a8_megamoe import stage_ct_w4a8
from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.hip, reason="HCU required")
def test_staging_writes_only_active_rows():
    rows, hidden, topk = 17, 5120, 6
    device = "cuda"
    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device=device)
    x[:3].zero_()
    x[1, 0] = 1e-12
    x[2, :5] = x.new_tensor([127, 1.5, -1.5, 2.5, -2.5])
    ids = torch.arange(topk, device=device, dtype=torch.int32).expand(rows, -1)
    weights = torch.rand((rows, topk * 2), device=device)[:, ::2]
    buf = SimpleNamespace(
        x=torch.full((rows + 1, hidden), -7, device=device, dtype=torch.int8),
        x_sf=torch.full((rows + 1,), -7., device=device),
        topk_idx=torch.full((rows + 1, topk), -7, device=device, dtype=torch.int64),
        topk_weights=torch.full((rows + 1, topk), -7., device=device),
    )
    q, sf = per_token_quant_int8(x)
    stage_ct_w4a8(buf, x, ids, weights)
    for actual, expected in ((buf.x, q), (buf.x_sf, sf.flatten()),
                             (buf.topk_idx, ids.long()), (buf.topk_weights, weights)):
        assert torch.equal(actual[:rows], expected)
        assert torch.all(actual[rows] == -7)
