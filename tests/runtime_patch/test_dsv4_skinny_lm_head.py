# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("skinny lm_head tests need a working GPU", allow_module_level=True)

pytestmark = pytest.mark.hcu

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402

from vllm.model_executor.layers import logits_processor as lp  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: E402
    UnquantizedEmbeddingMethod,
)

from vllm_hcu.model_executor.layers import skinny_lm_head  # noqa: E402
from vllm_hcu.patch.worker.framework_opt import (  # noqa: E402
    patch_step5_skinny_lm_head,
)


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


class _Head(torch.nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(vocab, dim, dtype=torch.bfloat16, device="cuda") * 0.02,
            requires_grad=False,
        )
        self.quant_method = UnquantizedEmbeddingMethod()
        self.tp_size = 1


class _Draft(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.logits_processor = lp.LogitsProcessor(vocab)
        self.inner = torch.nn.Module()
        self.inner.logits_processor = lp.LogitsProcessor(vocab)


def test_enable_for_model_flags_every_logits_processor():
    model = _Draft(1024)
    assert skinny_lm_head.enable_for_model(model) == 2
    assert model.logits_processor._step5_skinny_lm_head
    assert model.inner.logits_processor._step5_skinny_lm_head
    assert skinny_lm_head.enable_for_model(None) == 0


@pytest.mark.parametrize(("rows", "dim"), [(8, 256), (48, 4096)])
def test_dsv4_head_shapes_use_skinny_kernel(monkeypatch, rows, dim):
    patch_step5_skinny_lm_head.apply()
    calls = []
    real = skinny_lm_head.skinny_linear

    def spy(x, w):
        calls.append(tuple(x.shape))
        return real(x, w)

    monkeypatch.setattr(skinny_lm_head, "skinny_linear", spy)
    vocab = 129280
    head = _Head(vocab, dim)
    processor = lp.LogitsProcessor(vocab)
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device="cuda")
    ref = (x.float() @ head.weight.float().T)
    plain = processor(head, x)
    assert calls == []
    skinny_lm_head.enable_for_model(processor)
    out = processor(head, x)
    assert calls == [(rows, dim)]
    assert out.shape == (rows, vocab)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out.float(), plain.float(), rtol=2e-2, atol=2e-2)
