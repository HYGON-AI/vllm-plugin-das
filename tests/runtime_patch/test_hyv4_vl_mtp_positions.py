"""CPU regression checks for image-dependent PCP draft positions."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def draft_positions(owner, buffers, local, count):
    path = (Path(__file__).resolve().parents[2]
            / "vllm_hcu/v1/spec_decode/pcp_mtp.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef)
               and node.name == "HcuPCPMTPSpeculator")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_draft_positions")
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["_draft_positions"](owner, buffers, local, count)


def test_text_draft_keeps_scalar_positions():
    buffers = NS(positions=torch.arange(6))
    owner = NS(target_model_state=None)
    assert torch.equal(draft_positions(owner, buffers, None, 4), torch.arange(4))


def test_mrope_start_token_mask_does_not_add_an_embedding_axis():
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/models/hy_v4/mtp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "HYV4MultiTokenPredictor")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "forward")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    embeddings = torch.ones(4, 8)
    owner = NS(embed_input_ids=lambda ids: embeddings, mtp_start_layer_idx=78,
               layers={"78": lambda ids, pos, hidden, embeds: embeds})
    positions = torch.tensor([[0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 3, 2]])
    result = scope["forward"](owner, torch.arange(4), positions, embeddings)
    assert result.shape == (4, 8)
    assert torch.equal(result[0], torch.zeros(8))
    assert torch.equal(result[1:], embeddings[1:])


def test_vl_language_model_exposes_causal_lm_wrapper_for_native_mtp():
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/models/hy_v4/vl.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef)
               and node.name == "HYV4VLForConditionalGeneration")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "get_language_model")
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    owner = NS(model=NS(embed_tokens=object()))
    assert scope["get_language_model"](owner) is owner


def test_local_prefill_preserves_all_image_axes():
    positions = torch.tensor([[0, 1, 1], [0, 2, 3], [0, 4, 5]])
    state = NS(rope_state=object(), _local_positions=positions)
    owner = NS(target_model_state=state)
    result = draft_positions(owner, NS(positions=torch.arange(3)), object(), 3)
    assert torch.equal(result, positions)


def test_replicated_decode_uses_request_order_delta_and_keeps_padding():
    state = NS(rope_state=NS(prefill_delta=NS(gpu=torch.tensor([-20, 0, -7]))))
    owner = NS(target_model_state=state, _draft_idx_mapping=torch.tensor([2, 0]))
    buffers = NS(positions=torch.tensor([100, 200, 0]))
    result = draft_positions(owner, buffers, None, 3)
    assert torch.equal(result, torch.tensor([[93, 180, 0]]).expand(3, -1))
    assert torch.equal(buffers.positions, torch.tensor([100, 200, 0]))


def test_dummy_draft_before_request_binding_keeps_scalar_positions():
    owner = NS(target_model_state=NS(rope_state=object()), _draft_idx_mapping=None)
    buffers = NS(positions=torch.arange(3))
    assert torch.equal(draft_positions(owner, buffers, None, 3), buffers.positions)
