# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only preflight for the aux-docked DSpark markov tail.

Run after any dspark / role-separation edit; needs no GPU, no distributed
init and no checkpoint weights, so boot-blocking regressions surface in
seconds instead of after a full weight load:

    python3 tests/patch/test_dspark_cpu_preflight.py

Covers three boot-failure classes:

* ``20261009 19:59`` — ``RuntimeError: LMHead's weights should be used in the
  sampler.`` (``ParallelLMHead.forward`` guard hit by ``markov_bias``).
* ``20261009 19:41`` — RCCL ``runTreeUpDown/ProtoSimple`` 16-rank wedge (markov
  collectives landing on the engine c10d comm that also carries the
  captured decode all_gather).
* ``20261010 08:41`` — eager_all_gather fed a dim-scaled buffer to
  ncclAllGather (flat dim-0 concat only): acceptance ~0 with healthy
  throughput/accuracy. Guarded in comment-stripped source (value-level
  layout semantics are invisible to structural checks and ws=1 numerics).

Import order matters: ``vllm_hcu`` must be imported before ``vllm`` so the
hcu platform plugin activates (it patches wheel modules the model tree
depends on, e.g. the sparse attention indexer).
"""

from __future__ import annotations

import ast
import inspect
import io
import tokenize
from pathlib import Path
from types import SimpleNamespace

import vllm_hcu  # noqa: F401  -- activates the hcu platform plugin

import pytest
import torch

_DSPARK = Path(__file__).resolve().parents[2] / (
    "vllm_hcu/models/kimi_k3/dspark.py"
)
_TAIL_PATCH = Path(__file__).resolve().parents[2] / (
    "vllm_hcu/patch/worker/framework_opt/patch_dspark_role_separation.py"
)
_ROLE_SEP = Path(__file__).resolve().parents[2] / (
    "vllm_hcu/models/kimi_k3/role_separation.py"
)


def _sources() -> tuple[str, str]:
    return (
        _DSPARK.read_text(encoding="utf-8"),
        _TAIL_PATCH.read_text(encoding="utf-8"),
    )


def _fn(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def test_wheel_guard_still_exists_and_bites() -> None:
    """Sensitivity anchor: the direct-forward guard that killed 19:59 is
    the thing this preflight defends against."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
    )

    src = inspect.getsource(ParallelLMHead.forward)
    assert "should be used in the sampler" in src


def test_markov_bias_never_calls_forward_or_processor() -> None:
    """The 19:59 crash is ``markov_w2(...)`` / LogitsProcessor calls inside
    ``markov_bias``; the only allowed seams are the guard-free shard
    projection and the auxiliary gather."""
    src, _ = _sources()
    calls = [
        ast.unparse(c)
        for c in ast.walk(_fn(ast.parse(src), "markov_bias"))
        if isinstance(c, ast.Call)
    ]
    joined = " | ".join(calls)
    assert "markov_w2(" not in joined, joined
    assert "logits_processor(" not in joined, joined
    assert ".bias(" not in joined, joined
    assert any("apply_lm_head_shard(" in c for c in calls), calls
    assert any("eager_all_gather(" in c for c in calls), calls


def test_markov_bias_executes_on_cpu() -> None:
    """Dynamic execution with a fake sharded head: the aux-docked bias path
    must run clean (world size 1 => gather is identity) and be bit-equal to
    the full-GEMM fork reference.

    Runs in a fresh interpreter on purpose: importing the real model module
    depends on the hcu platform plugin having patched wheel modules, and
    under pytest the collection-time platform resolution races that patch
    (bare ``python3`` does not).  The subprocess replicates the verified
    bare-interpreter flow exactly.
    """
    import os
    import subprocess
    import sys

    code = r'''
import inspect
from types import SimpleNamespace
import torch
import vllm.distributed as _dist
import vllm_hcu.models.kimi_k3.role_separation as role_sep
from vllm_hcu.models.kimi_k3.dspark import K3DSparkForCausalLM

# World-size-1 bare-process harness: a fake auxiliary comm (gather == pass-
# through) plus a patched TP world size; the real path only ever calls
# comm.all_gather(out, t) on the module-level _aux_ar_comm.
role_sep._aux_ar_comm = SimpleNamespace(
    all_gather=lambda out, t: out.copy_(t)
)
_dist.get_tensor_model_parallel_world_size = lambda: 1

vocab, rank = 32, 8
torch.manual_seed(11)
weight = torch.nn.Parameter(torch.randn(vocab, rank, dtype=torch.float32))
head = SimpleNamespace(
    weight=weight,
    quant_method=SimpleNamespace(
        apply=lambda layer, x, bias=None: x @ layer.weight.T
    ),
)
model = object.__new__(K3DSparkForCausalLM)
model.config = SimpleNamespace(vocab_size=vocab, draft_vocab_size=vocab)
model.model = SimpleNamespace(markov_head=SimpleNamespace(markov_w2=head))
model.logits_processor = SimpleNamespace()  # no _apply_head seam

x = torch.randn(5, rank, dtype=torch.float32)
bias = model.markov_bias(x)
assert bias.shape == (5, vocab)
assert torch.equal(bias, x @ weight.T)
print("DYNAMIC PASS")
'''
    repo = Path(__file__).resolve().parents[2]
    # vllm's pytest plugin pollutes the inherited environment (device
    # visibility vars et al.) in a way that keeps the hcu platform plugin
    # from activating in child processes; strip those prefixes so the child
    # sees the same environment a bare-shell import would.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("CUDA_", "HIP_", "ROCR_", "VLLM_", "PYTEST_")
        )
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    assert "DYNAMIC PASS" in proc.stdout


def test_shard_concat_numerics_match_full_gemm() -> None:
    """Shard GEMM + concat equals the fork's unsharded GEMM bit-for-bit."""
    torch.manual_seed(5)
    vocab, rank, world = 32, 8, 4
    w2 = torch.randn(vocab, rank, dtype=torch.float32)
    x = torch.randn(6, rank, dtype=torch.float32)
    reference = x @ w2.T
    shards = [
        x @ w2[s * vocab // world : (s + 1) * vocab // world].T
        for s in range(world)
    ]
    assert torch.equal(torch.cat(shards, dim=-1)[..., :vocab], reference)


def _strip_comments(text: str) -> str:
    """Comment-free source so textual order checks see only real code."""
    import io
    import tokenize

    tokens = [
        tok
        for tok in tokenize.generate_tokens(io.StringIO(text).readline)
        if tok.type != tokenize.COMMENT
    ]
    return tokenize.untokenize(tokens)


def test_tail_call_paths_are_aux_scoped() -> None:
    """Both eager call sites of the markov head must be scoped to the
    auxiliary comm, and the captured (shard) branch of generate_draft must
    stay free of markov/all_gather code."""
    _, raw = _sources()
    src = _strip_comments(raw)

    # (a) hcu_generate_draft: the CAPTURED true-branch (between the guard
    # condition and its else) must be only the collective-free shard write;
    # the eager else must trip the anti-capture guard first, then run the
    # wheel sampler inside the aux scope.
    i_gen = src.index("def hcu_generate_draft")
    i_if = src.index("if getattr(self, _SHARD_STATE, False)", i_gen)
    i_else = src.index("else:", i_if)
    i_guard = src.index("_guard_captured_draft_all_gather", i_else)
    i_with = src.index("with eager_ar_scope():", i_else)
    i_call = src.index("self._sample_sequential(", i_gen)
    assert i_if < i_else < i_guard < i_with < i_call, (
        i_gen,
        i_if,
        i_else,
        i_guard,
        i_with,
        i_call,
    )
    captured = src[i_if:i_else]
    assert "markov" not in captured, captured
    assert "all_gather" not in captured, captured
    assert "all_reduce" not in captured, captured
    assert "_sample_sequential" not in captured, captured

    # (b) _gather_and_markov: scope precedes both markov calls.
    i_gam = src.index("def _gather_and_markov")
    i_with2 = src.index("with eager_ar_scope():", i_gam)
    i_emb = src.index("speculator.model.markov_embed", i_gam)
    i_bias = src.index("speculator.model.markov_bias", i_gam)
    assert i_gam < i_with2 < i_emb < i_bias, (
        i_gam,
        i_with2,
        i_emb,
        i_bias,
    )


def test_scope_names_are_bound_in_tail_functions() -> None:
    """Regression (20261009 20:2x): ``_gather_and_markov`` used
    ``with eager_ar_scope()`` without importing it — a NameError that only
    detonates at the first replay exit.  py_compile cannot see it; this
    guards the import binding of every name used by the aux-routing edits.
    """
    _, patch_src = _sources()
    tree = ast.parse(patch_src)

    module_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_imports.update(alias.name for alias in node.names)

    for fn_name in ("_gather_and_markov", "hcu_generate_draft"):
        fn = _fn(tree, fn_name)
        local_imports: set[str] = set()
        used: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom):
                local_imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Name) and isinstance(
                node.ctx, ast.Load
            ):
                used.add(node.id)
        # Names referenced by the aux-routing edits must resolve somewhere.
        for required in ("eager_ar_scope", "eager_all_gather"):
            if required in used:
                assert (
                    required in local_imports or required in module_imports
                ), f"{fn_name}: {required} used but never imported"


def test_eager_all_gather_layout_is_the_official_port() -> None:
    """Regression (20261010 08:41 boot): eager_all_gather allocated a
    dim-scaled buffer and fed it to ncclAllGather directly, but ncclAllGather
    has no dim semantics -- it concats flat chunks along dim 0 only.  The
    [B, V/ws] shards reassembled with wrong row blocks: per-position
    acceptance ~0 while drafted throughput and GSM8K accuracy stayed healthy.
    Structural checks and the ws=1 numerics are blind to this class; guard
    the official layout port (DeviceCommunicatorBase.all_gather: flat gather
    buffer + reshape/movedim/reshape) in comment-stripped source.
    """
    tokens = [
        t.string
        for t in tokenize.generate_tokens(
            io.StringIO(_ROLE_SEP.read_text(encoding="utf-8")).readline
        )
        if t.type != tokenize.COMMENT
    ]
    text = " ".join(tokens)
    assert "movedim" in text, "official layout port (movedim) missing"
    assert "reshape" in text
    assert "*= world_size" not in text, "naive dim-scaled buffer regression"


def _main() -> int:
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    _main()
