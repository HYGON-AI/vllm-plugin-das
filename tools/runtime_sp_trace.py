# SPDX-License-Identifier: Apache-2.0
"""Opt-in eager intermediate capture and offline SP/TP layout comparison.

This diagnostic synchronizes CPU snapshots. Never enable capture during Graph
capture/replay or serving measurements. It adds no production provider policy.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import re

import torch


def _layout(name, arm):
    if name == "model.embed_tokens":
        return "replicated"
    if re.fullmatch(r"model\.layers\.\d+\.self_attn\.(qkv_proj|q_norm|k_norm|rotary_emb|attn)", name):
        return "head_sharded"
    if name == "model.norm" or re.fullmatch(
        r"model\.layers\.\d+(\.(input_layernorm|post_attention_layernorm|mlp|self_attn|self_attn\.o_proj))?", name
    ):
        return "token_sharded" if arm == "candidate" else "replicated"
    return None


def _leaves(value, path="output"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}.{index}")
    elif value is not None:
        raise TypeError(f"unsupported trace output at {path}: {type(value).__name__}")


class RuntimeSPTrace:
    def __init__(self, model, *, arm, rank, size, num_tokens, max_bytes=8 * 1024 * 1024):
        if arm not in ("reference", "candidate") or not 0 <= rank < size or num_tokens <= 0:
            raise ValueError("invalid trace geometry or arm")
        self.data = dict(schema=1, arm=arm, rank=rank, size=size,
                         num_tokens=num_tokens, samples={})
        self._sample = None
        self._bytes = 0
        self._max_bytes = max_bytes
        self._handles = []
        for name, module in model.named_modules():
            layout = _layout(name, arm)
            if layout is not None:
                self._handles.append(module.register_forward_hook(self._hook(name, layout)))

    def _hook(self, name, layout):
        def record(module, args, output):
            if self._sample is None:
                return
            if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("CPU intermediate capture is forbidden inside a GPU Graph")
            leaves = {}
            for key, tensor in _leaves(output):
                self._bytes += tensor.numel() * tensor.element_size()
                if self._bytes > self._max_bytes:
                    raise RuntimeError("runtime SP trace exceeds its CPU snapshot budget")
                leaves[key] = tensor.detach().to(device="cpu", copy=True)
            if not leaves:
                raise ValueError("trace event has no tensor outputs")
            self._sample["events"].append(dict(name=name, layout=layout, leaves=leaves))
        return record

    @contextmanager
    def capture(self, label, *, active_tokens):
        if self._sample is not None or label in self.data["samples"]:
            raise ValueError("nested or duplicate trace sample")
        if not 0 < active_tokens <= self.data["num_tokens"]:
            raise ValueError("invalid active token count")
        sample = dict(active_tokens=active_tokens, complete=False, events=[])
        self.data["samples"][label] = sample
        self._sample = sample
        try:
            yield
        except BaseException:
            raise
        else:
            sample["complete"] = True
        finally:
            self._sample = None

    def save(self, path):
        torch.save(self.data, path)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _ordered(records, arm):
    if not records:
        raise ValueError("missing trace ranks")
    size, tokens = records[0]["size"], records[0]["num_tokens"]
    if len(records) != size or {r["rank"] for r in records} != set(range(size)):
        raise ValueError("missing or duplicate trace rank")
    records = sorted(records, key=lambda r: r["rank"])
    for record in records:
        if (record["schema"] != 1 or record["arm"] != arm or record["size"] != size
                or record["num_tokens"] != tokens):
            raise ValueError("inconsistent trace metadata")
        if not record["samples"] or any(not s["complete"] for s in record["samples"].values()):
            raise ValueError("empty or incomplete trace sample")
        if any(not 0 < s["active_tokens"] <= tokens for s in record["samples"].values()):
            raise ValueError("invalid active token count")
    if any(tuple(r["samples"]) != tuple(records[0]["samples"]) for r in records):
        raise ValueError("trace sample order differs across ranks")
    return records


def _canonical(records, label, event_index, leaf, rank):
    event = records[rank]["samples"][label]["events"][event_index]
    tensor = event["leaves"][leaf]
    tokens, size = records[rank]["num_tokens"], len(records)
    if tensor.device.type != "cpu" or tensor.ndim == 0:
        raise ValueError("expected CPU trace with token axis")
    if event["layout"] == "token_sharded":
        pieces = [r["samples"][label]["events"][event_index]["leaves"][leaf] for r in records]
        if any(p.shape != tensor.shape or p.dtype != tensor.dtype or p.device.type != "cpu"
               for p in pieces):
            raise ValueError("inconsistent token shard shapes or dtypes")
        if tensor.shape[0] != (tokens + size - 1) // size:
            raise ValueError("invalid token shard extent")
        tensor = torch.cat(pieces, dim=0)
    elif event["layout"] not in ("replicated", "head_sharded"):
        raise ValueError("unknown trace layout")
    if tensor.shape[0] not in (tokens, ((tokens + size - 1) // size) * size):
        raise ValueError("trace has an invalid global token extent")
    return tensor[:records[rank]["samples"][label]["active_tokens"]]


def compare_trace_sets(reference, candidate, *, atol=.016, rtol=.016):
    """Report first observed differences; neither layout parity nor root cause proof."""
    reference, candidate = _ordered(reference, "reference"), _ordered(candidate, "candidate")
    if (len(reference) != len(candidate)
            or reference[0]["num_tokens"] != candidate[0]["num_tokens"]
            or tuple(reference[0]["samples"]) != tuple(candidate[0]["samples"])):
        raise ValueError("reference/candidate metadata differ")
    rows = []
    for label in reference[0]["samples"]:
        exemplar = reference[0]["samples"][label]
        names = [e["name"] for e in exemplar["events"]]
        if not names:
            raise ValueError("trace contains no module events")
        for record in reference + candidate:
            sample = record["samples"][label]
            if (sample["active_tokens"] != exemplar["active_tokens"]
                    or [e["name"] for e in sample["events"]] != names):
                raise ValueError("trace execution order or active tokens differ")
        for records in (reference, candidate):
            for index in range(len(names)):
                first = records[0]["samples"][label]["events"][index]
                for record in records[1:]:
                    event = record["samples"][label]["events"][index]
                    if (event["layout"] != first["layout"]
                            or event["leaves"].keys() != first["leaves"].keys()):
                        raise ValueError("event layout or leaves differ across ranks")
        for index, name in enumerate(names):
            for rank in range(len(reference)):
                left, right = (records[rank]["samples"][label]["events"][index]
                               for records in (reference, candidate))
                if (left["layout"], right["layout"]) not in (
                    ("replicated", "replicated"), ("replicated", "token_sharded"),
                    ("head_sharded", "head_sharded"),
                ) or not left["leaves"] or left["leaves"].keys() != right["leaves"].keys():
                    raise ValueError("incompatible event layouts or output leaves")
                for leaf in left["leaves"]:
                    expected = _canonical(reference, label, index, leaf, rank)
                    actual = _canonical(candidate, label, index, leaf, rank)
                    if actual.shape != expected.shape or actual.dtype != expected.dtype:
                        raise ValueError("trace tensor shape or dtype differs")
                    try:
                        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                        within_tolerance = True
                    except AssertionError:
                        within_tolerance = False
                    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                    difference = (actual.float() - expected.float()).abs().max()
                    rows.append(dict(sample=label, event_index=index, event=name,
                        leaf=leaf, rank=rank,
                        exact=torch.equal(actual, expected), within_tolerance=within_tolerance and finite,
                        max_abs_error=float(difference) if torch.isfinite(difference) else None,
                        finite=finite))
    return dict(scope="eager intermediates; first difference is not root cause proof",
                atol=atol, rtol=rtol, comparisons=rows,
                first_difference=next((r for r in rows if not r["exact"]), None),
                first_outside_tolerance=next((r for r in rows if not r["within_tolerance"]), None))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("family", choices=("glm4", "hy3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = [[torch.load(p, map_location="cpu", weights_only=True)
             for p in sorted(args.directory.glob(f"trace-{args.family}-{arm}-rank*.pt"))]
            for arm in ("reference", "candidate")]
    result = compare_trace_sets(*arms)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
