"""Build native Prefill metadata from the scheduler's contiguous spans."""
import numpy as np
import torch

from hcu_vllm_das.parallel import minimax31_msa as msa

_original = msa._forward_meta


def _build(page_table, lengths, offsets):
    if (page_table.ndim != 2 or page_table.dtype != torch.int32
            or len(lengths) != len(offsets) or len(lengths) != page_table.shape[0]
            or not lengths or any(n <= 0 for n in lengths) or any(s < 0 for s in offsets)):
        raise ValueError("Unexpected native PCP query span or page table")
    end = tuple(s + n for s, n in zip(offsets, lengths))
    counts = tuple((n + 127) // 128 for n in end)
    if max(counts) > page_table.shape[1]:
        raise ValueError("Native PCP page table is shorter than its KV spans")
    dev = page_table.device
    table = page_table.contiguous()
    if min(counts) == max(counts):
        packed = table[:, :counts[0]].contiguous().view(-1)
    else:
        packed = torch.cat(tuple(table[i, :n] for i, n in enumerate(counts)))
    torch._assert_async((packed >= 0).all(), "active native page table contains invalid page")
    cu = np.cumsum((0, *lengths), dtype=np.int32)
    last = np.arange(1, int(cu[-1]) + 1, dtype=np.int32)
    last += np.repeat(np.asarray(offsets, dtype=np.int32) - cu[:-1], lengths)
    spans = tuple((i, int(cu[i]), s, n) for i, (s, n) in enumerate(zip(offsets, lengths)))
    return msa.ForwardMetadata(
        spans, torch.tensor(lengths, dtype=torch.int32), torch.tensor(end, dtype=torch.int32),
        torch.from_numpy(cu).to(dev), torch.tensor(offsets, dtype=torch.int32, device=dev),
        torch.tensor(end, dtype=torch.int32, device=dev), torch.from_numpy(last).to(dev),
        table, packed, max(end),
    )


def _compact(sub, req_start, req_count):
    if msa._HOST is None:
        raise RuntimeError("Native MSA has no PCP planner metadata")
    key = (id(sub), req_start, req_count)
    if key not in msa._META:
        lengths, offsets, _ = msa._HOST
        end = req_start + req_count
        if end > len(lengths) or sub.block_table.shape[0] != req_count:
            raise RuntimeError("Native MSA split metadata differs from planner rows")
        meta = _build(sub.block_table, lengths[req_start:end], offsets[req_start:end])
        torch._assert_async((meta.seq_lens == sub.seq_lens).all(), "native MSA planner seq_lens mismatch")
        if hasattr(sub, "cu_seqlens_q"):
            torch._assert_async(
                (meta.cu_q == sub.cu_seqlens_q).all(), "native MSA planner cu_seqlens mismatch",
            )
        msa._META[key] = meta
    return msa._META[key]


def configure(enabled):
    if type(enabled) is not bool:
        raise ValueError("Compact Prefill metadata takes a boolean flag")
    if enabled:
        for lengths, offsets in (((1,), (0,)), ((129, 6), (0, 257)), ((8, 128), (130, 10))):
            owners = [i for i, n in enumerate(lengths) for _ in range(n)]
            positions = [p for s, n in zip(offsets, lengths) for p in range(s, s + n)]
            pages = torch.arange(48, dtype=torch.int32, device="cuda").view(6, 8)
            pages = pages[::2][:len(lengths)]
            ref = msa.prepare_pcp_forward(
                pages, owners, positions, [s + n for s, n in zip(offsets, lengths)],
            )
            out = _build(pages, lengths, offsets)
            for name in out.__dataclass_fields__:
                left, right = getattr(ref, name), getattr(out, name)
                equal = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
                if not equal:
                    raise RuntimeError(f"Compact native metadata changed {name}")
    msa._forward_meta = _compact if enabled else _original
    return {"enabled": enabled, "qualification_cases": 3 if enabled else 0}
