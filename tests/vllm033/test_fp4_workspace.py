"""CPU bounds for the FP4 reader, independent of model or device imports."""
import pytest

from hcu_vllm_das.fp4_workspace import (
    prefill_chunk_rows,
    query_chunk_rows,
    workspace_bytes,
)


def test_query_rows_respect_fixed_aligned_score_width():
    assert query_chunk_rows(65536, 256 * 1024**2) == 1024
    assert query_chunk_rows(65, 4 * 68 * 3) == 3
    assert query_chunk_rows(65536, 1) == 1


@pytest.mark.parametrize('width,budget', [(0, 10), (10, 0), (-1, 10)])
def test_invalid_bound_refused(width, budget):
    with pytest.raises(ValueError):
        query_chunk_rows(width, budget)


def budget(**overrides):
    args = dict(query_rows=32768, decode_rows=128, max_model_len=65536,
                total_seq_lens=65536, topk_tokens=2048, num_query_heads=32,
                max_logits_bytes=256 * 1024**2, physical_page_size=64)
    return workspace_bytes(**dict(args, **overrides))


def test_prefill_budget_saturates_after_query_chunk_cap():
    assert budget(query_rows=2048) == budget(query_rows=32768)
    assert budget() < 1100 * 1024**2


def test_long_model_keeps_32k_prefill_row_capacity():
    logits_budget = 512 * 1024**2
    assert query_chunk_rows(524288, logits_budget) == 256
    assert prefill_chunk_rows(524288, 16384, logits_budget) == 4096
    assert prefill_chunk_rows(524288, 524288, logits_budget) == 256


@pytest.mark.parametrize('width', [4, 16384, 32768, 65536, 524288])
def test_profile_covers_live_width_and_rows(width):
    max_logits_bytes = 512 * 1024**2
    rows = min(8192, prefill_chunk_rows(524288, width, max_logits_bytes))
    live = width * 76 + rows * (10 * width + 512 * 96 + 32 * 128 * 2) + width * 8
    assert budget(query_rows=8192, decode_rows=0, max_model_len=524288,
                  total_seq_lens=524288, topk_tokens=512,
                  max_logits_bytes=max_logits_bytes) >= live


def test_only_noncanonical_decode_pages_need_staging():
    assert budget(physical_page_size=32) > budget(physical_page_size=64)
    assert budget(physical_page_size=128) == budget(physical_page_size=32)
    assert budget(decode_rows=0, physical_page_size=32) == budget(decode_rows=0)


def test_empty_and_decode_rows_are_bounded_by_actual_queries():
    assert budget(query_rows=0) == 0
    assert budget(query_rows=1, decode_rows=128) == budget(query_rows=1, decode_rows=1)


def test_persistent_decode_pool_is_excluded_from_transient_budget():
    args = dict(query_rows=128, decode_rows=128, max_model_len=65536,
                total_seq_lens=0, topk_tokens=2048, num_query_heads=32,
                max_logits_bytes=256 * 1024**2, physical_page_size=32)
    transient = workspace_bytes(**args, persistent_staging=True)
    combined = workspace_bytes(**args)
    pool_bytes = 128 * 65536 * 68
    assert combined - transient == pool_bytes
