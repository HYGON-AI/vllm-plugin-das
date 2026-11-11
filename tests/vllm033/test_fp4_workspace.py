"""CPU bounds for the FP4 reader, independent of model or device imports."""
import pytest

from hcu_vllm_das.fp4_workspace import query_chunk_rows, workspace_bytes


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
    assert budget(query_rows=1024) == budget(query_rows=32768)
    assert budget() < 1024**3


def test_only_noncanonical_decode_pages_need_staging():
    assert budget(physical_page_size=32) > budget(physical_page_size=64)
    assert budget(physical_page_size=128) == budget(physical_page_size=32)
    assert budget(decode_rows=0, physical_page_size=32) == budget(decode_rows=0)


def test_empty_and_decode_rows_are_bounded_by_actual_queries():
    assert budget(query_rows=0) == 0
    assert budget(query_rows=1, decode_rows=128) == budget(query_rows=1, decode_rows=1)
