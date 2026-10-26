"""Post-process client timestamps; do not estimate hidden per-token timings."""
from __future__ import annotations
import math
import statistics


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * q / 100
    low, high = math.floor(rank), math.ceil(rank)
    return values[low] + (values[high] - values[low]) * (rank - low)


def summarize_requests(records, base_metrics):
    ttfts, tpots, gaps, exported = [], [], [], []
    for record in records:
        chunks = [c for c in record['token_data'] if c['chunk_tokens'] > 0]
        if not chunks or not record['is_complete']:
            raise ValueError('Missing output timestamps or incomplete request')
        tokens = record['completion_tokens']
        if sum(c['chunk_tokens'] for c in chunks) != tokens:
            raise ValueError('Chunk token counts differ from completed token count')
        start = record['request_start_timestamp']
        first = chunks[0]['absolute_timestamp']
        last = chunks[-1]['absolute_timestamp']
        ttft = first - start
        if ttft <= 0 or last < first:
            raise ValueError('Invalid monotonic client timestamps')
        ttfts.append(ttft * 1000)
        tpot = (last - first) / (tokens - 1) * 1000 if tokens > 1 else None
        if tpot is not None:
            tpots.append(tpot)
        request_gaps = [(b['absolute_timestamp'] - a['absolute_timestamp']) * 1000
                        for a, b in zip(chunks, chunks[1:])]
        gaps.extend(request_gaps)
        exported.append(dict(
            request_id=record['request_id'], input_tokens=record['input_tokens'],
            output_tokens=tokens, ttft_ms=ttft * 1000, tpot_ms=tpot,
            request_start_s=start, first_token_s=first, last_token_s=last,
            response_end_s=start + record['total_latency'],
            chunks=[[c['absolute_timestamp'], c['chunk_tokens']] for c in chunks],
        ))
    if not exported:
        raise ValueError('No completed requests')
    duration = max(r['response_end_s'] for r in exported) - min(r['request_start_s'] for r in exported)
    output_tokens = sum(r['output_tokens'] for r in exported)
    itl = base_metrics['itl_avg'] * 1000 if base_metrics['decode_window_itl_count'] else None
    stats = dict(
        request_count=len(exported), ttft_mean_ms=statistics.mean(ttfts),
        ttft_p50_ms=percentile(ttfts, 50), ttft_p90_ms=percentile(ttfts, 90),
        ttft_p99_ms=percentile(ttfts, 99), ttft_max_ms=max(ttfts),
        tpot_mean_ms=statistics.mean(tpots) if tpots else None,
        tpot_sample_count=len(tpots), itl_mean_ms=itl,
        itl_common_window_sample_count=base_metrics['decode_window_itl_count'],
        itl_all_chunks_mean_ms=statistics.mean(gaps) if gaps else None,
        itl_all_chunks_sample_count=len(gaps),
        output_throughput_tok_s=output_tokens / duration,
        output_throughput_duration_s=duration, total_output_tokens=output_tokens,
        prefill_throughput_tok_s=base_metrics['prefill_tps'],
        decode_throughput_common_window_tok_s=base_metrics['decode_tps_e2e_window'],
    )
    return stats, exported
