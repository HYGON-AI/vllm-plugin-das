"""Per-request and batch metrics for the colocated 3P8/D16 proxy benchmark."""

import json
from pathlib import Path


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    higher = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[higher] - ordered[lower]) * (index - lower)


def summarize_pd(rows, events, concurrency, *, proxy_finish_fallback=False):
    """Use one monotonic clock for the client and the proxy on the same host."""
    admission = {event['transfer_id']: event for event in events if event['event'] == 'request'}
    done = {event['transfer_id']: event['perf_counter'] for event in events if event['event'] == 'done'}
    assert rows and concurrency > 0
    origin = min(row['request_start_timestamp'] for row in rows)
    seen_ids = set()
    records = []
    for row in rows:
        request_id = row['request_id']
        assert request_id not in seen_ids, request_id
        seen_ids.add(request_id)
        transfer = row.get('transfer_id')
        request = admission.get(transfer)
        start = row['request_start_timestamp']
        admitted = request.get('proxy_admitted', request['perf_counter']) if request else None
        first = row.get('first_token_timestamp') or None
        last = row.get('last_token_timestamp') or None
        finish = row.get('request_finish_timestamp')
        if finish is None and proxy_finish_fallback and transfer in done and last is not None:
            finish = max(last, done[transfer])
        output_tokens = row['completion_tokens']
        success = bool(row['strict_stream_valid'])
        if success:
            assert None not in (admitted, first, last, finish), request_id
            assert start <= admitted <= first <= last <= finish and output_tokens > 0, request_id
        relative = lambda value: None if value is None else value - origin
        records.append({'request_id': request_id, 'concurrency': concurrency, 'mode': 'PD',
                        't_submit': relative(start), 't_admit': relative(admitted),
                        't_first_token': relative(first), 't_last_token': relative(last),
                        't_finish': relative(finish), 'input_tokens': row['input_tokens'],
                        'output_tokens': output_tokens, 'success': success})
    records.sort(key=lambda record: (record['t_submit'], record['request_id']))
    successful = [record for record in records if record['success']]
    ttfts = [1000 * (r['t_first_token'] - r['t_submit']) for r in successful]
    tpots = [1000 * (r['t_last_token'] - r['t_first_token']) / (r['output_tokens'] - 1)
             for r in successful if r['output_tokens'] > 1]
    input_tokens = sum(r['input_tokens'] for r in successful)
    output_tokens = sum(r['output_tokens'] for r in successful)
    finishes = [r['t_finish'] for r in records if r['t_finish'] is not None]
    duration = max(finishes, default=0.0)
    return records, {
        'mode': 'PD', 'concurrency': concurrency, 'request_count': len(records),
        'success_count': len(successful), 'tpot_sample_count': len(tpots), 'batch_seconds': duration,
        'input_tokens': input_tokens, 'output_tokens': output_tokens,
        'ttft_p50_ms': percentile(ttfts, .50), 'ttft_p90_ms': percentile(ttfts, .90),
        'tpot_p50_ms': percentile(tpots, .50), 'tpot_p90_ms': percentile(tpots, .90),
        'output_tps': output_tokens / duration if duration > 0 else None,
        'total_tps': (input_tokens + output_tokens) / duration if duration > 0 else None,
        'finish_clock': 'max(last_token,proxy_done)' if proxy_finish_fallback else 'client_stream_eof',
    }


def write_pd_metrics(folder: Path, rows, events, concurrency):
    records, summary = summarize_pd(rows, events, concurrency)
    with (folder / 'pd-requests.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(',', ':')) + '\n')
    (folder / 'pd-metrics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary
