"""Measure fixed-length Step5 SLOs on an already qualified deployment."""
import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

from pd_request_metrics import percentile


INPUT_TOKENS = 128 * 1024
OUTPUT_TOKENS = 4096
REQUESTS_PER_SLOT = 10
TTFT_LIMIT_MS = 15000
TPOT_LIMIT_MS = 30
CACHE_METRICS = ('kv_cache', 'kv_transfer', 'prefix_cache', 'preemption',
                 'num_requests_running', 'num_requests_waiting', 'request_prompt_tokens')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def make_prompt(seed):
    rng = random.Random(seed)
    return [rng.randrange(256, 65536) for _ in range(INPUT_TOKENS)]


async def request(client, args, request_index, seed):
    prompt = await asyncio.to_thread(make_prompt, seed)
    digest = hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest()
    submitted = time.monotonic()
    first = last = None
    usage = None
    ended = False
    seen_tokens = 0
    failure = None
    try:
        async with client.stream('POST', args.endpoint + '/v1/completions', json={
            'model': args.model, 'prompt': prompt, 'max_tokens': OUTPUT_TOKENS,
            'stream': True, 'ignore_eos': True, 'temperature': 0.0,
            'return_token_ids': True, 'stream_options': {'include_usage': True},
        }) as reply:
            reply.raise_for_status()
            async for line in reply.aiter_lines():
                if not line.startswith('data: '):
                    continue
                if ended:
                    raise ValueError('stream sent data after the terminal event')
                if line == 'data: [DONE]':
                    ended = True
                    continue
                now = time.monotonic()
                chunk = json.loads(line[6:])
                if chunk.get('error'):
                    raise RuntimeError(str(chunk['error']))
                if chunk.get('usage') is not None:
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    ids = choice.get('token_ids') or []
                    if ids:
                        if first is None:
                            first = now
                        last = now
                        seen_tokens += len(ids)
        if not ended or usage is None or seen_tokens != OUTPUT_TOKENS:
            raise ValueError(f'incomplete stream: ended={ended}, usage={usage}, token_ids={seen_tokens}')
        if (usage.get('prompt_tokens'), usage.get('completion_tokens')) != (INPUT_TOKENS, OUTPUT_TOKENS):
            raise ValueError(f'unexpected token usage: {usage}')
    except Exception as exc:
        failure = f'{type(exc).__name__}: {str(exc)[:600]}'
    return {'request_id': f'req_{request_index}', 'seed': seed, 'prompt_sha256': digest,
            't_submit': submitted, 't_first_token': first, 't_last_token': last,
            't_finish': time.monotonic(), 'input_tokens': usage.get('prompt_tokens') if usage else None,
            'output_tokens': seen_tokens, 'success': failure is None, 'error': failure}


def summarize(records, concurrency, slo_profile='p90'):
    good = [row for row in records if row['success']]
    ttft = [1000*(r['t_first_token']-r['t_submit']) for r in good]
    tpot = [1000*(r['t_last_token']-r['t_first_token'])/(r['output_tokens']-1) for r in good]
    wall = max(r['t_finish'] for r in records)-min(r['t_submit'] for r in records)
    ttft90, tpot90 = percentile(ttft, .90), percentile(tpot, .90)
    ttft50, tpot50 = percentile(ttft, .50), percentile(tpot, .50)
    sla = ((ttft50, 10000), (tpot50, 60)) if slo_profile == 'p50' else (
        (ttft90, TTFT_LIMIT_MS), (tpot90, TPOT_LIMIT_MS))
    input_tokens = sum(r['input_tokens'] for r in good)
    output_tokens = sum(r['output_tokens'] for r in good)
    return {'concurrency': concurrency, 'requests': len(records), 'successes': len(good),
            'success_rate': len(good)/len(records), 'ttft_p50_ms': ttft50,
            'ttft_p90_ms': ttft90, 'tpot_p50_ms': tpot50, 'tpot_p90_ms': tpot90,
            'input_tokens': input_tokens, 'output_tokens': output_tokens, 'wall_seconds': wall,
            'input_tps': input_tokens/wall, 'output_tps': output_tokens/wall,
            'total_tps': (input_tokens+output_tokens)/wall,
            'slo_met': len(good)==len(records) and all(value is not None and value <= limit for value, limit in sla)}


async def scrape(client, metrics, origin, phase):
    async def one(label, address):
        try:
            response = await client.get(address + '/metrics', timeout=8)
            response.raise_for_status()
            return label, [line for line in response.text.splitlines()
                           if any(key in line.split('{', 1)[0] for key in CACHE_METRICS)]
        except Exception as exc:
            return label, {'error': f'{type(exc).__name__}: {str(exc)[:400]}'}

    return {'time_s': time.monotonic()-origin, 'phase': phase,
            'stages': dict(await asyncio.gather(*(one(k, v) for k, v in metrics.items())))}


def summarize_cache(path):
    names = {'vllm:kv_cache_usage_perc': 'usage', 'vllm:prefix_cache_queries_total': 'queries',
             'vllm:prefix_cache_hits_total': 'hits', 'vllm:num_preemptions_total': 'preemptions'}
    pattern = re.compile(r'^([^\s{]+)(?:\{.*\})?\s+([^\s]+)')
    stages = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            snapshot = json.loads(line)
            phase = snapshot['phase']
            for label, sample in snapshot['stages'].items():
                stage = stages.setdefault(label, {'peak_kv_usage_ratio': None, 'scrape_errors': 0,
                                                  'before': {}, 'after': {}})
                if isinstance(sample, dict):
                    stage['scrape_errors'] += 1
                    continue
                values = {}
                for metric in sample:
                    match = pattern.match(metric)
                    if match is None or match[1] not in names:
                        continue
                    value = float(match[2])
                    if math.isfinite(value):
                        values.setdefault(names[match[1]], []).append(value)
                if values.get('usage'):
                    stage['peak_kv_usage_ratio'] = max(stage['peak_kv_usage_ratio'] or 0, *values['usage'])
                if phase in ('before', 'after'):
                    stage[phase] = {key:sum(rows) for key, rows in values.items() if key != 'usage'}
    for stage in stages.values():
        before, after = stage.pop('before'), stage.pop('after')
        deltas = {key:after[key]-before[key] for key in before.keys() & after.keys() if after[key]>=before[key]}
        stage['prefix_cache_query_tokens_delta'] = deltas.get('queries')
        stage['prefix_cache_hit_tokens_delta'] = deltas.get('hits')
        stage['preemptions_delta'] = deltas.get('preemptions')
    return stages


async def run_round(client, args, metrics, concurrency):
    folder = args.output_dir / f'c{concurrency:04d}'
    folder.mkdir(exist_ok=False)
    total = concurrency * (1 if args.quick_pd and not args.quick_pd_sweep else REQUESTS_PER_SLOT)
    origin = time.monotonic()
    record_rows = []
    jobs = iter(range(total))
    with ((folder / 'requests.jsonl').open('w', encoding='utf-8') as log,
          (folder / 'kv-cache-metrics.jsonl').open('w', encoding='utf-8') as trace):
        async def snapshot(phase):
            row = await scrape(client, metrics, origin, phase)
            trace.write(json.dumps(row, separators=(',', ':'))+'\n')
            trace.flush()

        async def monitor():
            while True:
                await snapshot('during')
                await asyncio.sleep(args.scrape_interval)

        async def worker():
            for index in jobs:
                row = await request(client, args, index, args.seed+index)
                record_rows.append(row)
                logged = dict(row, concurrency=concurrency)
                for key in ('t_submit', 't_first_token', 't_last_token', 't_finish'):
                    if logged[key] is not None:
                        logged[key] -= origin
                log.write(json.dumps(logged, separators=(',', ':'))+'\n')
                log.flush()

        await snapshot('before')
        recording = asyncio.create_task(monitor())
        try:
            await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:
            recording.cancel()
            await asyncio.gather(recording, return_exceptions=True)
            await snapshot('after')
    assert len(record_rows) == total
    summary = summarize(record_rows, concurrency, 'p50' if args.quick_pd else 'p90')
    summary['kv_cache'] = summarize_cache(folder/'kv-cache-metrics.jsonl')
    write_json(folder/'summary.json', summary)
    print('FIXED_128K_ROUND', json.dumps(summary), flush=True)
    return summary


async def main(args):
    import httpx

    metrics = dict(item.split('=', 1) for item in args.metrics)
    deployment = json.loads(args.deployment_manifest.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status':'running', 'scenario':'fixed_128k', 'load':'closed_loop',
        'input_tokens_per_request':INPUT_TOKENS, 'output_tokens_per_request':OUTPUT_TOKENS,
        'requests_per_concurrency':1 if args.quick_pd and not args.quick_pd_sweep else REQUESTS_PER_SLOT, 'seed':args.seed,
        'success_rate_min':1.0,
        'requested_max_concurrency':args.max_concurrency, 'scrape_interval_s':args.scrape_interval,
        'deployment':deployment,
    }
    if args.quick_pd:
        manifest.update(ttft_p50_limit_ms=10000, tpot_p50_limit_ms=60)
    else:
        manifest.update(ttft_p90_limit_ms=TTFT_LIMIT_MS, tpot_p90_limit_ms=TPOT_LIMIT_MS)
    write_json(args.output_dir/'manifest.json', manifest)
    results = []
    maximum = 0
    first_failure = None
    first_power_of_two_failure = None
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(args.timeout, connect=15),
                                 limits=httpx.Limits(max_connections=args.max_concurrency+len(metrics)+4,
                                                     max_keepalive_connections=args.max_concurrency+len(metrics)+4)) as client:
        (await client.get(args.endpoint + '/health')).raise_for_status()
        if args.quick_pd_sweep:
            for concurrency in (1, 2, 4, 8, 16, 32, 64, 128):
                results.append(await run_round(client, args, metrics, concurrency))
            maximum = max((r['concurrency'] for r in results if r['slo_met']), default=0)
            first_power_of_two_failure = next((r['concurrency'] for r in results if not r['slo_met']), None)
            first_failure = first_power_of_two_failure
        concurrency = args.max_concurrency+1 if args.quick_pd_sweep else args.max_concurrency if args.quick_pd else 1
        while concurrency <= args.max_concurrency:
            result = await run_round(client, args, metrics, concurrency)
            results.append(result)
            if not result['slo_met']:
                first_failure = concurrency
                first_power_of_two_failure = concurrency
                break
            maximum = concurrency
            concurrency *= 2
        if not args.quick_pd_sweep and first_failure is None and maximum < args.max_concurrency:
            result = await run_round(client, args, metrics, args.max_concurrency)
            results.append(result)
            if result['slo_met']:
                maximum = args.max_concurrency
            else:
                first_failure = args.max_concurrency
        if not args.quick_pd_sweep and first_failure is not None and maximum > 0:
            for concurrency in range(first_failure-10, maximum, -10):
                result = await run_round(client, args, metrics, concurrency)
                results.append(result)
                if result['slo_met']:
                    maximum = concurrency
                    break
    report = {'maximum_tested_slo_concurrency':maximum or None, 'first_power_of_two_failure':first_power_of_two_failure,
              'upper_boundary_failed_concurrency':first_failure,
              'tested_ceiling':max(r['concurrency'] for r in results), 'rounds':results}
    write_json(args.output_dir/'summary.json', report)
    manifest['status'] = 'complete'
    write_json(args.output_dir/'manifest.json', manifest)
    print('FIXED_128K_SUITE_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', default='step5')
    parser.add_argument('--metrics', nargs='+', required=True, help='Stage addresses as NAME=URL')
    parser.add_argument('--deployment-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-concurrency', type=int, required=True)
    parser.add_argument('--quick-pd', action='store_true')
    parser.add_argument('--quick-pd-sweep', action='store_true')
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--scrape-interval', type=float, default=2)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    if args.quick_pd_sweep and (not args.quick_pd or args.max_concurrency != 128):
        parser.error('--quick-pd-sweep requires --quick-pd and --max-concurrency 128')
    if args.max_concurrency < 1 or args.scrape_interval <= 0 or args.timeout <= 0:
        parser.error('concurrency and time intervals must be positive')
    if any('=' not in item for item in args.metrics):
        parser.error('--metrics requires NAME=URL for each stage')
    asyncio.run(main(args))
