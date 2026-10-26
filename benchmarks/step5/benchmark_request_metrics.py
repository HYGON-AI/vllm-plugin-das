"""Run the existing benchmark and export statistics after its timing finishes."""
import asyncio
import json
from pathlib import Path
import sys

import benchmark_hcu_metadata as adapter
from request_latency_stats import summarize_requests

iterations = []
original_generate = adapter.b.VLLMBenchmarkClient.generate_completion_stream
original_test = adapter.b.VLLMBenchmarkClient.test_streaming_benchmark


async def capture_request(self, *args, **kwargs):
    result = await original_generate(self, *args, **kwargs)
    # Constant-time bookkeeping only; all statistics and writes happen later.
    self._request_metrics_capture.append(result)
    return result


async def capture_iteration(self, batch_prompts, output_len, validate_decode_quality=True):
    self._request_metrics_capture = []
    metrics = await original_test(self, batch_prompts, output_len, validate_decode_quality)
    stats, requests = summarize_requests(self._request_metrics_capture, metrics)
    iterations.append(dict(measured=validate_decode_quality, batch=len(requests),
                           input_len=requests[0]['input_tokens'], output_len=output_len,
                           statistics=stats, requests=requests))
    return metrics


adapter.b.VLLMBenchmarkClient.generate_completion_stream = capture_request
adapter.b.VLLMBenchmarkClient.test_streaming_benchmark = capture_iteration

if __name__ == '__main__':
    result = asyncio.run(adapter.b.main())
    if result:
        raise SystemExit(result)
    output = Path(sys.argv[sys.argv.index('--output-file') + 1])
    report = json.loads(output.read_text())
    measured = [x for x in iterations if x['measured']]
    if len(measured) != 1 or measured[0]['batch'] != 64:
        raise ValueError('This export requires one complete measured C64 iteration')
    item = measured[0]
    key = f"bs{item['batch']}_input{item['input_len']}_output{item['output_len']}"
    report[key].update(item['statistics'])
    report['latency_measurement'] = dict(
        units='new *_ms fields: milliseconds; existing latency/ITL fields: original seconds; throughput: tokens/s; *_s timestamps/durations: seconds',
        clock='client time.perf_counter, monotonic seconds',
        percentiles='linear interpolation at (sample_count-1)*q/100',
        ttft='request start to first nonempty output chunk, includes queueing and transport',
        tpot='arithmetic mean of each request (last actual token-first actual token)/(output_tokens-1)',
        itl='original benchmark common-decode-window SSE chunk interval; multi-token MTP chunks are not split',
        output_throughput='all output tokens / (last response timestamp-earliest request start), includes prefill',
        prefill_throughput='original total input tokens / max request TTFT',
        token_level_itl_limitation='individual emission times inside one multi-token chunk are not observable',
        unavailable_metrics='null when too few output tokens or no common-window ITL samples',
        statistics_written_after_original_timing=True,
    )
    raw = output.with_suffix('.requests.json')
    raw.write_text(json.dumps(dict(schema_version=1, iterations=iterations), separators=(',', ':')))
    report['request_timestamps_file'] = raw.name
    output.write_text(json.dumps(report, indent=2))
    print('REQUEST_LATENCY_METRICS ' + json.dumps(item['statistics']), flush=True)
