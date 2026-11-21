#!/usr/bin/env python3
"""Native W4A8 activation differential, graph, consumer and latency gates."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import traceback

import torch

# Helpers from the original CT W4A8 runtime-capacity probe.
def exact(a, b, message):
    if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
        raise AssertionError(message)


def timing(call, warmup=15, repeats=80):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    # Ten native launches in each replay amortize Python/event submission.
    # This is also the execution mode used for the captured serving fragments.
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(10):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / 10)
    ordered = sorted(values)
    return {'median_us': statistics.median(values),
            'p10_us': ordered[int(len(ordered) * .1)],
            'p90_us': ordered[int(len(ordered) * .9)], 'samples_us': values,
            'method': 'CUDA events around graph replay; 10 native launches per replay'}


def guarded_scale(shape, value, device):
    # Match serving's HIPC channel-scale allocation/read-ahead contract.
    size = 1
    for extent in shape:
        size *= extent
    storage = torch.zeros(size + (2 * 1024 * 1024) // 4, device=device, dtype=torch.float32)
    result = storage[:size].view(shape)
    result.fill_(value)
    return result


def run(args, report):
    import deepgemm
    from lightop import fuse_silu_mul_clamp_quant, fuse_silu_mul_clamp_quant_ep

    module_path = Path(__file__).resolve().parents[2] / (
        'adapter_extension/src/hcu_vllm_das/moe/clamped_int8_triton.py')
    spec = importlib.util.spec_from_file_location('clamped_int8_probe', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = module.clamped_int8_2304
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(':')[0] == 'gfx938' and props.multi_processor_count == 64
    report['hardware'] = dict(name=props.name, arch=props.gcnArchName,
                              cus=props.multi_processor_count)
    torch.manual_seed(30419)
    report['cases'], report['timings'] = [], []

    for masked, experts in ((False, 16), (True, 16), (True, 48)):
        rows = 2048 if masked else 4096
        shape = (experts, rows, 4608) if masked else (rows, 4608)
        x = torch.empty(shape, device='cuda', dtype=torch.bfloat16)
        meta = torch.empty((experts if masked else rows,), device='cuda', dtype=torch.int32)

        def layout(count):
            if masked:
                counts = torch.tensor([max(0, min(rows, count + e % 3))
                                       if count else 0 for e in range(experts)],
                                      device='cuda', dtype=torch.int32)
                meta.copy_(counts)
                return torch.arange(rows, device='cuda')[None, :] < counts[:, None]
            ids = torch.arange(rows, device='cuda', dtype=torch.int32) // 256
            live = torch.arange(rows, device='cuda') % 256 < min(count, 256)
            meta.copy_(torch.where(live, ids, -1))
            return live

        def baseline():
            if masked:
                return fuse_silu_mul_clamp_quant_ep(x, 10., meta, 32)
            return fuse_silu_mul_clamp_quant(x, 10.)

        def optimized():
            return candidate(x, meta, masked=masked, expected_m=32)

        def compare(label, active, captured=None):
            native = baseline()
            result = optimized() if captured is None else captured
            torch.cuda.synchronize()
            exact(native[0][active], result[0][active], label + ': INT8 codes')
            exact(native[1][active], result[1][active], label + ': FP32 scale bits')
            if not torch.isfinite(result[1][active]).all():
                raise AssertionError(label + ': nonfinite active scales')
            report['cases'].append(dict(masked=masked, experts=experts,
                case=label, active_rows=int(active.sum()), bytes_exact=True))
            Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
            return native, result

        for pattern in ('zero', 'tiny', 'random', 'saturated', 'bf16_gate_sweep'):
            active = layout(32 if pattern != 'saturated' else rows)
            x.normal_(0, 8)
            if pattern == 'zero':
                x.zero_()
            elif pattern == 'tiny':
                x.mul_(2.**-25)
            elif pattern == 'saturated':
                x.mul_(100)
            elif pattern == 'bf16_gate_sweep':
                # Every finite BF16 gate bit pattern, repeated across live rows.
                values = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
                values = values.view(torch.bfloat16)
                values = values[torch.isfinite(values)]
                live = x[active].reshape(-1, 4608)
                gate = values.repeat((live.shape[0]*2304+len(values)-1)//len(values))
                live[:, :2304] = gate[:live.shape[0]*2304].view(-1, 2304)
                live[:, 2304:] = 10.
                x[active] = live
            x[~active] = float('nan')
            original = x.clone()
            compare(pattern, active)
            exact(original, x, 'activation changed input')

        active = layout(24)
        x.normal_(0, 8)
        optimized()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = optimized()
        torch.cuda.current_stream().wait_stream(stream)
        for bucket in (0, 1, 4, 8, 16, 32, 40, 48, 64, 160, 192, 256, 2048):
            active = layout(bucket)
            x.normal_(0, 4 if bucket % 2 else 25)
            x[~active] = float('nan')
            graph.replay()
            compare('fresh_graph_count%d' % bucket, active, captured)
        del graph, captured

        if args.consumer:
            # Actual native down projection: INT8 codes/scales are observable
            # through W4A8 packed-weight ABI, including inactive NaN padding.
            logical = torch.randint(-8, 8, (experts, 5120, 2304),
                                    device='cuda', dtype=torch.int8)
            packed = (((logical[..., 0::2].to(torch.int16) & 15) << 4)
                      | (logical[..., 1::2].to(torch.int16) & 15)).to(torch.int8)
            del logical
            weight = deepgemm.pack_w4a8_moe_hipc_weight(packed)
            if masked:
                weight = deepgemm.view_w4a8_moe_hipc_weight_n32_layout(weight)
            scale = guarded_scale((experts, 5120, 1), 1/32, 'cuda')
            out_shape = (*shape[:-1], 5120)
            old = torch.empty(out_shape, device='cuda', dtype=torch.bfloat16)
            new = torch.empty_like(old)
            for count in (0, 1, 8, 32, 64, 256, 2048):
                active = layout(count)
                x.normal_(0, 8)
                x[~active] = float('nan')
                a, b = compare('consumer_count%d' % count, active)
                if masked:
                    for activation, out in ((a, old), (b, new)):
                        deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc(
                            activation, (weight, scale), out, meta, 32)
                else:
                    for activation, out in ((a, old), (b, new)):
                        deepgemm.m_grouped_w4a8_gemm_nt_contiguous_hipc(
                            activation, (weight, scale), out, meta)
                torch.cuda.synchronize()
                exact(old[active], new[active], 'actual native down GEMM active bytes')
                report['cases'][-1]['native_consumer_exact'] = True
            del weight, packed, scale, old, new

        for count in (8, 32, 64):
            active = layout(count)
            x.normal_(0, 8)
            old_time, new_time = timing(baseline), timing(optimized)
            report['timings'].append(dict(masked=masked, experts=experts,
                count=count, baseline=old_time, candidate=new_time))
        del x, meta
        torch.cuda.empty_cache()
    report['passed'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--consumer', action='store_true')
    args = parser.parse_args()
    report = {'passed': False, 'status': 'running', 'consumer_requested': args.consumer}
    try:
        run(args, report)
        report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
