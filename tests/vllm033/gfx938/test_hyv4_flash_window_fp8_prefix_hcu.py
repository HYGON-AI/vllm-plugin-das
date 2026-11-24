# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""gfx938 production FP8 window leaf with real writer, producer and remap.

Construct only the leaf runtime state; actual model startup independently covers
the full constructor/IndexGroup lifecycle. No numerical operator is mocked.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import time
import traceback
import gc


import pytest
from types import SimpleNamespace


@pytest.mark.hcu
def test_production_window_prefix_native_replay(tmp_path, monkeypatch):
    args = SimpleNamespace(device=0, output=tmp_path/'native-report.json',
                           rtol=.01, atol=.01, reference_rtol=.01,
                           reference_atol=.01, repeats=10, warmup=3, seed=147)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'scope': 'actual production leaf numerical/capture qualification; full model separate',
              'started_unix': time.time(), 'arguments': vars(args).copy(), 'cases': []}
    report['arguments']['output'] = str(args.output)

    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    try:
        import torch
        import torch.nn.functional as F
        from vllm.platforms import current_platform
        from vllm import _custom_ops as ops
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index as remap,
        )
        from hcu_vllm_das.models.hy_v4_flash import swa
        from hcu_vllm_das.models.hy_v4_flash.sparse_backend import FlashSparseImpl
        from hcu_vllm_das.models.hy_v4_flash.flashmla_ops import FlashMLAOps
        from flash_mla import flash_mla_interface as vendor

        assert type(current_platform).__module__ == 'hcu_adapter.bridges.vllm', (
            'Use the qualified HCU plugin environment; do not spoof platform/arch')
        assert torch.version.hip and torch.cuda.is_available()
        torch.cuda.set_device(args.device)
        device = torch.device('cuda', args.device)
        prop = torch.cuda.get_device_properties(device)
        arch = getattr(prop, 'gcnArchName', '')
        assert arch.split(':')[0] == 'gfx938', arch
        torch.manual_seed(args.seed)
        report['identity'] = {'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'torch': torch.__version__, 'hip': torch.version.hip,
                              'device': str(device), 'architecture': arch,
                              'platform': type(current_platform).__module__,
                              'vendor_file': vendor.__file__,
                              'vendor_sha256': hashlib.sha256(Path(vendor.__file__).read_bytes()).hexdigest(),
                              'window_source': inspect.getsourcefile(swa.window_indices),
                              'window_source_sha256': hashlib.sha256(
                                  Path(inspect.getsourcefile(swa.window_indices)).read_bytes()).hexdigest(),
                              'writer_schema': str(torch.ops._C_cache_ops.concat_and_cache_mla.default._schema)}

        # Actual DS-MLA V3.2 format, documented by installed Core and vendor:
        # 512 FP8 NoPE bytes + 4 FP32 group scales (16 bytes) + 64 BF16 RoPE (128).
        # All physical slots are filled by the same native writer used by the
        # production MLAImpl.do_kv_cache_update, never by random uint8 values.
        blocks, page, row_bytes = 64, 64, 656
        cache = torch.zeros((blocks, page, row_bytes), dtype=torch.uint8, device=device)
        kv_c = torch.randn((blocks * page, 512), dtype=torch.bfloat16, device=device) * .25
        k_pe = torch.randn((blocks * page, 64), dtype=torch.bfloat16, device=device) * .125
        slots = torch.arange(blocks * page, dtype=torch.int64, device=device)
        scale = torch.ones((), dtype=torch.float32, device=device)

        def encode_cache():
            ops.concat_and_cache_mla(kv_c, k_pe, cache, slots, 'fp8_ds_mla', scale, False)
            torch.cuda.synchronize(device)
            rows = cache.view(-1, row_bytes)
            stored_scales = rows[:, 512:528].contiguous().view(torch.float32)
            assert bool(torch.isfinite(stored_scales).all()) and bool((stored_scales > 0).all())
            rope = rows[:, 528:656].contiguous().view(torch.bfloat16)
            assert torch.equal(rope, k_pe), 'Native writer/cache format mismatch: RoPE roundtrip'

        encode_cache()
        report['cache_encoding'] = {'format': 'fp8_ds_mla', 'shape': list(cache.shape),
                                     'stride': list(cache.stride()),
                                     'filled_by': 'ops.concat_and_cache_mla -> _C_cache_ops',
                                     'checks': ['finite positive native group scales', 'exact BF16 RoPE roundtrip']}
        save()

        # A logical long history mapped to compact physical storage. The logical
        # block table is a permutation modulo 64 blocks, so the recent <=9 blocks
        # touched by any 512-token window do not alias each other. No random
        # invalid cache pointer or fabricated byte encoding is introduced.
        max_logical = 1_000_123
        logical_blocks = math.ceil((max_logical + 1) / page)
        block_table = ((torch.arange(logical_blocks, device=device, dtype=torch.int32)
                        * 17 + 3) % blocks).unsqueeze(0).contiguous()
        boundaries = [max_logical, 0, 1, 510, 511, 512, 513, 1023, 2047, 4095, -1]

        def make_physical(T, shift=0):
            positions = torch.tensor([boundaries[(i + shift) % len(boundaries)]
                                      for i in range(T)], dtype=torch.int64, device=device)
            # One request per query row supplies arbitrary long and boundary
            # histories, plus one inactive trailing row when positions is -1.
            # Only the context lookup is a fixture: execute the real producer
            # forward, its live buffer writes and the real Core remapper.
            order = torch.argsort((positions < 0).int(), stable=True)
            positions = positions[order]
            active = int((positions >= 0).sum())
            buffer = torch.empty((T, 2048), dtype=torch.int32, device=device)
            producer = swa.FlashWindowIndexer(512, buffer, "native-window")
            metadata = SimpleNamespace(
                num_actual_tokens=active,
                seq_lens=(positions[:active] + 1).int(),
                query_start_loc=torch.arange(active+1, dtype=torch.int32, device=device),
            )
            monkeypatch.setattr(swa, "get_forward_context", lambda: SimpleNamespace(
                attn_metadata={"native-window": metadata}))
            logical = producer(torch.empty((T, 1), device=device), None, positions)
            assert torch.equal(logical, swa.window_indices(positions, 512, 2048))
            assert bool((logical[:, 512:] == -1).all())
            req_id = torch.zeros(T, dtype=torch.int32, device=device)
            physical = remap(req_id, block_table, logical, BLOCK_SIZE=page,
                             NUM_TOPK_TOKENS=2048, return_valid_counts=False)
            assert bool((physical[:, 512:] == -1).all()), 'Remap moved valid index beyond prefix512'
            # Independent CPU interpretation of actual Core remap formula.
            lcpu = logical.cpu(); table_cpu = block_table.cpu()[0]
            expected = lcpu.clone()
            valid = lcpu >= 0
            expected[valid] = table_cpu[(lcpu[valid] // page).long()] * page + lcpu[valid] % page
            assert torch.equal(physical.cpu(), expected)
            assert bool((physical[physical >= 0] < blocks * page).all())
            return physical.unsqueeze(0), positions >= 0, producer

        def difference(a, b):
            mask = torch.isfinite(a) & torch.isfinite(b)
            delta = (a.float() - b.float()).abs()[mask]
            return {'max_abs_finite': float(delta.max()) if delta.numel() else None,
                    'finite_entries': int(mask.sum()),
                    'same_nan_mask': bool(torch.equal(torch.isnan(a), torch.isnan(b))),
                    'same_posinf_mask': bool(torch.equal(torch.isposinf(a), torch.isposinf(b))),
                    'same_neginf_mask': bool(torch.equal(torch.isneginf(a), torch.isneginf(b)))}

        for T in (1, 6, 63, 64, 65, 192, 384, 768, 8192):
            for real_heads in (16, 8):
                physical, live, owner = make_physical(T)
                q_real = torch.randn((1, T, real_heads, 576), dtype=torch.bfloat16, device=device) * .125
                q = F.pad(q_real, (0, 0, 0, 64 - real_heads))
                sinks = torch.full((64,), float('-inf'), dtype=torch.float32, device=device)
                sinks[:real_heads] = torch.linspace(-1, 1, real_heads, device=device)
                # Independent scheduler objects per width, retained through capture.
                schedulers = {2048: vendor.get_mla_metadata()[0]}
                fresh_capture_audit = []
                fresh_call_count = 0

                # Same Core metadata object is deliberately interleaved between
                # compressed2048 and actual window512 instances. The latter
                # must allocate its own fresh scheduler without changing shared.
                from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
                metadata = FlashMLASparseMetadata.FP8KernelMetadata(
                    scheduler_metadata=schedulers[2048],
                    dummy_block_table=torch.empty((1,1),dtype=torch.int32,device=device),
                    cache_lens=torch.full((1,),1048576,dtype=torch.int32,device=device))
                source = owner.topk_indices_buffer
                impls = {}

                def recorded_vendor(**kwargs):
                    nonlocal fresh_call_count
                    scheduler = kwargs['tile_scheduler_metadata']
                    width = kwargs['indices'].shape[-1]
                    before = scheduler.have_initialized
                    if width == 512:
                        assert not before and scheduler is not metadata.scheduler_metadata
                        fresh_call_count += 1
                    result = vendor.flash_mla_with_kvcache(**kwargs)
                    assert scheduler.have_initialized and scheduler.config.topk == width
                    if width == 512 and torch.cuda.is_current_stream_capturing():
                        fresh_capture_audit.append(dict(before_initialized=before,
                            after_initialized=scheduler.have_initialized,
                            topk=scheduler.config.topk))
                    return result

                provider = FlashMLAOps(vendor.flash_mla_sparse_fwd, recorded_vendor,
                                      vendor.get_mla_metadata, vendor.__name__, False)
                for width in (2048,512):
                    impl = FlashSparseImpl.__new__(FlashSparseImpl)
                    impl._hcu_ops = provider
                    impl._hcu_window_device = device
                    impl._indexer = owner if width == 512 else None
                    impl._topk_indices_buffer = source
                    impl.fp8_decode_padded_heads = 64
                    impl.softmax_scale = 576 ** -.5
                    impl.sinks = sinks[:real_heads]
                    impls[width] = impl

                def call(width, use_metadata=True):
                    return impls[width]._fp8_flash_mla_kernel(
                        q[:, :, :real_heads], cache, physical,
                        metadata if use_metadata else None)

                def oracle(output, label):
                    # CPU float64 reference of the SAME native-encoded cache.
                    # The format is verified by encode_cache(), not an invented
                    # uint8 approximation. Sample all boundary patterns, valid rows only.
                    chosen = [i for i in range(min(T, len(boundaries))) if bool(live[i])]
                    rows = cache.detach().cpu().reshape(-1, 656)
                    nope_fp8 = rows[:, :512].contiguous().view(torch.float8_e4m3fn).double()
                    scales = rows[:, 512:528].contiguous().view(torch.float32).double()
                    nope = nope_fp8 * scales.repeat_interleave(128, dim=1)
                    rope = rows[:, 528:656].contiguous().view(torch.bfloat16).double()
                    keys = torch.cat((nope, rope), dim=1)
                    assert torch.isfinite(keys).all() and (nope != 0).any()
                    idx = physical[0, chosen, :512].cpu().long()
                    qq = q[0, chosen, :real_heads].cpu().double()
                    ss = sinks[:real_heads].cpu().double()
                    predicted, predicted_lse = [], []
                    for row, qrow in zip(idx, qq):
                        ids = row[row >= 0]
                        assert ids.numel() > 0
                        logits = qrow @ keys[ids].T * (576 ** -.5)
                        lse = torch.logsumexp(logits, dim=1)
                        denominator = torch.logaddexp(lse, ss)
                        weights = torch.exp(logits - denominator[:, None])
                        predicted.append((weights @ nope[ids]).to(torch.bfloat16).double())
                        predicted_lse.append(lse)
                    predicted = torch.stack(predicted)
                    predicted_lse = torch.stack(predicted_lse)
                    actual = output[0][0, chosen, :real_heads].cpu().double()
                    actual_lse = output[1][0, :real_heads, chosen].T.cpu().double()
                    # Finite valid output/LSE must be real data, not uninitialized/NaN.
                    assert torch.isfinite(actual).all() and torch.isfinite(actual_lse).all()
                    if not torch.isposinf(ss).all():
                        assert (actual != 0).any(), 'Degenerate all-zero native output'
                        assert (predicted != 0).any(), 'Degenerate all-zero reference output'
                    torch.testing.assert_close(actual, predicted, rtol=args.reference_rtol,
                                               atol=args.reference_atol)
                    torch.testing.assert_close(actual_lse, predicted_lse,
                                               rtol=args.reference_rtol, atol=args.reference_atol)
                    return dict(label=label, sampled_rows=chosen,
                                reference_output_maxabs=float(predicted.abs().max()),
                                native_output_maxabs=float(actual.abs().max()),
                                output_maxabs_error=float((actual-predicted).abs().max()),
                                lse_maxabs_error=float((actual_lse-predicted_lse).abs().max()))

                def compare(left, right):
                    # Native empty rows can be undefined before Core's optional
                    # neutralization. Compare valid rows strictly and record
                    # whether native inactive outputs are finite, without
                    # inventing a production neutralization contract.
                    for output in (left, right):
                        assert tuple(output[0].shape) == (1, T, real_heads, 512)
                        assert tuple(output[1].shape) == (1, real_heads, T)
                    a, b = left[0][:, :, :real_heads], right[0][:, :, :real_heads]
                    la, lb = left[1][:, :real_heads], right[1][:, :real_heads]
                    torch.testing.assert_close(a[:, live], b[:, live], rtol=args.rtol, atol=args.atol)
                    torch.testing.assert_close(la[..., live], lb[..., live], rtol=args.rtol, atol=args.atol)
                    return {'output': difference(a[:, live], b[:, live]),
                            'lse': difference(la[..., live], lb[..., live]),
                            'inactive_rows': int((~live).sum()),
                            'inactive_native_finite': [bool(torch.isfinite(x[0][:, ~live]).all())
                                                       for x in (left, right)]}

                baseline = call(2048); again = call(2048)
                repeatability = compare(baseline, again)
                narrow = call(512)
                initial = compare(baseline, narrow)
                compare(baseline, call(512, use_metadata=False))
                assert metadata.scheduler_metadata.config.topk == 2048
                original_physical = physical.clone()
                call(512); call(2048); call(512)
                assert torch.equal(physical, original_physical)
                assert metadata.scheduler_metadata.config.topk == 2048
                oracles = [oracle(baseline, 'eager2048-initial'), oracle(narrow, 'eager512-initial')]
                graphs, captured, kept = {}, {}, {}
                for width in (2048, 512):
                    for _ in range(args.warmup):
                        call(width)
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    # Independent graph pools: no graph_pool_handle sharing.
                    with torch.cuda.graph(graph):
                        captured[width] = call(width)
                    graphs[width] = graph
                    torch.cuda.synchronize(device)
                    kept[width] = tuple(x.clone() for x in captured[width])
                torch.cuda.synchronize(device)
                gc.collect()  # Fresh512 Python metadata must not be kept alive by this probe.
                assert fresh_capture_audit and all(not x['before_initialized'] for x in fresh_capture_audit)
                # Capture of the opposite bank itself must preserve the first.
                for now, before in zip(captured[2048], kept[2048]):
                    assert torch.equal(now.reshape(-1).contiguous().view(torch.uint8),
                                       before.reshape(-1).contiguous().view(torch.uint8)), (
                        'Opposite graph capture overwrote retained output')
                for width in (2048,512):
                    graphs[width].replay()
                torch.cuda.synchronize(device)
                oracles += [oracle(captured[w], 'captured-initial-'+str(w)) for w in (2048,512)]
                replay_evidence = []
                for change in range(3):
                    q.copy_(F.pad(torch.randn_like(q_real) * .125, (0, 0, 0, 64-real_heads)))
                    kv_c.copy_(torch.randn_like(kv_c) * .25)
                    k_pe.copy_(torch.randn_like(k_pe) * .125)
                    encode_cache()
                    changed, changed_live, _ = make_physical(T, change + 1)
                    physical.copy_(changed); live = changed_live
                    # Same sink storage, changed contents: finite / +/-inf are
                    # qualified independently from width, under both graphs.
                    if change == 0:
                        sinks[:real_heads].fill_(float('inf'))
                    elif change == 1:
                        sinks[:real_heads].fill_(float('-inf'))
                    else:
                        sinks[:real_heads].copy_(torch.linspace(-2, 2, real_heads, device=device))
                    for width in (2048, 512):
                        graphs[width].replay()
                    torch.cuda.synchronize(device)
                    replay_evidence.append(compare(captured[2048], captured[512]))
                    for kept_width, opposite in ((2048,512),(512,2048)):
                        snap = tuple(x.clone() for x in captured[kept_width])
                        graphs[opposite].replay(); graphs[opposite].replay()
                        torch.cuda.synchronize(device)
                        for now, before in zip(captured[kept_width], snap):
                            # Include native inactive rows: ownership is byte-level.
                            assert torch.equal(now.reshape(-1).contiguous().view(torch.uint8),
                                               before.reshape(-1).contiguous().view(torch.uint8)), (
                                'Opposite graph overwrote retained output', kept_width, opposite)
                # Single-variable liveness tests: cached/equal zero outputs cannot pass.
                single_variable = []
                sinks[:real_heads].copy_(torch.linspace(-1,1,real_heads,device=device))
                for w in (2048,512): graphs[w].replay()
                torch.cuda.synchronize(device)
                previous = captured[2048][0].clone()
                previous_live = live.clone()
                for variable in ('q_only', 'kv_only', 'physical_indices_only', 'sink_only'):
                    if variable == 'q_only':
                        q.mul_(4)
                    elif variable == 'kv_only':
                        kv_c.mul_(2); k_pe.mul_(1.5); encode_cache()
                    elif variable == 'physical_indices_only':
                        changed, live, _ = make_physical(T, 5); physical.copy_(changed)
                    else:
                        sinks[:real_heads].add_(4)
                    for w in (2048,512): graphs[w].replay()
                    torch.cuda.synchronize(device)
                    compare(captured[2048], captured[512])
                    common_live = previous_live & live
                    assert bool(common_live.any())
                    common_delta = (captured[2048][0][:,common_live, :real_heads].float()
                                    -previous[:,common_live,:real_heads].float()).abs()
                    assert bool(torch.isfinite(common_delta).all())
                    assert bool((common_delta > 0).any()), ('Changed input did not change output',variable)
                    # Fresh eager references also detect a replay stuck on capture-time inputs.
                    fresh_eager = call(512)
                    compare(captured[512], fresh_eager)
                    oracles += [oracle(captured[w], variable+'-'+str(w)) for w in (2048,512)]
                    single_variable.append(dict(variable=variable, output_changed=True,
                                                maxabs_change=float(common_delta.max())))
                    previous = captured[2048][0].clone()
                    previous_live = live.clone()
                timing = {2048: [], 512: []}
                for rep in range(args.repeats):
                    order = (2048,512) if rep % 2 == 0 else (512,2048)
                    for width in order:
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(); graphs[width].replay(); end.record(); end.synchronize()
                        timing[width].append(begin.elapsed_time(end))
                record = {'T': T, 'actual_heads': real_heads, 'q_shape': list(q.shape),
                          'cache_shape': list(cache.shape), 'indices_shape_2048': list(physical.shape),
                          'indices_512_view_stride': list(physical[..., :512].stride()),
                          'baseline_repeatability': repeatability, 'width_comparison': initial,
                          'changed_input_graph_replays': replay_evidence,
                          'scheduler_policy': '2048persistent;512fresh-per-Python-call including first initialization inside capture',
                          'fresh512_call_count': fresh_call_count, 'fresh512_capture_audit': fresh_capture_audit,
                          'same_cache_cpu_float64_oracles': oracles, 'single_variable_liveness': single_variable,
                          'independent_scheduler_identity': True, 'retained_cross_width_output': True,
                          'graph_pools': 'independent default CUDAGraph pools',
                          'timing_ms': {str(w): {'samples': v, 'median': statistics.median(v),
                                                'p95_nearest_rank': sorted(v)[math.ceil(.95*len(v))-1]}
                                        for w,v in timing.items()}}
                report['cases'].append(record); save()
        report['status'] = 'pass'
        report['qualification_limits'] = [
            'Production leaf numerical test; full IndexGroup/model lifecycle and EvalScope separate',
            'Shared Core scheduler interleaved at real leaf; full model aliases require model controls',
            '512window qualification does not qualify1024compressed or TP8/EP/DP',
            'Native inactive outputs may be undefined; caller neutralization contract must be qualified in model',
            'No1200TPS/30ms/larger-concurrency claim from component event timing']
        save()
    except BaseException:
        report['status'] = 'fail'
        report['error'] = traceback.format_exc(); save(); raise
