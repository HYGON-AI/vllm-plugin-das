#!/usr/bin/env python3
"""Single-node FP4 P-capability test, followed by ON-only HumanEval-20. Default: plan only."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from gpu_guard import inventory, select, snapshot, require_idle

HERE = Path(__file__).resolve().parent
MODEL_NAME = 'dsv41-fp4-layer-split'
INPUTS = (128, 4096, 12288, 32768)
OUTPUT = 16
GPU_MEMORY_UTILIZATION = '0.9'
ENVIRONMENT = dict(
    PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
    HCU_ADAPTER_HOST='vllm', VLLM_PLUGINS='hcu_adapter',
    HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS='das_hcu,das_hcu_dsv41,das_hcu_v41_indexer,das_hcu_v41_pcp,das_hcu_fp4_packer',
    HCU_VLLM_FP4_INDEXER_IMPL='lightop', HCU_VLLM_FP4_PACKER_IMPL='triton',
    HCU_VLLM_SPARSE_INDEXER_IMPL='stock', HCU_VLLM_MHC_IMPL='aiter',
    VLLM_HCU_USE_AITER_TILELANG_MHC='1', HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES='',
    HCU_VLLM_PCP_IMPL='native', VLLM_HCU_DSV41_PCP_ATTENTION='1',
    VLLM_HCU_DSV41_FP8_INDEXER_REFERENCE='1', VLLM_USE_V2_MODEL_RUNNER='1',
    VLLM_USE_BREAKABLE_CUDAGRAPH='0', VLLM_ENGINE_READY_TIMEOUT_S='3600',
    VLLM_ROCM_USE_AITER='0', VLLM_ROCM_USE_AITER_MOE='0', AMDGCN_USE_BUFFER_OPS='0',
    VLLM_PLE_CPU_OFFLOAD='1', VLLM_DEEPEP_BUFFER_SIZE_MB='2048',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False', ROCSHMEM_BOOTSTRAP_TIMEOUT='1200',
    ROCSHMEM_GDR_DISABLE_XDP='1', ROCSHMEM_MAX_NUM_CONTEXTS='48',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
    VLLM_NO_USAGE_STATS='1', NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def environment(args, _split, *, cpu=False, client=False, selected=()):
    # Freeze experimental settings instead of inheriting another test's switches.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HCU_', 'VLLM_', 'TILELANG_', 'AITER_'))
           and k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'LD_PRELOAD', 'ROCBLAS_TENSILE_LIBPATH')}
    env.update(ENVIRONMENT)
    env['PYTHONPATH'] = ':'.join((str(args.project/'adapter-das'),
                                 str(args.project/'vllm-plugin-das/adapter_extension/src')))
    env.update(args.profile['runtime_env'])
    if args.profile.get('model_format') == 'ct-w4a8':
        env['HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS'] += ',das_hcu_ct_w4a8'
        env['HCU_VLLM_CT_W4A8_IMPL'] = 'native'
    if args.expert_map is not None and not (cpu or client):
        env.update(HCU_VLLM_FLASHMLA_PREFILL='native',
                   HCU_VLLM_FLASHMLA_DECODE='native', AITER_MHC_POST_KERNEL='v3')
    env['PATH'] = args.profile['path_prefix'] + ':' + env.get('PATH', '')
    env.update(VLLM_HOST_IP=args.profile['host_ip'],
               ROCBLAS_TENSILE_LIBPATH=args.profile['tuning_path'])
    env.update(HIP_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               ROCR_VISIBLE_DEVICES=','.join(g['uuid'] for g in selected))
    if cpu or client:
        env.update(HCU_ADAPTER_HOST='none', VLLM_PLUGINS='__disabled__',
                   HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
        env.pop('LD_PRELOAD', None)
    if client:
        env.pop('PYTHONPATH', None)
    return env


def static_eplb_settings(args):
    return {'hcu_static_eplb': {'expert_map_path': str(args.expert_map),
        'num_redundant_experts': 32, 'static_dispatch_policy': 'locality_fair'}}


def server_command(args, *, split=True):
    fixed_kv = ['--kv-cache-memory-bytes', str(args.kv_bytes)] if args.kv_bytes is not None else []
    command = [args.profile['python'], '-B', '-m', 'vllm.entrypoints.cli.main', 'serve', args.profile['model'],
        '--tensor-parallel-size', '1', '--data-parallel-size', '1', '--prefill-context-parallel-size', '8',
        '--enable-expert-parallel', '--language-model-only', '--moe-backend', 'deep_gemm',
        '--all2all-backend', 'deepep_high_throughput', '--enforce-eager',
        '--gpu-memory-utilization', GPU_MEMORY_UTILIZATION, *fixed_kv,
        '--block-size', '64', '--max-num-seqs', '128', '--max-num-batched-tokens', '32768',
        '--max-model-len', '65536', '--enable-chunked-prefill',
        '--attention-config', '{"indexer_kv_dtype":"mxfp4"}', '--no-enable-prefix-caching',
        '--swa-bounded-replay' if args.replay else '--no-swa-bounded-replay',
        '--enable-prompt-tokens-details', '--engram-config',
        '{"cpu_offload":true,"dp_shared_memory":false,"embedding_across_dp":false,"use_thp":false}',
        '--reasoning-parser', 'deepseek_v41', '--skip-mm-profiling', '--host', '127.0.0.1',
        '--port', str(args.port), '--served-model-name', MODEL_NAME] + (
        ['--quantization', 'compressed-tensors', '--dtype', 'bfloat16']
        if args.profile.get('model_format') == 'ct-w4a8' else []) + (
        ['--speculative-config', json.dumps(dict(method='dspark', num_speculative_tokens=5,
                                                moe_backend='deep_gemm'))]
        if args.profile.get('dspark', False) else [])
    if split:
        command.append('--enable-layer-split')
    if args.expert_map is not None:
        command += [
            '--additional-config', json.dumps(static_eplb_settings(args)),
            '--profiler-config', json.dumps(dict(
                profiler='torch',
                torch_profiler_dir=str(args.output/'capacity_on'/'profiles'),
                torch_profiler_with_stack=True,
                torch_profiler_use_gzip=True,
                torch_profiler_record_shapes=True,
            )),
        ]
    return command


def evaluator_command(args, output):
    return [args.profile['eval_python'], '-B', str(HERE/'evaluate_humaneval.py'),
        '--api-url', f'http://127.0.0.1:{args.port}/v1', '--model', MODEL_NAME,
        '--data', args.profile['data'], '--batch-size', '16', '--max-tokens', '2048',
        '--limit', '20', '--output', str(output)]


def source_state(args):
    result = {}
    for repo in ('adapter-das', 'vllm-plugin-das'):
        path = args.project/repo
        cmd = ['git', '-c', 'safe.directory='+str(path), '-C', str(path)]
        def git(*options):
            return subprocess.check_output(cmd+list(options), timeout=60)
        if git('diff', '--name-only', '--diff-filter=U').strip():
            raise RuntimeError('Unresolved conflicts: '+repo)
        diff = git('diff', 'HEAD', '--binary')
        paths = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
        result[repo] = dict(commit=git('rev-parse', 'HEAD').decode().strip(),
            diff_sha256=hashlib.sha256(diff).hexdigest(), dirty=bool(diff),
            untracked={n: hashlib.sha256((path/n).read_bytes()).hexdigest() for n in paths
                       if n and (path/n).is_file() and n.endswith(('.py', '.sh', '.json'))})
    return result


def preflight(args):
    for field in ('python', 'eval_python', 'model', 'data', 'tuning_path', 'path_prefix', 'vendor_manifest'):
        if not Path(args.profile[field]).exists():
            raise FileNotFoundError(args.profile[field])
    for key in ('AITER_JIT_DIR', 'LD_PRELOAD'):
        if key in args.profile['runtime_env'] and not Path(args.profile['runtime_env'][key]).exists():
            raise FileNotFoundError(args.profile['runtime_env'][key])
    command = [args.profile['python'], '-B', str(HERE/'preflight.py'),
               str(args.project), str(args.profile_path)]
    if args.expert_map is not None:
        command.append(json.dumps(static_eplb_settings(args)))
    raw = subprocess.check_output(command, env=environment(args, False, cpu=True),
        timeout=240, text=True)
    line = next(s for s in raw.splitlines() if s.startswith('PREFLIGHT_JSON='))
    proof = json.loads(line.split('=', 1)[1])
    subprocess.run(evaluator_command(args, args.output/'check-unused')+['--check'],
                   env=environment(args, False, client=True), check=True, timeout=240)
    proof['sources'] = source_state(args)
    proof['scripts'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir()
                        if p.is_file() and p.suffix in ('.py', '.sh', '.json')}
    return proof


def http(args, endpoint, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{args.port}'+endpoint, data,
                                     {'Content-Type': 'application/json'})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: '+error.read().decode(errors='replace')[:2000]) from error
    if result.get('error'):
        raise RuntimeError(str(result['error']))
    return result


def parse_capacity(log, split, replay, kv_bytes):
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    capacities = {int(n.replace(',', '')) for n in re.findall(r'GPU KV cache size:\s*([\d,]+) tokens', log)}
    if len(capacities) != 1 or min(capacities) <= 0:
        raise RuntimeError('Missing or conflicting actual GPU KV cache size')
    reserved = [float(n) for n in re.findall(r'reserved ([\d.]+) GiB memory for KV Cache', log)]
    if kv_bytes is None:
        profiled = [float(n) for n in re.findall(r'Available KV cache memory:\s*(-?\d+(?:\.\d+)?)\s*GiB', log)]
        if reserved or not profiled or min(profiled) <= 0:
            raise RuntimeError('Automatic KV memory budget not confirmed by the Core startup log')
        budget_log = profiled
    else:
        if not reserved or any(abs(n-kv_bytes/2**30) > 0.011 for n in reserved):
            raise RuntimeError('Fixed KV memory budget not confirmed by the Core startup log')
        budget_log = reserved
    main = set(map(int, re.findall(r'DSA layer split PCP rank (\d+)/8 owns KV sources', log)))
    swa = set(map(int, re.findall(r'DSA layer split PCP rank (\d+)/8 owns SWA layers', log)))
    pools = set(map(int, re.findall(r'DSA independent KV pools rank (\d+):', log)))
    if split and (main != set(range(8)) or swa != main or pools != main):
        raise RuntimeError('Layer Split main/SWA/independent-pool evidence is incomplete on eight ranks')
    if not split and (main or swa or pools):
        raise RuntimeError('OFF baseline unexpectedly enabled Layer Split')
    if 'Using MXFP4 indexer cache' not in log:
        raise RuntimeError('FP4 activation evidence missing')
    replay_active = 'Decoder SWA bounded replay: in eager prefill steps' in log
    if replay_active != replay:
        raise RuntimeError('SWA Replay activation differs from the plan')
    return dict(tokens=capacities.pop(), kv_budget_bytes_per_rank=kv_bytes,
                budget_log_gib=budget_log, main_owner_ranks=sorted(main), swa_owner_ranks=sorted(swa),
                pool_ranks=sorted(pools), layer_split=split, replay=replay)


def compare_capacity(off, on):
    if off['kv_budget_bytes_per_rank'] != on['kv_budget_bytes_per_rank'] or off['replay'] != on['replay']:
        raise RuntimeError('Capacity comparison changed budget or replay policy')
    ratio = on['tokens']/off['tokens']
    fixed = off['kv_budget_bytes_per_rank'] is not None
    return dict(off_tokens=off['tokens'], on_tokens=on['tokens'], increase_tokens=on['tokens']-off['tokens'],
                ratio=ratio, increase_percent=(ratio-1)*100, increased=ratio > 1,
                kv_budget_mode='fixed' if fixed else 'auto',
                same_kv_budget_bytes_per_rank=off['kv_budget_bytes_per_rank'],
                off_kv_budget_log_gib=off['budget_log_gib'], on_kv_budget_log_gib=on['budget_log_gib'],
                scope='Actual scheduler-reported logical token capacity; not GPU-summed or theoretical')


def validate_completion(response, length):
    usage = response.get('usage', {})
    choices = response.get('choices', [])
    if usage.get('prompt_tokens') != length or usage.get('completion_tokens') != OUTPUT or len(choices) != 1:
        raise RuntimeError('Fixed-input/output usage mismatch')
    c = choices[0]
    ids, logs = c.get('token_ids', []), (c.get('logprobs') or {}).get('token_logprobs', [])
    if (c.get('finish_reason') != 'length' or len(ids) != OUTPUT or len(logs) != OUTPUT
            or any(type(t) is not int for t in ids) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in logs)
            or (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)):
        raise RuntimeError('Incomplete token IDs/logprobs or unexpected cache reuse')
    return dict(token_ids=ids, logprobs=logs, text=c.get('text', ''))


def smoke_settings(args):
    return ((32768,), 0) if args.stage == 'on' else (INPUTS, 2)


def smoke(args, unit, output):
    lengths, warmups = smoke_settings(args)
    result = {}
    for length in lengths:
        tokens = (unit*((length+len(unit)-1)//len(unit)))[:length]
        save(output/f'input_{length}.json', tokens)
        for repeat in range(warmups + 1):
            response = http(args, '/v1/completions', dict(model=MODEL_NAME, prompt=tokens,
                temperature=0, seed=42, max_tokens=OUTPUT, min_tokens=OUTPUT, ignore_eos=True,
                stream=False, logprobs=5, return_token_ids=True, return_tokens_as_token_ids=True))
            save(output/f'smoke_{length}_{repeat}.json', response)
            parsed = validate_completion(response, length)
            if repeat == warmups:
                result[str(length)] = parsed
    for token_id in args.profile.get('repeated_token_smoke', ()):
        length = 32768
        label = f'32768_repeat_{token_id}'
        response = http(args, '/v1/completions', dict(model=MODEL_NAME,
            prompt=[token_id] * length, temperature=0, seed=42, max_tokens=OUTPUT,
            min_tokens=OUTPUT, ignore_eos=True, stream=False, logprobs=5,
            return_token_ids=True, return_tokens_as_token_ids=True))
        save(output/f'smoke_{label}.json', response)
        result[label] = validate_completion(response, length)
    save(output/'smoke_summary.json', result)
    return result


def compare_smoke(off, on):
    rows = {key: dict(tokens_equal=off[key]['token_ids'] == on[key]['token_ids'],
        text_equal=off[key]['text'] == on[key]['text'],
        max_logprob_abs_delta=max(abs(a-b) for a,b in zip(off[key]['logprobs'], on[key]['logprobs']))) for key in off}
    return dict(cases=rows, all_tokens_equal=all(r['tokens_equal'] for r in rows.values()),
                logprob_policy='Report deltas; no invented numerical equivalence threshold')


def stop(process):
    if process is None:
        return
    for sig, limit in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic()+limit
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(1)
    process.wait(timeout=10)


def idle(args, selected, all_cards):
    current = select(inventory(args.smi), args.gpus)
    if current != selected:
        raise RuntimeError('Physical GPU identity changed')
    return require_profile_idle(args, snapshot(selected, all_cards))


def require_profile_idle(args, state):
    return require_idle(state, args.profile['idle_vram_mib'],
                        context_idle_mib=args.profile.get('idle_kfd_context_vram_mib'))


def check_port_available(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))


@contextlib.contextmanager
def service(args, split, output, selected, all_cards):
    output.mkdir()
    save(output/'gpu_before_1.json', idle(args, selected, all_cards))
    time.sleep(2)
    check_port_available(args.port)
    save(output/'gpu_before_2.json', idle(args, selected, all_cards))
    env = environment(args, split, selected=selected)
    command = server_command(args, split=split)
    save(output/'launch.json', dict(command=command, layer_split=split,
        environment={k: env[k] for k in set(ENVIRONMENT)|set(args.profile['runtime_env'])|
                     {'PYTHONPATH', 'PATH', 'ROCBLAS_TENSILE_LIBPATH', 'VLLM_HOST_IP',
                      'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'}}))
    process = None
    stage_completed = False
    try:
        with (output/'server.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        save(output/'pid.json', dict(pid=process.pid, pgid=process.pid))
        deadline = time.monotonic()+args.startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError('Server exited during startup: '+str(process.returncode))
            try:
                models = http(args, '/v1/models', timeout=3)
                if any(m['id'] == MODEL_NAME for m in models['data']):
                    break
            except (OSError, ValueError, RuntimeError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Server startup timeout')
            time.sleep(3)
        log = (output/'server.log').read_text(errors='replace')
        capacity = parse_capacity(log, split, args.replay, args.kv_bytes)
        if args.profile.get('dspark', False):
            if log.count('HCU PCP DSpark drafter ready: attention_PCP=1 expert_PCP=8 speculative_tokens=5') != 8:
                raise RuntimeError('Missing eight-rank native DSpark draft construction evidence')
            capacity['dspark_tokens'] = 5
        if args.expert_map is not None:
            if log.count('HCU static EPLB routing ready: physical_experts=416 redundant_experts=32') != 8:
                raise RuntimeError('Missing eight-rank static EPLB routing evidence')
            capacity['static_eplb_redundant_experts'] = 32
        save(output/'capacity.json', capacity)
        yield process, capacity
        if process.poll() is not None:
            raise RuntimeError('Server exited during validation')
        stage_completed = True
    finally:
        retain = (args.keep_service and process is not None and process.poll() is None
                  and (split or not stage_completed))
        if retain:
            args.retained_service = dict(pid=process.pid, pgid=process.pid, layer_split=split,
                stage_completed=stage_completed, output=str(output), port=args.port)
            save(output/'retained_service.json', args.retained_service)
        else:
            stop(process)
            save(output/'exit.json', dict(returncode=process.returncode if process else None))
            deadline = time.monotonic()+60
            while True:
                state = snapshot(selected, all_cards)
                save(output/'gpu_after.json', state)
                try:
                    require_profile_idle(args, state)
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(3)


def evaluate(args, server, output):
    for repeat in range(2):
        response = http(args, '/v1/chat/completions', dict(model=MODEL_NAME,
            messages=[dict(role='user', content='Write Python add(a,b). Return only the function code.')],
            temperature=0, seed=42, max_tokens=64, chat_template_kwargs={'thinking': False}))
        if not response.get('choices', [{}])[0].get('message', {}).get('content'):
            raise RuntimeError('Empty HumanEval warmup')
        save(output.parent/f'chat_warmup_{repeat}.json', response)
    client = None
    try:
        with (output.parent/'evaluator.log').open('w') as log:
            client = subprocess.Popen(evaluator_command(args, output), env=environment(args, False, client=True),
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(output.parent/'evaluator_pid.json', dict(pid=client.pid, pgid=client.pid))
        deadline = time.monotonic()+args.eval_timeout
        while client.poll() is None:
            if server.poll() is not None:
                raise RuntimeError('Server exited during HumanEval')
            if time.monotonic() > deadline:
                raise TimeoutError('HumanEval timeout')
            time.sleep(3)
        if client.returncode:
            raise RuntimeError('HumanEval evaluator failed; inspect evaluator.log')
        return json.loads((output/'summary.json').read_text())
    finally:
        stop(client)
        save(output.parent/'evaluator_exit.json', dict(returncode=client.returncode if client else None))


def pipeline(args, proof):
    import fcntl
    with (args.project/'layer_split_acceptance.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_stages(args, proof)


def wait_for_idle(args, selected, all_cards, proof):
    """Wait before the first service only; never stop or replace a foreign task."""
    if not args.wait_idle_interval:
        return
    while True:
        if Path('/proc/sys/kernel/numa_balancing').read_text().strip() != '0':
            raise RuntimeError('kernel.numa_balancing changed while waiting')
        current = select(inventory(args.smi), args.gpus)
        if current != selected:
            raise RuntimeError('Physical GPU identity changed while waiting')
        state = snapshot(selected, all_cards)
        save(args.output/'waiting_gpu.json', state)
        try:
            require_profile_idle(args, state)
        except RuntimeError:
            save(args.output/'status.json', dict(state='waiting_for_idle',
                checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                controller_pid=os.getpid(), interval_seconds=args.wait_idle_interval))
            time.sleep(args.wait_idle_interval)
        else:
            # NFS Git scans are unnecessary while cards remain occupied. Verify
            # before admission; a transient timeout must not discard the wait.
            try:
                sources = source_state(args)
            except subprocess.TimeoutExpired as error:
                save(args.output/'status.json', dict(state='waiting_source_verification',
                    checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), error=repr(error),
                    controller_pid=os.getpid(), interval_seconds=args.wait_idle_interval))
                time.sleep(args.wait_idle_interval)
                continue
            if sources != proof['sources']:
                raise RuntimeError('Source changed while waiting; rerun CPU preflight')
            return  # service() performs two fresh admission checks before launch.


def run_stages(args, proof):
    args.output.mkdir(parents=True, exist_ok=False)
    args.owns_output = True
    save(args.output/'plan.json', plan(args))
    save(args.output/'preflight.json', proof)
    all_cards = inventory(args.smi)
    selected = select(all_cards, args.gpus)
    save(args.output/'gpu_selection.json', selected)
    if Path('/proc/sys/kernel/numa_balancing').read_text().strip() != '0':
        raise RuntimeError('kernel.numa_balancing must be 0; this script will not change host settings')
    ips = subprocess.check_output(['hostname', '-I'], text=True).split()
    if args.profile['host_ip'] not in ips:
        raise RuntimeError('Wrong node for this runtime profile')
    wait_for_idle(args, selected, all_cards, proof)
    def status(stage):
        if source_state(args) != proof['sources']:
            raise RuntimeError('Source changed during the test')
        save(args.output/'status.json', dict(state='running', stage=stage, controller_pid=os.getpid()))
    if args.stage != 'on':
        status('capacity_off')
        with service(args, False, args.output/'capacity_off', selected, all_cards) as (_, off):
            off_smoke = smoke(args, proof['prompt_unit'], args.output/'capacity_off')
    status('capacity_on')
    accuracy = None
    with service(args, True, args.output/'capacity_on', selected, all_cards) as (server, on):
        if args.stage == 'on':
            status('smoke_32k_on')
        on_smoke = smoke(args, proof['prompt_unit'], args.output/'capacity_on')
        if args.stage == 'on':
            capacity, smoke_result = on, on_smoke
        else:
            capacity = compare_capacity(off, on)
            smoke_result = compare_smoke(off_smoke, on_smoke)
            save(args.output/'capacity_comparison.json', capacity)
            save(args.output/'smoke_comparison.json', smoke_result)
            if not capacity['increased'] or not smoke_result['all_tokens_equal']:
                raise RuntimeError('Capacity did not increase or fixed-input tokens differ; inspect saved comparison before HumanEval')
        if args.stage != 'capacity':
            status('humaneval_on')
            accuracy = summarize_accuracy(evaluate(args, server, args.output/'capacity_on/humaneval'))
    result = dict(validated_stage=args.stage, capacity=capacity, smoke=smoke_result, accuracy=accuracy)
    status('finalizing')
    result['passed'] = result['accuracy'] is None or result['accuracy']['evaluation_valid']
    save(args.output/'summary.json', result)
    save(args.output/'status.json', dict(state='complete' if result['passed'] else 'failed_acceptance',
        controller_pid=os.getpid(), owned_services_stopped=not bool(args.retained_service),
        retained_service=args.retained_service))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['passed'] else 1


def summarize_accuracy(value):
    expected = {'HumanEval/'+str(i) for i in range(20)}
    if (value['num_tasks'] != 20 or set(value['task_results']) != expected
            or value['http_errors'] or any(type(v) is not bool for v in value['task_results'].values())):
        raise RuntimeError('Incomplete HumanEval-20 coverage or HTTP errors')
    passed = sum(value['task_results'].values())
    if not math.isfinite(value['pass_at_1']) or abs(value['pass_at_1']-passed/20) > 1e-9:
        raise RuntimeError('HumanEval score differs from per-task results')
    return dict(layer_split=True, num_tasks=20, passed=passed,
        pass_at_1=value['pass_at_1'], failed_task_ids=sorted(k for k,v in value['task_results'].items() if not v),
        truncated_task_ids=value['truncated_task_ids'], dataset_sha256=value['dataset_sha256'],
        evaluation_valid=not value['truncated_task_ids'], accuracy_regression_assessed=False,
        quality_threshold=None, scope='HumanEval/0 through HumanEval/19; ON-only subset, not full-suite accuracy')


def plan(args):
    lengths, warmups = smoke_settings(args)
    order = ['capacity_on', 'smoke_32k_on'] if args.stage == 'on' else ['capacity_off', 'capacity_on']
    if args.stage != 'capacity':
        order.append('humaneval_on_same_service')
    return dict(project=str(args.project), profile=args.profile, gpu_ids=args.gpus, stage=args.stage,
        order=order,
        independent_switch='--enable-layer-split', fp4=True, swa_replay=args.replay,
        kv_budget_bytes_per_rank=args.kv_bytes,
        kv_budget_mode='auto' if args.kv_bytes is None else 'fixed',
        gpu_memory_utilization=float(GPU_MEMORY_UTILIZATION),
        kv_budget_overrides_gpu_memory_utilization=args.kv_bytes is not None,
        input_tokens=list(lengths), output_tokens=OUTPUT, warmups_per_length=warmups,
        humaneval=dict(tasks=20, layer_split=True, task_ids='HumanEval/0..19', temperature=0, concurrency=16, max_output_tokens=2048),
        server_command=server_command(args), output=str(args.output),
        no_pd=True, no_dspark=not args.profile.get('dspark', False), no_prefix_cache=True,
        no_eplb=args.expert_map is None,
        keep_service=args.keep_service, wait_idle_interval=args.wait_idle_interval)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true', help='CPU/source/dataset checks; never start a server')
    action.add_argument('--run', action='store_true', help='Explicitly start the sequential GPU test')
    p.add_argument('--project', type=Path, default=HERE.parents[3])
    p.add_argument('--profile', dest='profile_path', type=Path, default=HERE/'shenzhen.json')
    p.add_argument('--gpus', help='Exactly eight physical hy-smi indices, e.g. 0,1,2,3,4,5,6,7')
    p.add_argument('--kv-cache-gib', type=float, help='Override the profile with the same fixed KV budget per GPU for both arms')
    p.add_argument('--expert-map', type=Path,
        help='Enable static EPLB32 and the native FlashMLA, mHC and PCP RoPE settings for ON-only W4A8')
    p.add_argument('--replay', type=int, choices=(0, 1), default=1)
    p.add_argument('--stage', choices=('capacity', 'all', 'on'), default='all',
        help='Paired capacity, paired capacity plus HumanEval, or ON-only 32K plus HumanEval')
    p.add_argument('--keep-service', action='store_true',
        help='Retain the final ON service, or an alive service on failure, for the user')
    p.add_argument('--wait-idle-interval', type=int, default=0,
        help='Before the first launch, wait this many seconds between idle checks; 0 means no wait')
    p.add_argument('--port', type=int, default=6546)
    p.add_argument('--startup-timeout', type=int, default=3600)
    p.add_argument('--eval-timeout', type=int, default=14400)
    p.add_argument('--smi', default='/opt/hyhal/bin/hy-smi')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    args.project, args.profile_path = args.project.resolve(), args.profile_path.resolve()
    args.profile = json.loads(args.profile_path.read_text())
    args.kv_bytes = args.profile.get('kv_cache_bytes')
    if args.kv_cache_gib is not None:
        if not math.isfinite(args.kv_cache_gib) or args.kv_cache_gib <= 0:
            p.error('Invalid fixed KV memory budget')
        args.kv_bytes = int(args.kv_cache_gib*2**30)
    if ((args.kv_bytes is not None and args.kv_bytes <= 0)
            or not 1 <= args.port <= 65535 or min(args.startup_timeout, args.eval_timeout) <= 0):
        p.error('Invalid memory budget, port or timeout')
    args.replay = bool(args.replay)
    if args.expert_map is not None:
        args.expert_map = args.expert_map.resolve()
        if (args.stage != 'on' or not args.replay or args.kv_bytes is not None
                or args.profile.get('model_format') != 'ct-w4a8' or not args.profile.get('dspark')):
            p.error('--expert-map requires ON-only W4A8 with dSpark, SWA replay and automatic KV sizing')
    args.gpus = [int(v) for v in args.gpus.split(',')] if args.gpus else []
    if args.gpus and (len(args.gpus) != 8 or len(set(args.gpus)) != 8 or min(args.gpus) < 0):
        p.error('Exactly eight distinct nonnegative GPU IDs required')
    if args.run and not args.gpus:
        p.error('--run requires an explicit --gpus selection')
    if args.wait_idle_interval < 0:
        p.error('--wait-idle-interval must be nonnegative')
    args.output = (args.output or args.project/'acceptance_runs'/time.strftime('layer_split_%Y%m%d_%H%M%S')).resolve()
    args.owns_output = False
    args.retained_service = None
    return args


def main():
    args = parse_args()
    print(json.dumps(plan(args), indent=2), flush=True)
    if not (args.check or args.run):
        return 0
    proof = preflight(args)
    if args.check:
        print('CPU_CHECK_COMPLETE '+json.dumps(proof), flush=True)
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Controller received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return pipeline(args, proof)
    except BaseException as error:
        if args.owns_output:
            save(args.output/'status.json', dict(state='failed', error=repr(error),
                controller_pid=os.getpid(), retained_service=args.retained_service))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
