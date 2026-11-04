#!/usr/bin/env python3
"""Capture a static EPLB proposal using an existing serve and benchmark script.

With --start-after-warmup, the benchmark must run the following after warmup
and before its first measurement request:

    if [ -n "${EPLB_CAPTURE_START_FILE:-}" ]; then
        touch -- "$EPLB_CAPTURE_START_FILE"
    fi

Workers discard warmup load until EP rank zero observes this signal. All EP
ranks agree on the boundary before counting the first measurement pass.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-script', required=True)
    parser.add_argument('--bench-script', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=10018)
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--redundant', type=int, default=16)
    parser.add_argument('--ready-timeout', type=int, default=3600)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument(
        '--start-after-warmup', action='store_true',
        help='Ignore warmup load until the benchmark creates EPLB_CAPTURE_START_FILE',
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error('Output already exists; use a new filename to preserve earlier records')
    if args.window_size < 1:
        parser.error('window-size must be positive')
    start_file = Path(str(output) + '.capture-start')
    if args.start_after_warmup and start_file.exists():
        parser.error('Capture start signal already exists; use a new output filename')
    source = Path(args.run_script).resolve()
    bench = Path(args.bench_script).resolve()
    if not bench.is_file():
        parser.error('Benchmark script not found')
    text = source.read_text()
    model_match = re.search(r'vllm serve\s+(\S+)', text)
    if not model_match:
        parser.error('Cannot identify vllm serve model')
    model = shlex.split(model_match.group(1))[0]
    config = dict(num_redundant_experts=args.redundant, use_async=False,
                  window_size=args.window_size, step_interval=args.window_size,
                  expert_map_record_path=str(output), disable_rearrange=True,
                  communicator='torch_gloo')
    # Replace the existing trailing EPLB block, retaining all model/runtime flags.
    start = text.index('  --eplb-config ')
    end = text.index('  2>&1', start)
    text = text[:start] + '  --eplb-config ' + shlex.quote(json.dumps(config)) + '\n'
    text = re.sub(r'--port\s+\d+', '--port '+str(args.port), text)
    if '--enable-eplb' not in text:
        parser.error('Source must enable EPLB')
    output.parent.mkdir(parents=True, exist_ok=True)
    generated = Path(str(output)+'.capture.sh')
    capture_env = 'unset VLLM_HCU_EPLB_CAPTURE_START_FILE\n'
    if args.start_after_warmup:
        capture_env = (
            'export VLLM_HCU_EPLB_CAPTURE_START_FILE='
            + shlex.quote(str(start_file)) + '\n'
        )
    generated.write_text('set -euo pipefail\n' + capture_env + text)
    subprocess.run(['bash','-n',str(generated)],check=True)
    print('Generated:', generated, flush=True)
    if args.prepare_only:
        return
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',args.port)) == 0:
            parser.error('Port is occupied; stop the old server or select --port')
    env = dict(os.environ, MODEL_PATH=model, BENCH_PORT=str(args.port),
               OUTPUTS_DIR=str(output)+'.bench')
    env.pop('EPLB_CAPTURE_START_FILE', None)
    env.pop('VLLM_HCU_EPLB_CAPTURE_START_FILE', None)
    if args.start_after_warmup:
        env['EPLB_CAPTURE_START_FILE'] = str(start_file)
        print('Waiting for benchmark warmup boundary:', start_file, flush=True)
    with open(str(output)+'.server.log','w') as log:
        proc = subprocess.Popen(['bash',str(generated)],cwd=source.parent,
                                env=env,stdout=log,stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = time.monotonic()+args.ready_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('Serve exited; inspect '+log.name)
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health',timeout=3):
                        break
                except Exception:
                    if time.monotonic()>deadline:
                        raise TimeoutError('Readiness timeout; inspect '+log.name)
                    time.sleep(3)
            # The benchmark signals its explicit warmup boundary via this file.
            # Workers discard warmup counters and begin a fresh window afterwards.
            subprocess.run(['bash',str(bench)],cwd=bench.parent,env=env,check=True)
            if args.start_after_warmup and not start_file.is_file():
                raise RuntimeError(
                    'Benchmark did not signal warmup completion. Add '
                    '\'touch "$EPLB_CAPTURE_START_FILE"\' after warmup.'
                )
            from vllm_hcu.model_executor.layers.fused_moe.static_eplb import load_static_eplb_plan
            cfg = json.loads((Path(model)/'config.json').read_text())
            plan = load_static_eplb_plan(output, model_key='DeepseekV4ForCausalLM',
                expected_model_name=model,expected_shape=(cfg['num_hidden_layers'],
                    cfg['n_routed_experts']+args.redundant),
                num_logical_experts=cfg['n_routed_experts'],
                num_redundant_experts=args.redundant)
            print('Validated real-load proposal:',plan.fingerprint(),flush=True)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()


if __name__ == '__main__':
    main()
