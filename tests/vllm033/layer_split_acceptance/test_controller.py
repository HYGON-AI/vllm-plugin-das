"""No servers/devices: refusal, comparison, scoring and stage-order regressions."""
import contextlib
import copy
import importlib.util
import json
import socket
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as controller
import gpu_guard
from evaluate_humaneval import sample_results
from preflight import check_entries, check_model


def startup(split=False, replay=True, count=1000):
    text = f'Initial free memory 100 GiB, reserved 24.00 GiB memory for KV Cache\nGPU KV cache size: {count:,} tokens\nUsing MXFP4 indexer cache\n'
    if replay:
        text += 'Decoder SWA bounded replay: in eager prefill steps\n'
    if split:
        for rank in range(8):
            text += f'DSA layer split PCP rank {rank}/8 owns KV sources []\n'
            text += f'DSA layer split PCP rank {rank}/8 owns SWA layers []\n'
            text += f'DSA independent KV pools rank {rank}: pages=[]\n'
    return text


def score():
    tasks = {'HumanEval/'+str(i): True for i in range(32)}
    return dict(num_tasks=32, task_results=tasks, pass_at_1=1.0, dataset_sha256='dataset',
                http_errors=0, truncated_task_ids=[], completion_sha256={k: 'same' for k in tasks})


class ControllerTests(unittest.TestCase):
    def test_dspark_profile_uses_the_original_thirty_two_task_cohort(self):
        args = controller.parse_args(['--profile', str(HERE/'nmz18_w4a8_dspark.json')])
        command = controller.server_command(args)
        spec = json.loads(command[command.index('--speculative-config')+1])
        self.assertEqual(spec, dict(method='dspark', num_speculative_tokens=5, moe_backend='deep_gemm'))
        self.assertFalse(controller.plan(args)['no_dspark'])
        self.assertEqual(controller.plan(args)['humaneval']['tasks'], 32)
        self.assertEqual(controller.plan(args)['humaneval']['task_ids'], 'HumanEval/0..31')
        self.assertIsNone(args.kv_bytes)
        self.assertEqual(controller.plan(args)['kv_budget_mode'], 'auto')
        self.assertFalse(controller.plan(args)['kv_budget_overrides_gpu_memory_utilization'])
        self.assertNotIn('--kv-cache-memory-bytes', command)
        self.assertEqual(command[command.index('--gpu-memory-utilization')+1], '0.9')

        fixed = controller.parse_args(['--profile', str(HERE/'nmz18_w4a8_dspark.json'),
                                       '--kv-cache-gib', '8'])
        command = controller.server_command(fixed)
        self.assertEqual(command[command.index('--kv-cache-memory-bytes')+1], str(8*2**30))

    def test_all_on_plan_preserves_automatic_kv_and_explicit_native_providers(self):
        argv = ['--profile', str(HERE/'nmz18_w4a8_dspark.json'), '--expert-map', 'map.json', '--stage', 'on']
        args = controller.parse_args(argv)
        plan = controller.plan(args)
        command = plan['server_command']
        added = json.loads(command[command.index('--additional-config')+1])
        self.assertEqual(added['hcu_static_eplb']['num_redundant_experts'], 32)
        self.assertEqual(added['hcu_static_eplb']['static_dispatch_policy'], 'locality_fair')
        profiler = json.loads(command[command.index('--profiler-config')+1])
        self.assertEqual(profiler['profiler'], 'torch')
        self.assertTrue(profiler['torch_profiler_with_stack'])
        self.assertEqual(Path(profiler['torch_profiler_dir']), args.output/'capacity_on'/'profiles')
        self.assertFalse(plan['no_eplb'])
        self.assertNotIn('--enable-eplb', command)
        self.assertNotIn('--kv-cache-memory-bytes', command)
        self.assertIn('--swa-bounded-replay', command)
        self.assertIn('--speculative-config', command)
        env = controller.environment(args, True)
        self.assertIn('--enable-layer-split', command)
        self.assertNotIn('VLLM_HCU_DSV41_LAYER_SPLIT', env)
        self.assertEqual(env['HCU_VLLM_FLASHMLA_PREFILL'], 'native')
        self.assertEqual(env['HCU_VLLM_FLASHMLA_DECODE'], 'native')
        self.assertEqual(env['AITER_MHC_POST_KERNEL'], 'v3')
        self.assertNotIn('HCU_CONFIG', env)
        self.assertNotIn('HCU_VLLM_PCP_ROPE_IMPL', env)
        with self.assertRaises(SystemExit):
            controller.parse_args(argv + ['--kv-cache-gib', '8'])

    def test_port_probe_refuses_a_live_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            with self.assertRaises(OSError):
                controller.check_port_available(listener.getsockname()[1])

    def test_port_probe_allows_restarting_after_a_closed_connection(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
            listener.listen()
            with socket.create_connection(('127.0.0.1', port), timeout=3) as client:
                connection, _ = listener.accept()
                connection.close()
                self.assertEqual(client.recv(1), b'')
        controller.check_port_available(port)
    def test_w4a8_profile_keeps_capacity_protocol_and_requires_quantization(self):
        args = controller.parse_args(['--profile', str(HERE/'nmz18_w4a8.json')])
        command = controller.server_command(args)
        self.assertEqual(command[command.index('--quantization')+1], 'compressed-tensors')
        off, on = controller.environment(args, False), controller.environment(args, True)
        self.assertEqual(off, on)
        self.assertEqual([p for p in controller.server_command(args) if p != '--enable-layer-split'],
                         controller.server_command(args, split=False))
        self.assertIn('das_hcu_ct_w4a8', on['HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS'])
        self.assertNotIn('das_hcu_aiter_int8', on['HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS'])
        self.assertEqual(on['HCU_VLLM_CT_W4A8_IMPL'], 'native')
        self.assertEqual(args.kv_bytes, 12*2**30)
        self.assertIn('--swa-bounded-replay', command)

    def test_checkpoint_format_cannot_be_silently_swapped(self):
        group = dict(format='pack-quantized',
            weights=dict(num_bits=4, type='int', strategy='channel', symmetric=True, dynamic=False, group_size=-1),
            input_activations=dict(num_bits=8, type='int', strategy='token', symmetric=True, dynamic=True))
        raw = dict(quantization_config=dict(quant_method='compressed-tensors', config_groups={
            'group_int4_routed_experts': group, 'group_int4_mtp_experts': group}))
        self.assertIs(check_model(raw, 'ct-w4a8'), raw)
        with self.assertRaises(RuntimeError):
            check_model(raw, 'channel-fp8')
        for field, value in [('num_bits', 8), ('type', 'float'), ('group_size', 128)]:
            invalid = copy.deepcopy(raw)
            invalid['quantization_config']['config_groups']['group_int4_routed_experts']['weights'][field] = value
            with self.assertRaises(RuntimeError):
                check_model(invalid, 'ct-w4a8')
        with self.assertRaises(RuntimeError):
            check_entries('hcu_adapter.extensions', {'das_hcu_ct_w4a8': 'ct:spec'}, [], {'das_hcu_ct_w4a8'})

    def test_entrypoints_require_pcp_and_keep_unrelated_metadata_visible(self):
        declared = {'das_hcu_v41_pcp': 'pcp:spec', 'minimax31_pcp_ep': 'mini:spec'}
        self.assertEqual(check_entries('hcu_adapter.extensions', declared, [('das_hcu_v41_pcp', 'pcp:spec')]),
                         ['hcu_adapter.extensions/minimax31_pcp_ep'])
        for actual in ([], [('das_hcu_v41_pcp', 'old:spec')],
                       [('das_hcu_v41_pcp', 'pcp:spec')]*2):
            with self.assertRaises(RuntimeError):
                check_entries('hcu_adapter.extensions', declared, actual)

    def test_plan_is_safe_and_only_split_changes_cli(self):
        args = controller.parse_args([])
        self.assertFalse(args.run)
        self.assertTrue(args.replay)
        self.assertEqual(args.project, HERE.parents[3])
        off, on = controller.environment(args, False), controller.environment(args, True)
        self.assertEqual(off, on)
        self.assertEqual([p for p in controller.server_command(args) if p != '--enable-layer-split'],
                         controller.server_command(args, split=False))
        command = controller.server_command(args)
        self.assertEqual(command[command.index('--kv-cache-memory-bytes')+1], str(24*2**30))
        self.assertNotIn('--speculative-config', command)
        self.assertNotIn('--kv-transfer-config', command)
        evaluator = controller.evaluator_command(args, args.output)
        self.assertEqual(evaluator[evaluator.index('--limit')+1], '32')
        self.assertNotIn('humaneval_off', controller.plan(args)['order'])

    def test_run_requires_explicit_eight_cards(self):
        for arguments in (['--run'], ['--gpus', '0,1'], ['--gpus', '0,0,2,3,4,5,6,7']):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                controller.parse_args(arguments)

    def test_actual_capacity_same_budget(self):
        off = controller.parse_capacity(startup(), False, True, 24*2**30)
        on = controller.parse_capacity(startup(True, count=4000), True, True, 24*2**30)
        self.assertEqual(controller.compare_capacity(off, on)['ratio'], 4)
        on['kv_budget_bytes_per_rank'] += 1
        with self.assertRaises(RuntimeError):
            controller.compare_capacity(off, on)

    def test_profiled_capacity_reports_each_budget_without_claiming_equal_bytes(self):
        fixed_line = 'Initial free memory 100 GiB, reserved 24.00 GiB memory for KV Cache'
        off_log = startup().replace(fixed_line, 'Available KV cache memory: 3.89 GiB')
        on_log = startup(True, count=4000).replace(fixed_line, 'Available KV cache memory: 4.10 GiB')
        off = controller.parse_capacity(off_log, False, True, None)
        on = controller.parse_capacity(on_log, True, True, None)
        result = controller.compare_capacity(off, on)
        self.assertEqual(result['ratio'], 4)
        self.assertIsNone(result['same_kv_budget_bytes_per_rank'])
        self.assertEqual(result['off_kv_budget_log_gib'], [3.89])
        self.assertEqual(result['on_kv_budget_log_gib'], [4.1])
        for line in (fixed_line, '', 'Available KV cache memory: -0.12 GiB'):
            with self.subTest(line=line), self.assertRaises(RuntimeError):
                controller.parse_capacity(off_log.replace('Available KV cache memory: 3.89 GiB', line),
                                          False, True, None)

    def test_never_infer_missing_evidence(self):
        for text, split in [(startup().replace('GPU KV cache size:', 'theory capacity:'), False),
                            (startup(True).replace('rank 7/8', 'rank 6/8'), True),
                            (startup(True), False),
                            (startup().replace('24.00', '25.00'), False),
                            (startup().replace('Using MXFP4', 'Using FP8'), False)]:
            with self.subTest(text=text[-100:]), self.assertRaises(RuntimeError):
                controller.parse_capacity(text, split, True, 24*2**30)

    def test_replay_must_match(self):
        with self.assertRaises(RuntimeError):
            controller.parse_capacity(startup(replay=False), False, True, 24*2**30)
        self.assertFalse(controller.parse_capacity(startup(replay=False), False, False, 24*2**30)['replay'])

    def test_non_gain_not_a_pass(self):
        off = controller.parse_capacity(startup(), False, True, 24*2**30)
        self.assertFalse(controller.compare_capacity(off, off)['increased'])

    def test_fixed_prompt_complete_ids_and_finite_logprobs(self):
        response = dict(usage=dict(prompt_tokens=128, completion_tokens=16),
                        choices=[dict(token_ids=list(range(16)), logprobs=dict(token_logprobs=[-1.0]*16),
                                      finish_reason='length', text='hello')])
        controller.validate_completion(response, 128)
        for mutate in (lambda r: r['choices'][0]['token_ids'].pop(),
                       lambda r: r['choices'][0]['logprobs']['token_logprobs'].__setitem__(0, float('nan')),
                       lambda r: r['usage'].__setitem__('prompt_tokens', 127)):
            bad = copy.deepcopy(response)
            mutate(bad)
            with self.assertRaises(RuntimeError):
                controller.validate_completion(bad, 128)

    def test_on_only_subset_reports_score_and_failures_without_regression_claim(self):
        value = score()
        value['task_results']['HumanEval/1'] = False
        value['pass_at_1'] = 31/32
        result = controller.summarize_accuracy(value)
        self.assertEqual(result['passed'], 31)
        self.assertEqual(result['failed_task_ids'], ['HumanEval/1'])
        self.assertFalse(result['accuracy_regression_assessed'])
        self.assertTrue(result['evaluation_valid'])

    def test_wrong_coverage_and_inconsistent_scores_refused(self):
        for mutate in (lambda r: r.__setitem__('pass_at_1', .99),
                       lambda r: r['task_results'].pop('HumanEval/31')):
            value = score()
            mutate(value)
            with self.assertRaises(RuntimeError):
                controller.summarize_accuracy(value)

    def test_truncation_is_recorded_and_not_accepted(self):
        value = score()
        value['truncated_task_ids'] = ['HumanEval/3']
        self.assertFalse(controller.summarize_accuracy(value)['evaluation_valid'])

    def test_zero_utilization_is_not_idle_with_memory_or_clients(self):
        clean = dict(cards=[dict(used_bytes=2*2**20, busy_percent=0, total_bytes=144*2**30)]*8, clients=[])
        gpu_guard.require_idle(clean)
        for mutate in (lambda r: r['cards'][0].__setitem__('used_bytes', 1024*2**20),
                       lambda r: r['clients'].append(dict(pid=123)),
                       lambda r: r['cards'].pop()):
            state = copy.deepcopy(clean)
            mutate(state)
            with self.assertRaises(RuntimeError):
                gpu_guard.require_idle(state)

    def test_nmz18_empty_kfd_contexts_require_complete_idle_evidence(self):
        state = dict(cards=[dict(used_bytes=2383872, busy_percent=0, total_bytes=96*2**30)]*8,
                     clients=[dict(pid=123, allocated_pci=[], context_only=True)])
        gpu_guard.require_idle(state, 2.05078125, context_idle_mib=3)
        for change in (lambda s: s['clients'][0].pop('context_only'),
                       lambda s: s['clients'][0].update(context_only=False),
                       lambda s: s['clients'][0]['allocated_pci'].append('0000:01:00.0'),
                       lambda s: s['cards'][0].update(used_bytes=4*2**20),
                       lambda s: s['cards'][0].update(busy_percent=1)):
            modified = copy.deepcopy(state)
            change(modified)
            with self.assertRaises(RuntimeError):
                gpu_guard.require_idle(modified, 2.05078125, context_idle_mib=3)

    def test_review_and_prediction_coverage_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'reviews').mkdir()
            (root/'predictions').mkdir()
            reviews, predictions = [], []
            for i in range(2):
                task = f'HumanEval/{i}'
                reviews.append(dict(sample_score=dict(sample_metadata=dict(task_id=task),
                    score=dict(value=dict(acc=i == 0), status='success'))))
                predictions.append(dict(metadata=dict(task_id=task), model_output=dict(error=None,
                    choices=[dict(stop_reason='length' if i else 'stop', message=dict(content='pass'))])))
            (root/'reviews/humaneval_test.jsonl').write_text('\n'.join(map(json.dumps, reviews)))
            target = root/'predictions/humaneval_test.jsonl'
            target.write_text('\n'.join(map(json.dumps, predictions)))
            result = sample_results(root, 2)
            self.assertEqual(sum(result['task_results'].values()), 1)
            self.assertEqual(result['truncated_task_ids'], ['HumanEval/1'])
            predictions[1]['model_output']['error'] = 'HTTP 500'
            target.write_text('\n'.join(map(json.dumps, predictions)))
            with self.assertRaises(RuntimeError):
                sample_results(root, 2)

    def exercise_pipeline(self, gain=True, stage='all'):
        with tempfile.TemporaryDirectory() as tmp:
            args = controller.parse_args(['--project', tmp, '--output', str(Path(tmp)/'run'),
                                           '--gpus', '0,1,2,3,4,5,6,7', '--stage', stage])
            events = []
            requests = []
            @contextlib.contextmanager
            def service(args, split, output, *unused):
                events.append('start_on' if split else 'start_off')
                output.mkdir()
                try:
                    cap = controller.parse_capacity(startup(split, count=2000 if split and gain else 1000),
                                                    split, True, args.kv_bytes)
                    yield NS(poll=lambda: None), cap
                finally:
                    events.append('stop_on' if split else 'stop_off')
            def evaluate(args, server, output):
                events.append('eval_on' if 'capacity_on' in str(output) else 'eval_off')
                return score()
            def http(args, endpoint, payload):
                self.assertEqual(endpoint, '/v1/completions')
                length = len(payload['prompt'])
                requests.append(length)
                if stage == 'on':
                    events.append('smoke_on')
                return dict(usage=dict(prompt_tokens=length, completion_tokens=16),
                    choices=[dict(token_ids=list(range(16)), logprobs=dict(token_logprobs=[-1.0]*16),
                                  finish_reason='length', text='hello')])
            native_read = Path.read_text
            def read(path, *a, **kw):
                return '0' if str(path).replace('\\', '/') == '/proc/sys/kernel/numa_balancing' else native_read(path, *a, **kw)
            fcntl = NS(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2)
            with patch.dict(sys.modules, {'fcntl': fcntl}), patch.object(controller, 'inventory', return_value=[]), \
                 patch.object(controller, 'select', return_value=[]), patch.object(controller, 'source_state', return_value={}), \
                 patch.object(controller, 'service', service), patch.object(controller, 'http', http), \
                 patch.object(controller, 'evaluate', evaluate), patch.object(Path, 'read_text', read), \
                 patch.object(controller.subprocess, 'check_output', return_value=args.profile['host_ip']):
                if gain:
                    self.assertEqual(controller.pipeline(args, dict(sources={}, prompt_unit=[1])), 0)
                else:
                    with self.assertRaises(RuntimeError):
                        controller.pipeline(args, dict(sources={}, prompt_unit=[1]))
            if stage == 'on':
                self.assertEqual(requests, [32768])
                run_plan = json.loads((args.output/'plan.json').read_text())
                self.assertEqual(run_plan['input_tokens'], [32768])
                self.assertEqual(run_plan['warmups_per_length'], 0)
                self.assertNotIn('capacity_off', run_plan['order'])
                summary = json.loads((args.output/'summary.json').read_text())
                self.assertEqual(summary['capacity']['tokens'], 2000)
                self.assertEqual(summary['accuracy']['num_tasks'], 32)
                self.assertFalse((args.output/'capacity_off').exists())
                self.assertFalse((args.output/'capacity_comparison.json').exists())
                self.assertFalse((args.output/'smoke_comparison.json').exists())
            return events

    def test_stage_order_only_two_starts_and_one_evaluation(self):
        self.assertEqual(self.exercise_pipeline(), ['start_off', 'stop_off', 'start_on', 'eval_on',
                                                    'stop_on'])

    def test_capacity_failure_cleans_up_without_evaluation(self):
        self.assertEqual(self.exercise_pipeline(False), ['start_off', 'stop_off', 'start_on', 'stop_on'])

    def test_on_stage_sends_one_32k_request_then_evaluates_on_same_service(self):
        self.assertEqual(self.exercise_pipeline(stage='on'), ['start_on', 'smoke_on', 'eval_on', 'stop_on'])

    def test_existing_run_is_not_owned_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'existing'
            target.mkdir()
            (target/'status.json').write_text('old evidence')
            args = controller.parse_args(['--project', tmp, '--output', str(target)])
            with self.assertRaises(FileExistsError):
                controller.run_stages(args, {})
            self.assertFalse(args.owns_output)
            self.assertEqual((target/'status.json').read_text(), 'old evidence')

    def test_keep_service_retains_final_on_and_failed_off_but_transitions_successful_off(self):
        for split, fail in ((True, False), (True, True), (False, False), (False, True)):
            with self.subTest(split=split, fail=fail), tempfile.TemporaryDirectory() as tmp:
                args = controller.parse_args(['--project', tmp, '--keep-service'])
                child = NS(pid=987654321, poll=lambda: None, returncode=None)
                state = dict(cards=[dict(used_bytes=0, busy_percent=0, total_bytes=144*2**30)]*8, clients=[])
                with patch.object(controller, 'idle', return_value=state), \
                     patch.object(controller, 'snapshot', return_value=state), \
                     patch.object(controller.socket, 'socket'), patch.object(controller.time, 'sleep'), \
                     patch.object(controller.subprocess, 'Popen', return_value=child), \
                     patch.object(controller, 'http', return_value={'data':[{'id':controller.MODEL_NAME}]}), \
                     patch.object(controller, 'parse_capacity', return_value={}), \
                     patch.object(controller, 'stop') as stopped:
                    def execute():
                        with controller.service(args, split, Path(tmp)/'stage', [], []):
                            if fail:
                                raise ValueError('validation failed')
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'validation failed'):
                            execute()
                    else:
                        execute()
                    if split or fail:
                        stopped.assert_not_called()
                        self.assertEqual(args.retained_service['pid'], child.pid)
                        self.assertTrue((Path(tmp)/'stage/retained_service.json').is_file())
                        self.assertFalse((Path(tmp)/'stage/exit.json').exists())
                    else:
                        stopped.assert_called_once_with(child)
                        self.assertIsNone(args.retained_service)

    def test_idle_wait_refuses_memory_and_real_clients_until_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = controller.parse_args(['--output', tmp, '--wait-idle-interval', '60'])
            cards = [dict(used_bytes=2*2**20, busy_percent=0, total_bytes=144*2**30)]*8
            busy = dict(cards=cards, clients=[dict(pid=12345)])
            clear = dict(cards=cards, clients=[])
            native_read = Path.read_text
            def read(path, *a, **kw):
                return '0' if str(path).replace('\\', '/') == '/proc/sys/kernel/numa_balancing' else native_read(path, *a, **kw)
            with patch.object(controller, 'source_state', return_value={}) as sources, \
                 patch.object(controller, 'inventory', return_value=[]), \
                 patch.object(controller, 'select', return_value=[]), \
                 patch.object(controller, 'snapshot', side_effect=[busy, clear]), \
                 patch.object(Path, 'read_text', read), \
                 patch.object(controller.time, 'sleep') as sleep, \
                 patch.object(controller, 'stop') as stop:
                controller.wait_for_idle(args, [], [], {'sources':{}})
                sleep.assert_called_once_with(60)
                sources.assert_called_once()
                stop.assert_not_called()
                self.assertEqual(json.loads((Path(tmp)/'waiting_gpu.json').read_text())['clients'], [])

    def test_idle_wait_retries_git_timeout_without_admitting_unverified_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = controller.parse_args(['--output', tmp, '--wait-idle-interval', '60'])
            clear = dict(cards=[dict(used_bytes=0, busy_percent=0, total_bytes=1)]*8, clients=[])
            failure = controller.subprocess.TimeoutExpired(['git', 'diff'], 60)
            with patch.object(controller, 'source_state', side_effect=[failure, {'changed':True}]), \
                 patch.object(controller, 'inventory', return_value=[]), \
                 patch.object(controller, 'select', return_value=[]), \
                 patch.object(controller, 'snapshot', return_value=clear), \
                 patch.object(Path, 'read_text', return_value='0'), \
                 patch.object(controller.time, 'sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                    controller.wait_for_idle(args, [], [], {'sources':{}})
                sleep.assert_called_once_with(60)

    def test_keep_service_does_not_change_default_cleanup(self):
        args = controller.parse_args([])
        self.assertFalse(args.keep_service)
        self.assertEqual(args.wait_idle_interval, 0)

    def test_early_server_exit_only_cleans_its_own_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = controller.parse_args(['--project', tmp])
            child = NS(pid=987654321, poll=lambda: 1, returncode=1)
            state = dict(cards=[dict(used_bytes=0, busy_percent=0, total_bytes=144*2**30)]*8, clients=[])
            with patch.object(controller, 'idle', return_value=state), \
                 patch.object(controller, 'snapshot', return_value=state), \
                 patch.object(controller.socket, 'socket'), patch.object(controller.time, 'sleep'), \
                 patch.object(controller.subprocess, 'Popen', return_value=child), \
                 patch.object(controller, 'stop') as stopped:
                with self.assertRaisesRegex(RuntimeError, 'exited during startup'):
                    with controller.service(args, True, Path(tmp)/'stage', [], []):
                        self.fail('Unhealthy server reached request stage')
                stopped.assert_called_once_with(child)
            self.assertEqual(json.loads((Path(tmp)/'stage/exit.json').read_text())['returncode'], 1)


if __name__ == '__main__':
    unittest.main()
