from collections import Counter
import os
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper


class Stage4Worker:
    def stage4_set_mode(self, mode):
        if mode not in ('full_piecewise', 'padded_eager', 'decode_only', 'eager'):
            raise ValueError(mode)
        runner = self.model_runner
        if not hasattr(self, '_stage4_dispatch'):
            self._stage4_dispatch = runner.cudagraph_dispatcher.dispatch
            worker = self
            original_call = BreakableCUDAGraphWrapper.__call__
            def call(wrapper, *args, **kwargs):
                if (worker._stage4_mode == 'padded_eager' and
                    is_forward_context_available() and
                    get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE):
                    worker._stage4_bypass += 1
                    return wrapper.runnable(*args, **kwargs)
                return original_call(wrapper, *args, **kwargs)
            BreakableCUDAGraphWrapper.__call__ = call
            def dispatch(*args, **kwargs):
                allowed = kwargs.get('valid_modes')
                target = ({CUDAGraphMode.NONE} if worker._stage4_mode == 'eager' else
                          {CUDAGraphMode.NONE, CUDAGraphMode.FULL} if worker._stage4_mode == 'decode_only' else None)
                if target is not None:
                    kwargs['valid_modes'] = target if allowed is None else target & allowed
                result = worker._stage4_dispatch(*args, **kwargs)
                worker._stage4_counts[str(result[0])] += 1
                worker._stage4_shapes[str((args, kwargs, result))] += 1
                worker._stage4_real = kwargs.get('num_tokens', args[0] if args else None)
                worker._stage4_runtime = str(result[0])
                return result
            runner.cudagraph_dispatcher.dispatch = dispatch
        self._stage4_mode = mode
        self._stage4_counts = Counter()
        self._stage4_shapes = Counter()
        self._stage4_bypass = 0
        self._stage4_real = None
        self._stage4_runtime = None
        return {'mode': mode, 'runner': type(runner).__name__}

    def stage4_stats(self):
        return {'mode': self._stage4_mode, 'counts': dict(self._stage4_counts),
                'shapes': dict(self._stage4_shapes), 'bypass': self._stage4_bypass}

    def stage4_modules(self, pattern=''):
        return [(n, type(m).__name__) for n, m in self.model_runner.model.named_modules()
                if pattern in n][:1000]

    def stage4_snapshot(self, path, names):
        if getattr(self, '_stage4_hooks', None):
            for h in self._stage4_hooks:
                h.remove()
        self._stage4_hooks = []
        self._stage4_snapshots = {}
        self._stage4_snap_path = path
        if self.rank != 0:
            return {'rank': self.rank, 'hooks': 0}
        def serialize(value):
            if isinstance(value, torch.Tensor):
                t = value.detach()
                if t.ndim >= 2 and self._stage4_real and t.shape[0] >= self._stage4_real:
                    idx = sorted(set([0, self._stage4_real // 2, self._stage4_real - 1]))
                    return {'shape': tuple(t.shape), 'indices': idx, 'value': t[idx].cpu().clone()}
                return {'shape': tuple(t.shape), 'value': t.cpu().clone()}
            if isinstance(value, (tuple, list)):
                return [serialize(x) for x in value]
            if isinstance(value, dict):
                return {k: serialize(v) for k, v in value.items()}
            return str(value)
        def hook(name):
            def collect(module, args, kwargs, output):
                if self._stage4_real and self._stage4_real > 2 :
                    key = name + ':' + str(sum(k.startswith(name + ':') for k in self._stage4_snapshots))
                    self._stage4_snapshots[key] = {'input': serialize(args), 'kwargs': serialize(kwargs),
                                                     'output': serialize(output), 'real_tokens': self._stage4_real,
                                                     'runtime': self._stage4_runtime}
            return collect
        modules = dict(self.model_runner.model.named_modules())
        for name in names:
            self._stage4_hooks.append(modules[name].register_forward_hook(hook(name), with_kwargs=True))
        return {'rank': self.rank, 'hooks': len(self._stage4_hooks)}

    def stage4_flush(self):
        if self.rank == 0 and getattr(self, '_stage4_snap_path', None):
            torch.save(self._stage4_snapshots, self._stage4_snap_path)
            return {'path': self._stage4_snap_path, 'modules': len(self._stage4_snapshots)}
        return {'rank': self.rank}

    def stage4_exec(self, path, params=None):
        namespace = {'worker': self, 'params': params or {}}
        exec(compile(open(path).read(), path, 'exec'), namespace)
        return namespace.get('result')
