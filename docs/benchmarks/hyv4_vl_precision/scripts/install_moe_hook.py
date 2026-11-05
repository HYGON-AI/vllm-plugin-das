import torch
worker._stage4_moe_hooks = getattr(worker, '_stage4_moe_hooks', [])
for h in worker._stage4_moe_hooks:
    h.remove()
worker._stage4_moe_hooks = []
worker._stage4_moe_captured = set()
model = worker.model_runner.model
for idx in (0,1):
    layer = model.model.layers[idx]
    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            n = worker._stage4_real
            if n and 2 < n < 128 and layer_idx not in worker._stage4_moe_captured:
                worker._stage4_moe_captured.add(layer_idx)
                value = args[0] if args else kwargs['hidden_states']
                torch.save({'input':value.detach().cpu().clone(), 'output':output.detach().cpu().clone(), 'real_tokens':n},
                           f'/root/hy4-perf-stage4/moe-input-{params["label"]}-layer{layer_idx}-rank{worker.rank}.pt')
        return hook
    worker._stage4_moe_hooks.append(layer.mlp.register_forward_hook(make_hook(idx), with_kwargs=True))
result = {'rank':worker.rank,'hooks':len(worker._stage4_moe_hooks)}
