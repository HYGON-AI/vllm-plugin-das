#!/usr/bin/env python3
"""Build/run a deterministic public-LLM softcap integration control.

Use an isolated installed-wheel Python with -I. This synthetic checkpoint is
not a pretrained-model quality test. Compare TRITON_ATTN and FLASH_ATTN in
separate fresh processes using the same serialized checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def worker_identity(worker):
    import hcu_adapter
    import vllm

    expected = Path(sys.executable).absolute().parent.parent / (
        f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    for module in (vllm, hcu_adapter):
        if not Path(module.__file__).absolute().is_relative_to(expected):
            raise RuntimeError(f"worker imported outside its environment: {module.__file__}")
    layers = []
    for name, module in worker.model_runner.model.named_modules():
        impl = getattr(module, "impl", None)
        cap = getattr(impl, "logits_soft_cap", None)
        if cap:
            layers.append({"name": name, "softcap": cap,
                           "impl": type(impl).__module__ + "." + type(impl).__name__})
    if len(layers) != 2 or any(layer["softcap"] != 0.5 for layer in layers):
        raise RuntimeError(f"unexpected effective softcap layers: {layers}")
    config = worker.vllm_config
    return {"vllm": vllm.__file__, "adapter": hcu_adapter.__file__,
            "layers": layers, "graph": str(config.compilation_config.cudagraph_mode),
            "prefix": config.cache_config.enable_prefix_caching,
            "block_size": config.cache_config.block_size}


def build_checkpoint(checkpoint):
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM

    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.set_num_threads(8)
    torch.manual_seed(812)
    config = Gemma2Config(
        vocab_size=128, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, query_pre_attn_scalar=64, max_position_embeddings=256,
        sliding_window=None, layer_types=["full_attention", "full_attention"],
        attn_logit_softcapping=0.5, final_logit_softcapping=None,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=True,
    )
    model = Gemma2ForCausalLM(config).to(dtype=torch.bfloat16)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight.mul_(20)
            layer.self_attn.k_proj.weight.mul_(20)
    model.save_pretrained(checkpoint)


def run(checkpoint, backend, output_dir):
    import hcu_adapter
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    expected = Path(sys.executable).absolute().parent.parent / (
        f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    for module in (vllm, hcu_adapter):
        if not Path(module.__file__).absolute().is_relative_to(expected):
            raise RuntimeError(f"parent imported outside its environment: {module.__file__}")
    output_dir.mkdir(parents=True, exist_ok=False)
    llm = LLM(
        model=str(checkpoint), skip_tokenizer_init=True, attention_backend=backend,
        dtype="bfloat16", block_size=64, max_model_len=128, max_num_seqs=4,
        max_num_batched_tokens=128, gpu_memory_utilization=0.15, seed=0,
        enable_prefix_caching=True, max_logprobs=128,
    )
    workers = llm.collective_rpc(worker_identity, timeout=60)
    params = SamplingParams(temperature=0, seed=0, max_tokens=8, ignore_eos=True,
                            logprobs=128, detokenize=False)
    prompts = [{"prompt_token_ids": [4, 9, 13, 25, 8, 11, 15, 22] * 9 + [tail]}
               for tail in (26, 27, 28)]
    waves = []
    for _ in range(2):
        outputs = llm.generate(prompts, params, use_tqdm=False)
        waves.append([{
            "prompt_token_ids": result.prompt_token_ids,
            "token_ids": result.outputs[0].token_ids,
            "logprobs": [{str(k): value.logprob for k, value in row.items()}
                         for row in result.outputs[0].logprobs],
            "num_cached_tokens": result.num_cached_tokens,
            "finish_reason": result.outputs[0].finish_reason,
        } for result in outputs])
    if [row["token_ids"] for row in waves[0]] != [row["token_ids"] for row in waves[1]]:
        raise RuntimeError("prefix replay changed greedy outputs")
    if not all((row["num_cached_tokens"] or 0) >= 64 for row in waves[1]):
        raise RuntimeError("prefix replay did not reuse the complete shared page")
    record = {
        "backend": backend, "outputs": waves, "workers": workers,
        "architecture": torch.cuda.get_device_properties(0).gcnArchName,
        "vllm_origin": vllm.__file__, "adapter_origin": hcu_adapter.__file__,
        "torch_version": torch.__version__,
        "plugin_version": (importlib.metadata.version("hcu-vllm-das-plugin")
                           if backend == "FLASH_ATTN" else None),
        "scope": "seeded serialized fixture; not pretrained model accuracy",
    }
    (output_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--backend", choices=["TRITON_ATTN", "FLASH_ATTN"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.build:
        if args.backend or args.output_dir:
            parser.error("--build takes only --checkpoint")
        build_checkpoint(args.checkpoint)
    else:
        if not args.backend or args.output_dir is None:
            parser.error("run requires --backend and --output-dir")
        run(args.checkpoint, args.backend, args.output_dir)


if __name__ == "__main__":
    main()
