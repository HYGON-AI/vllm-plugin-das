"""Audit actual imports and official Kimi config without model forward/loading."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--nvidia-route", action="store_true")
    args = parser.parse_args()
    result = {"executable": sys.executable, "hostname": os.uname().nodename}
    try:
        import torch
        import vllm
        import hcu_adapter
        from vllm.platforms import current_platform

        result.update(torch=torch.__version__, hip=torch.version.hip,
                      vllm_distribution=metadata.version("vllm"),
                      vllm_module=vllm.__file__, adapter_module=hcu_adapter.__file__,
                      platform=str(type(current_platform)),
                      is_cuda=current_platform.is_cuda(), is_rocm=current_platform.is_rocm(),
                      device=str(torch.cuda.get_device_properties(0)),
                      extensions=[str(x) for x in metadata.entry_points(group="hcu_adapter.extensions")])
        from vllm.model_executor.models.registry import ModelRegistry
        if args.nvidia_route:
            for arch, cls in (("KimiK31ForConditionalGeneration", "KimiK3ForConditionalGeneration"),
                              ("KimiLinearForCausalLM", "KimiLinearForCausalLM")):
                ModelRegistry.register_model(arch, "vllm.models.kimi_k3.nvidia.model:" + cls)
        from vllm.engine.arg_utils import EngineArgs
        from vllm.config import KernelConfig
        # Nested HF overrides replace the sub-config; preserve the checkpoint
        # dimensions/quantization instead of restoring KimiLinear defaults.
        original = json.loads((Path(args.model) / "config.json").read_text())
        text_override = dict(original["text_config"])
        text_override["num_hidden_layers"] = 5
        text_override["min_moe_intermediate_per_partition"] = 192
        config = EngineArgs(
            model=args.model,
            trust_remote_code=True,
            hf_overrides={"text_config": text_override},
            tensor_parallel_size=16,
            load_format="safetensors",
            language_model_only=True,
            max_model_len=4096,
            max_num_seqs=2,
            kernel_config=KernelConfig(enable_flashinfer_autotune=False),
        ).create_engine_config()
        text = config.model_config.hf_text_config
        assert text.hidden_size == original["text_config"]["hidden_size"]
        assert text.num_hidden_layers == 5
        assert text.min_moe_intermediate_per_partition == 192
        result.update(text_config=text.to_dict(),
                      quantization=config.model_config.quantization,
                      kernel_config=str(config.kernel_config))
        cls, arch = ModelRegistry.resolve_model_cls(config.model_config.architectures, config.model_config)
        result.update(model_class=cls.__module__ + "." + cls.__qualname__, architecture=arch, ok=True)
    except BaseException:
        result.update(ok=False, traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
