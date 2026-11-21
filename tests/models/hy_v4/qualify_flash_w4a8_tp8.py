# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Real TP8 CT-loader/expert qualification; subset weights, not model accuracy."""

from pathlib import Path
import argparse, json, os, time, traceback, hashlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    assert (
        int(os.environ.get("WORLD_SIZE", "0")) == 8
    ), "This qualification requires torchrun TP8"
    R = args.output.resolve()
    R.mkdir(parents=True, exist_ok=True)
    assert not (
        R / f'rank{os.environ["RANK"]}.json'
    ).exists(), "Do not overwrite evidence"
    record = {
        "rank": int(os.environ["RANK"]),
        "finished": False,
        "cleanup_succeeded": False,
        "cases": [],
        "failures": [],
        "scope": "Real TP8, actual layer1 expert0..3 weights, real Core loader. No full model/DFly/HumanEval/image qualification.",
    }
    rank = record["rank"]
    initialized = False
    try:
        if args.release is not None:
            release = json.loads(args.release.read_text())
            assert 0 <= time.time() - release["released_at"] < 1800
            assert release["host_gpu_processes"] == []
            for fn, sha in release["sources"].items():
                assert hashlib.sha256(Path(fn).read_bytes()).hexdigest() == sha, fn
        assert not any(
            k.startswith(("NCCL_", "HIPBLAS_", "ROCBLAS_")) for k in os.environ
        )
        import torch
        import torch.distributed as dist
        from torch import nn
        from safetensors import safe_open
        import vllm
        from vllm.plugins import load_general_plugins

        torch.cuda.set_device(rank)
        torch.set_num_threads(1)
        load_general_plugins()
        from vllm.engine.arg_utils import EngineArgs
        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        from vllm.distributed.parallel_state import set_custom_all_reduce
        from vllm.model_executor.layers.fused_moe import FusedMoEFactory
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )
        from vllm_hcu.models.hy_v4_flash.nvidia.w4a8 import (
            HYV4FlashW4A8Config,
            HYV4FlashChannelW4A8Method,
        )
        from vllm_hcu.models.hy_v4_flash.nvidia.model import (
            HYV4FlashModel,
            validate_runtime,
        )

        model = args.model.resolve()
        raw = json.loads((model / "config.json").read_text())
        text = raw["text_config"]

        def overrides(c):
            c.speculative_config = None
            # vLLM first calls the override with a dummy PreTrainedConfig.
            if hasattr(c, "text_config"):
                c.text_config.flash_w4a8_backend = "adapter_reference"
            return c

        engine_args = EngineArgs(
            model=str(model),
            tensor_parallel_size=8,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=2,
            max_num_batched_tokens=512,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            kv_cache_dtype="fp8_ds_mla",
            attention_backend="FLASHMLA_SPARSE",
            speculative_config={"method": "dfly", "num_speculative_tokens": 5},
            hf_overrides=overrides,
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        cfg = engine_args.create_engine_config()
        validate_runtime(cfg)
        source = CompressedTensorsConfig.from_config(text["quantization_config"])
        source.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        quant = HYV4FlashW4A8Config.for_moe(source)
        assert source.config == text["quantization_config"]
        record.update(
            torch=torch.__version__,
            hip=torch.version.hip,
            vllm=vllm.__version__,
            head_dtype=str(cfg.model_config.head_dtype),
            config_sha256=hashlib.sha256(
                (model / "config.json").read_bytes()
            ).hexdigest(),
        )
        with set_current_vllm_config(cfg), torch.inference_mode():
            set_custom_all_reduce(False)
            init_distributed_environment(8, rank, "env://", rank, "nccl")
            initialize_model_parallel(tensor_model_parallel_size=8)
            initialized = True
            with torch.device("cuda", rank):
                runner = FusedMoEFactory(
                    num_experts=4,
                    top_k=2,
                    hidden_size=1536,
                    intermediate_size=1536,
                    params_dtype=torch.bfloat16,
                    quant_config=quant,
                    prefix="model.layers.1.mlp.experts",
                    swiglu_limit=10.0,
                )
            layer = runner.routed_experts
            assert isinstance(layer.quant_method, HYV4FlashChannelW4A8Method), type(
                layer.quant_method
            )
            record["method"] = type(layer.quant_method).__name__
            # Exercise the actual Flash streaming loader with a four-expert fixture.
            fixture = HYV4FlashModel.__new__(HYV4FlashModel)
            nn.Module.__init__(fixture)
            from types import SimpleNamespace

            fixture.config = SimpleNamespace(num_experts=4, engram_layer_ids=[])
            fixture.loaded_parts = {}
            fixture.engram_hash = None
            fixture.layers = nn.ModuleList([nn.Identity(), nn.Module()])
            fixture.layers[1].mlp = nn.Module()
            fixture.layers[1].mlp.experts = runner
            index = json.loads((model / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            raw_values = {}
            for expert in range(4):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    for leaf in ("weight_packed", "weight_scale", "weight_shape"):
                        key = f"model.layers.1.mlp.experts.{expert}.{projection}.{leaf}"
                        with safe_open(
                            str(model / index[key]), framework="pt", device="cpu"
                        ) as f:
                            value = f.get_tensor(key)
                        raw_values[(expert, projection, leaf)] = value
                        fixture.load_weights([(key.removeprefix("model."), value)])
            fixture.verify_loaded_parts()
            layer.quant_method.process_weights_after_loading(layer)
            record["loaded_parameters"] = sorted(fixture.loaded_parts)
            record["packed_storage_aliases"] = (
                layer.quant_method.w13.data_ptr() == layer.w13_weight_packed.data_ptr()
            )
            assert record["packed_storage_aliases"]

            # Independent raw-checkpoint unpack and TP shard verification, including
            # down projection's PACKED input-axis slice and channel scales replication.
            expected13, expected2, scales13, scales2 = [], [], [], []
            for expert in range(4):
                gate, up, down = [
                    raw_values[expert, p, "weight_packed"]
                    for p in ("gate_proj", "up_proj", "down_proj")
                ]
                sg, su, sd = [
                    raw_values[expert, p, "weight_scale"]
                    for p in ("gate_proj", "up_proj", "down_proj")
                ]
                expected13.append(
                    torch.cat(
                        [
                            gate[rank * 192 : (rank + 1) * 192],
                            up[rank * 192 : (rank + 1) * 192],
                        ]
                    )
                )
                expected2.append(down[:, rank * 24 : (rank + 1) * 24])
                scales13.append(
                    torch.cat(
                        [
                            sg[rank * 192 : (rank + 1) * 192],
                            su[rank * 192 : (rank + 1) * 192],
                        ]
                    )
                )
                scales2.append(sd)
            for name, expected in [
                ("w13_weight_packed", expected13),
                ("w2_weight_packed", expected2),
                ("w13_weight_scale", scales13),
                ("w2_weight_scale", scales2),
            ]:
                assert torch.equal(
                    getattr(layer, name).cpu(), torch.stack(expected)
                ), name
            record["real_core_loader_bitwise"] = True

            def unpack(words):
                shifts = torch.arange(8, device=words.device, dtype=torch.int64) * 4
                return (
                    ((words.long().unsqueeze(-1) >> shifts) & 15).flatten(-2) - 8
                ).float()

            ww13 = unpack(layer.w13_weight_packed)
            ww2 = unpack(layer.w2_weight_packed)

            def quantize(value):
                scale = (value.abs().amax(-1, keepdim=True).float() / 127.5).to(
                    value.dtype
                )
                scale = torch.where(scale == 0, torch.finfo(value.dtype).eps, scale)
                code = (
                    (value.float() / scale.float())
                    .to(value.dtype)
                    .float()
                    .clamp(-128, 127)
                    .round()
                )
                return code, scale.float()

            for rows, layout in [
                (m, l)
                for m in (1, 2, 7, 33, 128, 512)
                for l in ("contiguous", "features")
            ]:
                torch.manual_seed(1700 + rows)
                x = torch.randn(rows, 1536, device="cuda", dtype=torch.bfloat16)
                x[0].zero_()
                if rows > 2:
                    x[1].mul_(0.03125)
                dist.broadcast(x, src=0)
                if layout == "features":
                    x = torch.stack((x, x), dim=-1)[..., 0]
                    assert x.stride(-1) == 2
                ids = torch.tensor(
                    [[3, 0], [1, 2], [0, 1], [2, 3]], device="cuda", dtype=torch.int32
                ).repeat((rows + 3) // 4, 1)[:rows]

                weight = torch.tensor(
                    [[0.3, 0.7]], device="cuda", dtype=torch.float32
                ).repeat(rows, 1)
                actual = layer.forward_modular(x, weight, ids)
                outputs = []
                for e in range(4):
                    q, s = quantize(x)
                    first = (
                        (q @ ww13[e].T) * s * layer.w13_weight_scale[e, :, 0].float()
                    ).to(x.dtype)
                    g, u = first.chunk(2, -1)
                    g = g.clamp(max=10.0)
                    u = u.clamp(-10.0, 10.0)
                    activated = (torch.nn.functional.silu(g.float()) * u.float()).to(
                        x.dtype
                    )
                    q, s = quantize(activated)
                    second = (
                        (q @ ww2[e].T) * s * layer.w2_weight_scale[e, :, 0].float()
                    ).to(x.dtype)
                    outputs.append(second)
                token = torch.arange(rows, device="cuda")[:, None]
                slot_output = torch.stack(outputs)[ids.long(), token]
                expected = (slot_output.float() * weight[:, :, None]).sum(1).to(x.dtype)
                exact = torch.equal(actual, expected)
                max_abs = (actual.float() - expected.float()).abs().max().item()
                assert exact, (rows, max_abs)
                dist.all_reduce(actual)
                dist.all_reduce(expected)
                assert torch.equal(actual, expected)
                record["cases"].append(
                    {
                        "rows": rows,
                        "layout": layout,
                        "bitwise_equal": True,
                        "max_abs": max_abs,
                        "after_tp_sum_equal": True,
                    }
                )
            torch.cuda.synchronize()
            record["finished"] = True
    except BaseException as e:
        record["failures"].append(type(e).__name__ + ": " + str(e))
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        try:
            if initialized:
                destroy_model_parallel()
                destroy_distributed_environment()
            record["cleanup_succeeded"] = True
        finally:
            (R / f"rank{rank}.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
