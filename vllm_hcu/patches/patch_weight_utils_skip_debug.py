# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator

from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_safetensors_weights_iterator() -> None:
    from vllm.model_executor.model_loader import weight_utils
    from vllm.model_executor.model_loader import default_loader

    if getattr(weight_utils, "_hcu_skip_weight_patch_applied", False):
        return

    original_iterator = weight_utils.safetensors_weights_iterator
    skip_logged = False

    def wrapped_safetensors_weights_iterator(
        hf_weights_files: list[str],
        use_tqdm_on_load: bool,
        safetensors_load_strategy: str = "lazy",
        local_expert_ids: set[int] | None = None,
    ) -> Generator[tuple[str, object], None, None]:
        import vllm_hcu.platforms.envs as henvs

        skip_weight_debug_enabled = henvs.VLLM_HCU_USE_SKIP_WEIGHT_DEBUG
        for name, param in original_iterator(
            hf_weights_files=hf_weights_files,
            use_tqdm_on_load=use_tqdm_on_load,
            safetensors_load_strategy=safetensors_load_strategy,
            local_expert_ids=local_expert_ids,
        ):
            if skip_weight_debug_enabled and "layers." in name:
                try:
                    layer_id = int(name.split("layers.")[1].split(".")[0])
                    if layer_id >= 5:
                        nonlocal skip_logged
                        if not skip_logged:
                            logger.warning(
                                "VLLM_HCU_USE_SKIP_WEIGHT_DEBUG is enabled "
                                "skipping weights from layer %s starting with %s",
                                layer_id,
                                name,
                            )
                            skip_logged = True
                        continue
                except Exception:
                    pass
            yield name, param

    weight_utils.safetensors_weights_iterator = wrapped_safetensors_weights_iterator
    default_loader.safetensors_weights_iterator = wrapped_safetensors_weights_iterator
    weight_utils._hcu_skip_weight_patch_applied = True
