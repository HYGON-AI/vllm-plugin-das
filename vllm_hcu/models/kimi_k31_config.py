# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Config identity for Kimi-K3.1 Pro (model_type ``kimi_k31``).

The checkpoint ships remote ``configuration_kimi_k3.py``; with
``--trust-remote-code`` transformers would load that dynamic class, which is
identity-incompatible with vLLM's built-in ``KimiK3Config`` (the Kimi-K3
multimodal preprocessing plugin asserts ``isinstance(hf_config,
KimiK3Config)``).  Registering a thin built-in SUBclass keyed by
``model_type=kimi_k31`` resolves both: ``AutoConfig`` uses our class (no
remote module) and every ``isinstance`` gate passes by inheritance.

The remote ``KimiLinearConfig`` exposes ``is_moe`` / ``is_mla`` /
``is_linear_attn`` as properties and ``is_kda_layer()`` as a method; the
stock Kimi decoder stack reads them off ``hf_text_config``.  A bare
``PretrainedConfig(**dict)`` conversion only materializes stored keys, so
the text config is built as a dedicated subclass mirroring the remote
derived surface exactly.  All stored keys survive via the base-class
setattr behaviour — no allowlist to maintain.

Registration happens from ``HCUPlatform.pre_register_and_update`` (parser
phase, strictly before ``ModelConfig`` builds the HF config); this module is
kept import-side-effect free.
"""

from __future__ import annotations


def register_kimi_k31_config() -> bool:
    """Idempotently teach transformers AutoConfig about ``kimi_k31``.

    Invoked from ``HCUPlatform.pre_register_and_update`` (parser phase,
    strictly before ``ModelConfig`` builds the HF config).  Returns True when
    this call performed the registration.
    """
    # Function-level imports: importing vllm/transformers at plugin-package
    # import time would re-enter a partially initialized ``vllm_hcu`` via
    # vLLM's plugin discovery; keep this module import-side-effect free.
    import logging

    from transformers import AutoConfig
    from transformers import PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config

    logger = logging.getLogger(__name__)
    if "kimi_k31" in CONFIG_MAPPING_NAMES:
        return False

    class KimiK31TextConfig(PretrainedConfig):
        """Full-fidelity text config with the remote derived surface."""

        model_type = "kimi_linear"

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            # json configs store only non-default values, while the Kimi
            # decoder stack reads attributes that the remote __init__ always
            # assigns (defaulted parameters). Backfill any missing stored
            # attribute from the vLLM built-in KimiLinearConfig signature
            # defaults — data-driven, no hand-maintained allowlist.
            import inspect

            from vllm.transformers_utils.configs import kimi_k3 as _k3cfg

            for name, param in inspect.signature(
                _k3cfg.KimiLinearConfig.__init__
            ).parameters.items():
                if name in ("self",) or name.startswith("_"):
                    continue
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if param.default is inspect.Parameter.empty:
                    continue
                if name not in self.__dict__:
                    setattr(self, name, param.default)

        @property
        def is_mla(self):
            return (
                getattr(self, "q_lora_rank", None) is not None
                or getattr(self, "kv_lora_rank", None) is not None
                or getattr(self, "qk_nope_head_dim", None) is not None
                or getattr(self, "qk_rope_head_dim", None) is not None
                or getattr(self, "v_head_dim", None) is not None
                or getattr(self, "mla_use_nope", False) is True
            )

        @property
        def is_moe(self):
            return getattr(self, "num_experts", None) is not None

        @property
        def is_linear_attn(self):
            cfg = getattr(self, "linear_attn_config", None)
            if cfg is None:
                return False
            if isinstance(cfg, dict):
                kda = cfg.get("kda_layers")
            else:
                kda = getattr(cfg, "kda_layers", None)
            return bool(kda)

        def is_kda_layer(self, layer_idx: int) -> bool:
            cfg = getattr(self, "linear_attn_config", None)
            if cfg is None:
                return False
            if isinstance(cfg, dict):
                kda = cfg.get("kda_layers")
            else:
                kda = getattr(cfg, "kda_layers", None)
            return bool(kda) and (layer_idx + 1) in kda

    class KimiK31Config(KimiK3Config):
        """Kimi-K3.1 Pro config: K3-compatible identity, DSA text config."""

        model_type = "kimi_k31"

        def __init__(
            self,
            text_config: dict | PretrainedConfig | None = None,
            vision_config: dict | PretrainedConfig | None = None,
            **kwargs,
        ) -> None:
            if isinstance(text_config, dict):
                # Full-fidelity conversion with the derived attribute surface
                # the stock Kimi decoder stack expects (is_moe/is_mla/
                # is_linear_attn/is_kda_layer).
                text_config = KimiK31TextConfig(**text_config)
            super().__init__(
                text_config=text_config, vision_config=vision_config, **kwargs
            )
            # Mirror the built-in lifting rule: visibility for quant
            # detection and vllm gates. Use getattr: transformers only
            # materializes the attribute when present in the source dict.
            text = self.text_config
            # Lift the DSA knobs to the flat attribute surface the sparse-MLA
            # stack reads (hf_config.index_topk for topk buffer sizing,
            # config.index_{n_heads,head_dim,...} on the Deepseek-V3.2-style
            # indexer path).  The checkpoint nests them in
            # text_config.dsa_config and names the head count index_num_heads;
            # expose both spellings at both config levels.
            dsa = getattr(text, "dsa_config", None)
            if dsa is None:
                dsa = getattr(self, "dsa_config", None)
            if isinstance(dsa, dict):
                dsa_items = dsa.items()
            elif dsa is not None:
                dsa_items = vars(dsa).items()
            else:
                dsa_items = ()
            for owner in (self, text):
                for key, value in dsa_items:
                    if not hasattr(owner, key) and value is not None:
                        setattr(owner, key, value)
                if dsa_items and not hasattr(owner, "index_n_heads"):
                    heads = dict(dsa_items).get("index_num_heads")
                    if heads is not None:
                        setattr(owner, "index_n_heads", heads)

            text = self.text_config
            lifted = getattr(text, "quantization_config", None)
            if lifted is not None and getattr(
                self, "quantization_config", None
            ) is None:
                self.quantization_config = lifted

    try:
        AutoConfig.register("kimi_k31", KimiK31Config)
    except Exception:  # pragma: no cover - transformers version drift
        logger.warning(
            "AutoConfig.register(kimi_k31) failed; falling back to the "
            "checkpoint remote config (Kimi-K3 mm plugin may reject it).",
            exc_info=True,
        )
        return False
    logger.info_once(
        "Registered kimi_k31 -> KimiK31Config (vLLM KimiK3Config subclass)"
    )
    return True
