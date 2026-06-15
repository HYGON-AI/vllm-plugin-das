from __future__ import annotations

from vllm.multimodal.processing import PromptReplacement
from vllm.multimodal.parse import VisionChunkProcessorItems

IMAGE_PLACEHOLDER = (
    "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
)
VIDEO_PLACEHOLDER = (
    "<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
)
MEDIA_PAD_PLACEHOLDER = "<|media_pad|>"


def patch_kimi_k25_vision_chunk_prompt_updates() -> None:
    try:
        import vllm.model_executor.models.kimi_k25 as kimi_k25
    except Exception:
        return

    processor_cls = getattr(kimi_k25, "KimiK25MultiModalProcessor", None)
    if processor_cls is None:
        return
    if getattr(processor_cls, "_hcu_kimi_k25_prompt_patch_applied", False):
        return

    dummy_inputs_cls = getattr(kimi_k25, "KimiK25DummyInputsBuilder", None)
    original_get_prompt_updates = processor_cls._get_prompt_updates

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ):
        hf_config = self.info.get_hf_config()
        media_token_id = hf_config.media_placeholder_token_id

        def get_replacement(item_idx: int):
            media = mm_items.get_items(
                "vision_chunk",
                (VisionChunkProcessorItems,),
            )
            num_media_token = self.info.media_tokens_calculator(media[item_idx])
            return [media_token_id] * num_media_token

        updates = list(original_get_prompt_updates(
            self,
            mm_items,
            hf_processor_mm_kwargs,
            out_mm_kwargs,
        ))

        # Keep the original token-id match, but add string fallbacks so the
        # processor can still resolve vision_chunk placeholders if tokenization
        # changes across transformers releases.
        updates.extend(
            [
                PromptReplacement(
                    modality="vision_chunk",
                    target=IMAGE_PLACEHOLDER,
                    replacement=get_replacement,
                ),
                PromptReplacement(
                    modality="vision_chunk",
                    target=VIDEO_PLACEHOLDER,
                    replacement=get_replacement,
                ),
                PromptReplacement(
                    modality="vision_chunk",
                    target=MEDIA_PAD_PLACEHOLDER,
                    replacement=get_replacement,
                ),
            ]
        )
        return updates

    processor_cls._get_prompt_updates = _get_prompt_updates
    processor_cls._hcu_kimi_k25_prompt_patch_applied = True

    if dummy_inputs_cls is not None and not getattr(
        dummy_inputs_cls,
        "_hcu_kimi_k25_dummy_text_patch_applied",
        False,
    ):
        def get_dummy_text(self, mm_counts):
            num_media = mm_counts.get("vision_chunk", 0)
            return IMAGE_PLACEHOLDER * num_media

        dummy_inputs_cls.get_dummy_text = get_dummy_text
        dummy_inputs_cls._hcu_kimi_k25_dummy_text_patch_applied = True
