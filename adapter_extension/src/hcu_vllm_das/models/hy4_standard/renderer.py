# SPDX-License-Identifier: Apache-2.0
"""HY4 VL string rendering while preserving stock HF behavior for other models."""

from dataclasses import replace

from vllm.renderers.hf import HfRenderer


class HY4VLHfRenderer(HfRenderer):
    """Dispatch only default standard-HY4 VL templates to string content."""

    def _hy4_params(self, params):
        config = self.model_config.hf_config
        if (
            getattr(config, "model_type", None) == "hy_v4_vl"
            and getattr(getattr(config, "text_config", None), "model_type", None)
            == "hy_v4"
            and params.chat_template is None
            and params.chat_template_content_format == "auto"
        ):
            return replace(params, chat_template_content_format="string")
        return params

    def render_messages(self, messages, params):
        return super().render_messages(messages, self._hy4_params(params))

    async def render_messages_async(self, messages, params):
        return await super().render_messages_async(messages, self._hy4_params(params))
