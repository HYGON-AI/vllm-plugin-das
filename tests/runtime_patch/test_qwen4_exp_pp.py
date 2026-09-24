# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Qwen4Exp PLE pipeline-parallel compatibility contracts."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_qwen4_exp_pp_allows_ple_layers_owned_by_first_stage() -> None:
    script = textwrap.dedent(
        """
        from types import SimpleNamespace

        from vllm_hcu.patch.platform import apply_platform_patches

        apply_platform_patches()

        from vllm.model_executor.models.config import (
            Qwen4ExpForConditionalGenerationConfig,
        )

        text_config = SimpleNamespace(
            hc_count=4,
            indexer_n_heads=4,
            mamba_ssm_dtype="float32",
            num_hidden_layers=48,
            ple_layer_ids=[2],
        )
        config = SimpleNamespace(
            cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
            model_config=SimpleNamespace(
                hf_text_config=text_config,
                multimodal_config=None,
            ),
            parallel_config=SimpleNamespace(
                enable_dbo=False,
                pipeline_parallel_size=2,
                ubatch_size=1,
            ),
            speculative_config=SimpleNamespace(method="mtp"),
        )

        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)
        assert config.parallel_config.pipeline_parallel_size == 2
        assert text_config.ple_layer_ids == [2]
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_qwen4_exp_pp_rejects_ple_layer_owned_by_later_stage() -> None:
    script = textwrap.dedent(
        """
        from types import SimpleNamespace

        from vllm_hcu.patch.platform import apply_platform_patches

        apply_platform_patches()

        from vllm.model_executor.models.config import (
            Qwen4ExpForConditionalGenerationConfig,
        )

        text_config = SimpleNamespace(
            hc_count=4,
            indexer_n_heads=4,
            mamba_ssm_dtype="float32",
            num_hidden_layers=48,
            ple_layer_ids=[25],
        )
        config = SimpleNamespace(
            cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
            model_config=SimpleNamespace(
                hf_text_config=text_config,
                multimodal_config=None,
            ),
            parallel_config=SimpleNamespace(
                enable_dbo=False,
                pipeline_parallel_size=2,
                ubatch_size=1,
            ),
            speculative_config=SimpleNamespace(method="mtp"),
        )

        try:
            Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)
        except NotImplementedError as exc:
            assert "raw input_ids" in str(exc)
        else:
            raise AssertionError("later-stage PLE unexpectedly accepted")
        assert config.parallel_config.pipeline_parallel_size == 2
        assert text_config.ple_layer_ids == [25]
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_qwen4_exp_first_pipeline_stage_prepares_local_ple_inputs() -> None:
    script = textwrap.dedent(
        """
        from types import SimpleNamespace

        import torch

        from vllm_hcu.patch.worker import apply_worker_patches

        apply_worker_patches()

        import vllm.distributed.parallel_state as parallel_state
        from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
            MambaHybridModelState,
        )

        parallel_state.get_pp_group = lambda: SimpleNamespace(is_first_rank=True)

        def base_init(self, vllm_config, model, encoder_cache, device):
            self.model_config = vllm_config.model_config
            self.max_num_reqs = 2
            self.device = device

        MambaHybridModelState.__init__ = base_init
        text_config = SimpleNamespace(
            eos_token_id=248044,
            ngram_size=3,
            num_hidden_layers=48,
            ple_layer_ids=[2],
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=text_config),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        )

        state = Qwen4ExpModelState(config, object(), None, torch.device("cpu"))

        assert state.uses_ngram_embedding is True
        assert state.ngram_context.shape == (2, 2)
        assert config.parallel_config.pipeline_parallel_size == 2
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_qwen4_exp_later_pipeline_stage_skips_ple_input_state() -> None:
    script = textwrap.dedent(
        """
        from types import SimpleNamespace

        import torch

        from vllm_hcu.patch.worker import apply_worker_patches

        apply_worker_patches()

        import vllm.distributed.parallel_state as parallel_state
        from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
            MambaHybridModelState,
        )

        parallel_state.get_pp_group = lambda: SimpleNamespace(is_first_rank=False)

        def base_init(self, vllm_config, model, encoder_cache, device):
            self.model_config = vllm_config.model_config
            self.max_num_reqs = 2
            self.device = device

        MambaHybridModelState.__init__ = base_init
        text_config = SimpleNamespace(
            eos_token_id=248044,
            ngram_size=3,
            num_hidden_layers=48,
            ple_layer_ids=[2],
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=text_config),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        )

        state = Qwen4ExpModelState(config, object(), None, torch.device("cpu"))

        assert state.uses_ngram_embedding is False
        assert state.ngram_context_len == 0
        assert config.parallel_config.pipeline_parallel_size == 2
        assert text_config.ple_layer_ids == [2]
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_qwen4_exp_model_state_rejects_ple_owned_by_later_stage() -> None:
    script = textwrap.dedent(
        """
        from types import SimpleNamespace

        import torch

        from vllm_hcu.patch.worker import apply_worker_patches

        apply_worker_patches()

        from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
            MambaHybridModelState,
        )

        def base_init(self, vllm_config, model, encoder_cache, device):
            self.model_config = vllm_config.model_config
            self.max_num_reqs = 2
            self.device = device

        MambaHybridModelState.__init__ = base_init
        text_config = SimpleNamespace(
            eos_token_id=248044,
            ngram_size=3,
            num_hidden_layers=48,
            ple_layer_ids=[25],
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=text_config),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        )

        try:
            Qwen4ExpModelState(config, object(), None, torch.device("cpu"))
        except RuntimeError as exc:
            assert "raw input_ids" in str(exc)
        else:
            raise AssertionError("later-stage PLE unexpectedly accepted")
        assert config.parallel_config.pipeline_parallel_size == 2
        assert text_config.ple_layer_ids == [25]
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_qwen4_exp_mtp_last_pp_rank_accepts_speculator_hidden_states() -> None:
    script = textwrap.dedent(
        """
        from types import MethodType, SimpleNamespace

        import torch
        import torch.nn as nn

        from vllm_hcu.patch.worker import apply_worker_patches

        apply_worker_patches()

        import vllm.models.qwen4_exp.amd.mtp as mtp_module
        from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMultiTokenPredictor

        mtp_module.get_pp_group = lambda: SimpleNamespace(
            is_first_rank=False,
            is_last_rank=True,
        )

        class FakeLayer(nn.Module):
            def forward(self, **kwargs):
                hidden_states = kwargs["hidden_states"]
                return hidden_states, None, None

        class FakeMixer(nn.Module):
            def combine_and_mix(self, hidden_states, block_output, injection):
                return hidden_states, hidden_states[:, :3], None

        predictor = Qwen4ExpMultiTokenPredictor.__new__(
            Qwen4ExpMultiTokenPredictor
        )
        nn.Module.__init__(predictor)
        predictor.hc_count = 2
        predictor.hidden_size = 3
        predictor.num_mtp_layers = 1
        predictor.pre_fc_norm_embedding = nn.Identity()
        predictor.fc_embedding = nn.Identity()
        predictor.pre_fc_norm_hidden = nn.Identity()
        predictor.fc_hidden = nn.Identity()
        predictor.layers = nn.ModuleList([FakeLayer()])
        predictor.hyper_connection_mixer = FakeMixer()
        predictor.embed_input_ids = MethodType(
            lambda self, input_ids: torch.ones(input_ids.shape[0], 3),
            predictor,
        )

        hidden_states = torch.arange(12, dtype=torch.float32).view(2, 6)
        sample_hidden, feedback_hidden = predictor.forward(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            hidden_states=hidden_states,
        )

        expected_feedback = hidden_states + 1
        assert sample_hidden.shape == (2, 3)
        assert feedback_hidden.shape == (2, 6)
        torch.testing.assert_close(sample_hidden, expected_feedback[:, :3])
        torch.testing.assert_close(feedback_hidden, expected_feedback)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
