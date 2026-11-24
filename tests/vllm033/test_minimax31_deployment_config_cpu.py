# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real CLI artifact transport, default path binding and fail-closed identity."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hcu_vllm_das.parallel.minimax31_config import artifacts, decode_options
from hcu_vllm_das.parallel import minimax31_pcp as pcp
from hcu_vllm_das.parallel import minimax31_static_ipc as ipc
from hcu_vllm_das.parallel import minimax31_msa as msa


def test_public_engine_cli_transports_artifacts_without_environment(tmp_path, monkeypatch):
    import huggingface_hub
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    # Exercise the real offline CLI path without resolving a placeholder repo.
    monkeypatch.setattr(huggingface_hub.constants, 'HF_HUB_OFFLINE', True)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = EngineArgs.from_cli_args(parser.parse_args([
        '--model', str(checkpoint), '--additional-config',
        json.dumps({
            'minimax31': {'msa_extension_dir': '/artifacts/msa',
                         'loading_receipts': '/outputs/loading'},
            'minimax31_decode': {'shared_expert_schedule': 'stock'},
        })]))
    assert args.model == str(checkpoint)
    settings = artifacts(args)
    assert decode_options(args).shared_expert_schedule == "stock"
    assert settings.msa_extension_dir == '/artifacts/msa'
    assert settings.loading_receipts == '/outputs/loading'
    with pytest.raises(ValueError, match='ready_manifest'):
        settings.require('ready_manifest')


@pytest.mark.parametrize('value', [None, [], {'debug': True},
                                  {'msa_extension_dir': ''}, {'ready_manifest': 1}])
def test_unknown_debug_or_invalid_artifacts_are_rejected(value):
    with pytest.raises(ValueError):
        artifacts(SimpleNamespace(additional_config={'minimax31': value}))


def test_runner_binds_manager_to_its_config_outside_current_config_context(monkeypatch):
    cfg = SimpleNamespace(additional_config={'minimax31': {'msa_extension_dir': '/msa'}},
                          model_config=SimpleNamespace())
    manager = SimpleNamespace()
    def initialize(self, config):
        self.vllm_config = config
        self.pcp_manager = manager
    monkeypatch.setattr(pcp.GPUModelRunner, '__init__', initialize)
    from hcu_vllm_das.parallel import minimax31_static_contract as contract
    monkeypatch.setattr(contract, 'from_model_config', lambda config: object())
    prepared = []
    monkeypatch.setattr(msa, 'prepare_native_prefill', prepared.append)
    pcp.MiniMax31PCPModelRunner(cfg)
    assert len(prepared) == 1 and prepared[0].msa_extension_dir == '/msa'
    assert manager._hcu_native_msa is True
    assert manager._hcu_static is True
    cfg.additional_config = {}
    monkeypatch.setattr(contract, 'from_model_config', lambda config: None)
    pcp.MiniMax31PCPModelRunner(cfg)
    assert manager._hcu_native_msa is False
    assert manager._hcu_static is False


def test_msa_cached_artifact_cannot_drift(monkeypatch, tmp_path):
    adapter = object()
    monkeypatch.setattr(msa, '_ADAPTER', adapter)
    monkeypatch.setattr(msa, '_ADAPTER_ROOT', tmp_path.resolve())
    assert msa._get_adapter(str(tmp_path)) is adapter
    with pytest.raises(RuntimeError, match='artifact changed'):
        msa._get_adapter(str(tmp_path / 'different'))


def test_ready_manifest_is_explicit_and_rejects_mixed_generations(tmp_path):
    rows = [{'rank': i, 'static_publication': dict(generation='a'*32, rank=i,
             gpu_uuid=f'gpu{i}', map_sha256='b'*64, manifest_sha256='c'*64)}
            for i in range(8)]
    path = tmp_path / 'ready.json'
    path.write_text(json.dumps({'ranks': rows}))
    partition = SimpleNamespace(ep_rank=0, gpu_uuid='gpu0', map_sha256='b'*64)
    assert ipc.expected_publication(partition, str(path)) == rows[0]['static_publication']
    rows[-1]['static_publication']['generation'] = 'd'*32
    path.write_text(json.dumps({'ranks': rows}))
    with pytest.raises(ValueError, match='mixed'):
        ipc.expected_publication(partition, str(path))


def test_minimax_production_has_no_private_environment_or_debug_hook():
    root = Path(pcp.__file__).parent
    paths = (
        *root.glob("minimax31*.py"),
        *(root / "minimax31_decode_ops").rglob("*.py"),
        *(root / "minimax31_prefill_ops").rglob("*.py"),
        *(root.parent / "quantization").glob("minimax31*.py"),
        root.parent / "moe/minimax31_ll.py",
    )
    for path in paths:
        source = path.read_text()
        ast.parse(source)
        assert "HCU_MINIMAX_" not in source and "HCU_MINIMAX31_" not in source, path
        assert "static_eplb_capture_hook" not in source, path
    assert not (root / 'minimax31_static_diagnostics.py').exists()


def test_decode_options_keep_qualified_defaults_and_explicit_overrides():
    defaults = decode_options(SimpleNamespace())
    assert defaults.shared_expert_schedule == "auto"
    assert defaults.hipc_verify_tuned
    assert not defaults.shared_w8a8_fused
    assert not defaults.lightop_moe_tuned
    config = SimpleNamespace(additional_config={"minimax31_decode": {
        "shared_expert_schedule": "stock", "hipc_verify_tuned": False,
    }})
    options = decode_options(config)
    config.additional_config["minimax31_decode"]["hipc_verify_tuned"] = True
    assert options.shared_expert_schedule == "stock"
    assert not options.hipc_verify_tuned


@pytest.mark.parametrize("value", [
    {"shared_expert_schedule": "unknown"},
    {"hipc_verify_tuned": "false"},
    {"shared_expert_schedule": "stock", "shared_w8a8_fused": True},
    {"unregistered_option": True},
])
def test_decode_options_reject_invalid_values(value):
    with pytest.raises(ValueError):
        decode_options(SimpleNamespace(additional_config={"minimax31_decode": value}))


def test_prefill_msa_replay_restores_provider_environment(monkeypatch):
    import os
    from hcu_vllm_das.parallel.minimax31_prefill_ops.prefill_attention_helpers import scoped_knobs

    monkeypatch.setenv("MSA128_FP8_PREFILL_KERNEL", "original")
    monkeypatch.delenv("MSA128_FP8_CTA_TOKENS", raising=False)
    with pytest.raises(RuntimeError, match="candidate failed"):
        with scoped_knobs((16, 4, 64)):
            assert os.environ["MSA128_FP8_PREFILL_KERNEL"] == "v2"
            assert os.environ["MSA128_FP8_CTA_TOKENS"] == "16"
            raise RuntimeError("candidate failed")
    assert os.environ["MSA128_FP8_PREFILL_KERNEL"] == "original"
    assert "MSA128_FP8_CTA_TOKENS" not in os.environ


def test_static_hooks_refuse_another_manifest_before_reusing_state(monkeypatch):
    from hcu_vllm_das.parallel import minimax31_static_runtime as runtime
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.gpu.input_batch import InputBatch
    marker = object()
    monkeypatch.setattr(runtime, '_PLACEMENT', marker)
    monkeypatch.setattr(runtime, '_MANIFEST_PATH', '/old/manifest.json')
    monkeypatch.setattr(runtime, 'from_current', lambda: marker)
    monkeypatch.setattr(runtime, '_HOOKS', (RequestState.add_request,
        RequestState.remove_request, InputBatch.make_dummy.__func__))
    monkeypatch.setattr('vllm.config.get_current_vllm_config', lambda: SimpleNamespace(
        additional_config={'minimax31': {'dataset_manifest': '/new/manifest.json'}}))
    with pytest.raises(RuntimeError, match='identity changed'):
        runtime.install_identity_hooks()


def test_runner_fails_before_serving_when_native_msa_receipts_are_missing(monkeypatch, tmp_path):
    artifact = tmp_path / 'build_lib'
    artifact.mkdir()
    cfg = SimpleNamespace(additional_config={'minimax31': {'msa_extension_dir': str(artifact)}})

    def initialize(self, config):
        self.vllm_config = config
        self.pcp_manager = SimpleNamespace()

    monkeypatch.setattr(pcp.GPUModelRunner, '__init__', initialize)
    monkeypatch.setattr(msa, '_ADAPTER', None)
    with pytest.raises(RuntimeError, match='native MSA build receipts') as error:
        pcp.MiniMax31PCPModelRunner(cfg)
    assert isinstance(error.value.__cause__, FileNotFoundError)
