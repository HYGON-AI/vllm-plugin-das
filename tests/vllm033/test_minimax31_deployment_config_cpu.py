# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real CLI artifact transport, default path binding and fail-closed identity."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hcu_vllm_das.parallel.minimax31_config import artifacts
from hcu_vllm_das.parallel import minimax31_pcp as pcp
from hcu_vllm_das.parallel import minimax31_static_ipc as ipc
from hcu_vllm_das.parallel import minimax31_msa as msa


def test_public_engine_cli_transports_artifacts_without_environment():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = EngineArgs.from_cli_args(parser.parse_args([
        '--model', 'unused-checkpoint', '--additional-config',
        json.dumps({'minimax31': {'msa_extension_dir': '/artifacts/msa',
                                 'loading_receipts': '/outputs/loading'}})]))
    settings = artifacts(args)
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
    pcp.MiniMax31PCPModelRunner(cfg)
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
    for path in [*root.glob('minimax31*.py'), root.parent/'quantization/minimax31.py']:
        source = path.read_text()
        ast.parse(source)
        assert 'HCU_MINIMAX_' not in source, path
        assert 'static_eplb_capture_hook' not in source, path
    assert not (root / 'minimax31_static_diagnostics.py').exists()


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
