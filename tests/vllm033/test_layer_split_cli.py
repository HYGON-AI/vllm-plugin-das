"""CPU tests for the real vLLM serve parser and HCU's platform hook."""

import json
from types import SimpleNamespace

import pytest

from hcu_adapter.bridges import vllm as bridge, vllm_cli
from hcu_vllm_das.layer_split_config import FLAG, KEY, layer_split_enabled, register_cli
from vllm.engine import arg_utils
from vllm.entrypoints.common.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setattr(vllm_cli, '_switches', {})
    register_cli()
    register_cli()
    monkeypatch.setattr(arg_utils, 'current_platform', bridge.HcuPlatform)
    monkeypatch.setattr('huggingface_hub.constants.HF_HUB_OFFLINE', False)
    root = FlexibleArgumentParser(prog='vllm')
    serve = root.add_subparsers().add_parser('serve')
    make_arg_parser(serve)
    assert FLAG in serve.format_help()
    return root


@pytest.mark.parametrize('order', ['absent', 'first', 'last'])
def test_flag_reaches_engine_config_without_clobbering_static_eplb(parser, order):
    data = {'hcu_static_eplb': {'num_redundant_experts': 32}}
    argv = ['--additional-config', json.dumps(data)]
    argv = {'absent': argv, 'first': [FLAG, *argv], 'last': [*argv, FLAG]}[order]
    args = arg_utils.AsyncEngineArgs.from_cli_args(parser.parse_args(['serve', 'not-loaded', *argv]))
    assert layer_split_enabled(args) == (order != 'absent')
    assert args.additional_config['hcu_static_eplb'] == data['hcu_static_eplb']
    assert ('_hcu_cli_' + KEY) not in vars(args)


@pytest.mark.parametrize('value', [False, 1, 'true'])
def test_conflicting_json_is_refused(parser, value, capsys):
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['serve', 'not-loaded', FLAG, '--additional-config',
                           json.dumps({KEY: {'enabled': value}})])
    assert exc.value.code == 2
    assert FLAG + ' conflicts' in capsys.readouterr().err


def test_config_is_strict_and_does_not_read_legacy_env(monkeypatch):
    monkeypatch.setenv('VLLM_HCU_DSV41_LAYER_SPLIT', '1')
    assert not layer_split_enabled(SimpleNamespace(additional_config={}))
    with pytest.raises(ValueError, match='boolean'):
        layer_split_enabled(SimpleNamespace(additional_config={KEY: {'enabled': '1'}}))


def test_missing_companion_leaves_legacy_policy_and_refuses_layer_split(monkeypatch):
    from hcu_vllm_das import inverse_rope_policy, layer_split_config, v41_pcp

    monkeypatch.setattr(inverse_rope_policy, 'optional_adapter_module', lambda _: None)
    monkeypatch.setattr(layer_split_config, 'optional_adapter_module', lambda _: None)
    operators = {}
    config = inverse_rope_policy.adapter_module('hcu_config')
    monkeypatch.setattr(config, 'load', lambda: {'ops': operators, 'overrides': {}})
    monkeypatch.setenv('HCU_DISABLE_OPS', '')
    monkeypatch.setenv('VLLM_HCU_DSV41_PCP_ATTENTION', '1')
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    assert register_cli() is None
    assert inverse_rope_policy.selected_mode() == 'stock'
    operators['vllm_inverse_rope'] = 'hcu'
    assert inverse_rope_policy.selected_mode() == 'hcu'
    monkeypatch.setenv('HCU_DISABLE_OPS', 'vllm_inverse_rope')
    assert inverse_rope_policy.selected_mode() == 'stock'
    with pytest.raises(ValueError, match='companion HCU Adapter'):
        v41_pcp.validate_layer_split_policy(SimpleNamespace(
            additional_config={KEY: {'enabled': True}}))


def test_optional_companion_does_not_hide_an_inner_import_failure(monkeypatch):
    from hcu_vllm_das.adapter_api import importlib, optional_adapter_module

    def import_module(name):
        raise ModuleNotFoundError(name='missing_inner_dependency')

    monkeypatch.setattr(importlib, 'import_module', import_module)
    with pytest.raises(ModuleNotFoundError) as error:
        optional_adapter_module('hcu_adapter.bridges.vllm_cli')
    assert error.value.name == 'missing_inner_dependency'
