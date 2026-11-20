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
