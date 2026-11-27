"""Exercise real strict-loader/daemon methods; double only the final wire/export."""
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import pytest
from hcu_vllm_das.parallel import minimax31_decode_ipc as ipc
from hcu_vllm_das.parallel.minimax31_decode_contract import partition


def test_partition_generation_and_reload_refuse_before_native_export(monkeypatch):
    daemon=object.__new__(ipc.DecodeDaemon)
    daemon.decode_partition=partition('target',2,'GPU-2','a'*64)
    daemon.decode_generation='fresh'
    exported=[];responses=[]
    monkeypatch.setattr(daemon,'_handle_get_state',lambda conn,request:exported.append(request))
    monkeypatch.setattr(ipc,'send_msg',lambda conn,response:responses.append(response))
    key=object()
    base=dict(cmd=ipc.COMMAND,cache_key=key,partition=daemon.decode_partition.to_dict(),
              generation='fresh',is_reload=False)
    changed=[dict(base,generation='stale'),dict(base,is_reload=True),
             dict(base,partition=replace(daemon.decode_partition,role='draft').to_dict()),
             dict(base,partition=replace(daemon.decode_partition,pcp_size=8).to_dict()),
             dict(base,partition=replace(daemon.decode_partition,gpu_uuid='wrong').to_dict())]
    for request in changed:
        monkeypatch.setattr(ipc,'recv_msg',lambda conn,r=request:r)
        daemon._handle_connection(None)
        assert responses[-1]['status']=='mismatch'
        assert not exported
    monkeypatch.setattr(ipc,'recv_msg',lambda conn:base)
    daemon._handle_connection(None)
    assert len(exported)==1 and exported[0]['cache_key'] is key


def test_custom_format_accounts_for_both_roles_without_core_dspark_allowlist(monkeypatch):
    from vllm.config import LoadConfig
    def config(draft):
        return LoadConfig(load_format=ipc.LOAD_FORMAT,model_loader_extra_config={
            'socket_dir':'/tmp/contract','mode':'zero_copy','fallback':False,
            'is_draft':draft,'decode_ready_manifest':'/tmp/not-opened-by-budget.json'})
    loader=ipc.DecodeIpcLoader(config(False))
    monkeypatch.setattr(ipc.DecodeIpcLoader,'_daemon_memory',lambda self:17 if self.is_draft else 101)
    spec=SimpleNamespace(use_dspark=lambda:True,draft_load_config=config(True))
    assert loader.get_external_weight_memory(SimpleNamespace(speculative_config=spec))==118
    spec.draft_load_config=config(False)
    with pytest.raises(ValueError,match='role was lost'):
        loader.get_external_weight_memory(SimpleNamespace(speculative_config=spec))
    spec.draft_load_config=LoadConfig(load_format='safetensors')
    with pytest.raises(ValueError,match='strict draft IPC'):
        loader.get_external_weight_memory(SimpleNamespace(speculative_config=spec))


@pytest.mark.parametrize("settings", [
    {"decode_dynamic_eplb": True},
    {"decode_eplb": {"window": 16, "interval": 96}},
])
def test_dynamic_eplb_fails_before_initializing_ipc(monkeypatch, settings):
    from vllm.config import LoadConfig

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported EPLB reached IPC initialization")

    monkeypatch.setattr(ipc.IpcModelLoader, "__init__", unexpected)
    config = LoadConfig(
        load_format=ipc.LOAD_FORMAT,
        model_loader_extra_config={
            "decode_ready_manifest": "/tmp/unused-ready.json",
            **settings,
        },
    )
    with pytest.raises(ValueError, match="use a frozen expert placement"):
        ipc.DecodeIpcLoader(config)


def test_weight_identity_checks_actual_import_resolution(monkeypatch,tmp_path):
    actual=tmp_path/'installed.py';actual.write_text('installed = True\n')
    expected=hashlib.sha256(actual.read_bytes()).hexdigest()
    seen=[]
    def spec(module):
        seen.append(module);return SimpleNamespace(origin=str(actual))
    monkeypatch.setattr(ipc.importlib.util,'find_spec',spec)
    declared='/checkout/adapter_extension/src/hcu_vllm_das/quantization/minimax31.py'
    ipc.verify_imported_weight_source(declared,expected)
    assert seen==['hcu_vllm_das.quantization.minimax31']
    actual.write_text('installed = False\n')
    with pytest.raises(ValueError,match='actual imported'):
        ipc.verify_imported_weight_source(declared,expected)
