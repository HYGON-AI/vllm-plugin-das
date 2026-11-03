"""Actual installed CLI + explicit backend selector, without provider doubles."""
import importlib,json,os,runpy,sys,traceback
from pathlib import Path
out=Path(sys.argv[1]);assert not out.exists();assert not os.environ.get('PYTHONPATH')
result={'scope':'actual installed official CLI, HCU detection, vendor callable preparation and backend selection only; no kernel/model claim'}
try:
 sys.argv=['vllm.entrypoints.openai.api_server','--help']
 try:runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
 except SystemExit as exc:assert exc.code in (None,0),exc.code
 from hcu_adapter.bridges.vllm import extension_reports
 from hcu_adapter.extensions import ExtensionStatus
 reports=extension_reports()
 result['reports']=[{'entry_point':r.entry_point_name,'status':r.status.value,'capabilities':sorted(r.capabilities),'reason':r.reason} for r in reports]
 owners=[r for r in reports if 'hcu-flash-attention' in r.capabilities]
 assert len(owners)==1 and owners[0].status is ExtensionStatus.ACTIVE,result['reports']
 import torch
 from vllm.platforms import current_platform
 from vllm.v1.attention.selector import AttentionSelectorConfig
 from vllm.v1.attention.backends.registry import AttentionBackendEnum
 selected=[]
 for dtype in ['auto','fp8_e5m2']:
  config=AttentionSelectorConfig(head_size=128,dtype=torch.bfloat16,kv_cache_dtype=dtype,block_size=64)
  name=current_platform.get_attn_backend_cls(AttentionBackendEnum.FLASH_ATTN,config)
  assert name=='hcu_vllm_das.attention.flash_attn.HcuFlashAttentionBackend',name
  selected.append({'kv_dtype':dtype,'class':name})
 config=AttentionSelectorConfig(head_size=128,dtype=torch.bfloat16,kv_cache_dtype='auto',block_size=128)
 try:current_platform.get_attn_backend_cls(AttentionBackendEnum.FLASH_ATTN,config)
 except ValueError as exc:result['invalid_block_size_refused']=str(exc)
 else:raise AssertionError('invalid Flash block size accepted')
 config=AttentionSelectorConfig(head_size=128,dtype=torch.bfloat16,kv_cache_dtype='auto',block_size=64)
 default=current_platform.get_attn_backend_cls(None,config)
 assert 'triton_attn' in default.lower(),default
 result['default_backend']=default
 result['explicit_flash_selections']=selected
 result['origins']={name:str(Path(importlib.import_module(name).__file__).resolve()) for name in ['hcu_adapter','hcu_adapter.bridges.vllm','hcu_vllm_das.flash_activation','hcu_vllm_das.attention.flash_attn','vllm']}
 assert all(Path(v).is_relative_to(Path(sys.prefix)) for v in result['origins'].values())
 result['status']='passed'
except BaseException as exc:
 result['status']='failed';result['error']=type(exc).__name__+': '+str(exc);traceback.print_exc();raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n')
print('Actual Flash owner CLI/selector passed',flush=True)
