# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from vllm_hcu.v1.locateanything.request_contract import validate_request

assert validate_request(SimpleNamespace()) is None
assert validate_request(SimpleNamespace(stream=False,n=1,temperature=.7,top_p=.9)) is None
for key,value in [('stream',True),('n',2),('logprobs',True),('presence_penalty',.1),
                  ('stop',['foo']),('ignore_eos',True),('min_tokens',3),('structured_outputs',{'json':{}})]:
    assert key in validate_request(SimpleNamespace(**{key:value}))

class Status:
    FINISHED_STOPPED='stop'
    FINISHED_LENGTH_CAPPED='length'
class Request:
    def __init__(self,prompt=5,already=0,budget=10):
        self.num_tokens=prompt+already;self.num_output_tokens=already
        self.max_tokens=budget;self.status=None;self.output=[]
    def append_output_token_ids(self,ids):
        self.output.extend(ids);self.num_tokens+=len(ids);self.num_output_tokens+=len(ids)

tree=ast.parse((Path(importlib.util.find_spec('vllm_hcu').origin).parent/'v1/locateanything/scheduler.py').read_text())
node=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body
          if isinstance(n,ast.FunctionDef) and n.name=='_update_request_with_output')
ns={'Request':Request,'RequestStatus':Status}
# Offline oracle over fixed package source, never HTTP/client input; see
# docs/locateanything.md#validation-source-trust for this narrow exception.
exec(compile(ast.Module(body=[node],type_ignores=[]),'fast_stop','exec'),ns)  # nosemgrep: rules.hygon.python.dynamic-eval
update=ns['_update_request_with_output']
tested=0
for budget in range(1,7):
    for size in range(1,7):
        r=Request(already=budget-1,budget=budget)
        out,stop=update(None,r,list(range(10,10+size)))
        assert stop and r.status=='length' and len(out)==size and r.num_output_tokens==budget-1+size
        tested+=1
r=Request();out,stop=update(None,r,[9,151645,10]);assert not stop and out==[9,151645,10]
r=Request();out,stop=update(None,r,[151645]);assert stop and r.status=='stop'
r=Request(prompt=16383,budget=10);out,stop=update(None,r,[8,9,10,11,12,13]);assert stop and r.status=='length' and len(out)==6
print({'passed':True,'budget_cases':tested,'other_stop_cases':3,'API_option_cases':10})
