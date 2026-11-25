# SPDX-License-Identifier: Apache-2.0
"""Explicit MiniMax DSpark admission at Core's two closed architecture lists.

Core 0.33 has no registered-draft configuration hook. Keep its real function
objects (including Pydantic's cached alias), changing only these two guards in
memory. Exact source digests reject drift. No framework files are modified.
This admission belongs to the plugin's new model capability.
"""
import ast
import hashlib
import inspect
import textwrap

ARCH='DSparkMiniMaxDraftModel'
_HASHES={'hf_config_override':'53efea6e5530790f900a4bf0919a521b3bf728ee354e6a0475ea172b91019ea5',
         '__post_init__':'66f588ec0831ac3462fc4b2a77d8683ced349e1a07d7ae947491ad4c6d6d9be8'}
_state=None


def _is_minimax_dspark(config):
    if getattr(config,'architectures',None)!=[ARCH]:return False
    if (getattr(config,'model_type',None)!='minimax_m3_vl' or
            getattr(config,'dspark_block_size',None)!=7 or
            getattr(config,'dspark_target_layer_ids',None)!=[3,17,31,45,59] or
            getattr(config,'num_hidden_layers',None)!=5 or
            getattr(config,'hidden_size',None)!=6144):
        raise ValueError('unsupported MiniMax DSpark configuration')
    return True


def _prepare(fn,name):
    source=inspect.getsource(fn)
    if hashlib.sha256(source.encode()).hexdigest()!=_HASHES[name]:
        raise RuntimeError('MiniMax DSpark Core configuration source changed: '+name)
    tree=ast.parse(textwrap.dedent(source));node=tree.body[0]
    node.decorator_list=[]
    if name=='hf_config_override':
        # Preserve explicit Transformers-MTP overrides and every other model.
        guard=ast.parse('if transformers_mtp_overrides is None and _hcu_minimax_dspark_config(hf_config):\n    return hf_config').body[0]
        node.body.insert(0,guard)
    else:
        matches=[]
        for item in ast.walk(node):
            if not isinstance(item,ast.If) or not isinstance(item.test,ast.BoolOp):continue
            text=ast.unparse(item.test)
            if ("'K3DSparkModel' not in self.draft_model_config.architectures" in text
                    and "self.method == 'dspark'" in text):matches.append(item)
        if len(matches)!=1:raise RuntimeError('Core DSpark fallback guard changed')
        matches[0].test.values.append(ast.parse(
            repr(ARCH)+' not in self.draft_model_config.architectures',mode='eval').body)
    ast.fix_missing_locations(tree)
    namespace=dict(fn.__globals__)
    namespace['_hcu_minimax_dspark_config']=_is_minimax_dspark
    exec(compile(tree,fn.__code__.co_filename,'exec'),namespace)
    code=namespace[fn.__name__].__code__
    if code.co_freevars!=fn.__code__.co_freevars:
        raise RuntimeError('MiniMax DSpark configuration closure changed')
    return code


def install():
    global _state
    if _state is not None:check_health();return
    from vllm.config.speculative import SpeculativeConfig
    functions=[(getattr(SpeculativeConfig,n),n) for n in _HASHES]
    plans=[(fn,_prepare(fn,name)) for fn,name in functions]
    for fn,code in plans:
        existing=fn.__globals__.get('_hcu_minimax_dspark_config',_is_minimax_dspark)
        if existing is not _is_minimax_dspark:
            raise RuntimeError('foreign MiniMax DSpark configuration owner')
    for fn,code in plans:
        fn.__globals__['_hcu_minimax_dspark_config']=_is_minimax_dspark
        fn.__code__=code
    _state=plans


def check_health():
    if _state is None or any(fn.__code__ is not code or
            fn.__globals__.get('_hcu_minimax_dspark_config') is not _is_minimax_dspark
            for fn,code in _state):
        raise RuntimeError('MiniMax DSpark configuration admission drift')
