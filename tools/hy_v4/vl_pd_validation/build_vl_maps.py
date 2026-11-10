"""Owned CPU-only HYV4-VL placement generator and count-balance validation.

No capture here is a FLOP/utilization/TTFT measure. Policy predictions assume
an ideal equal split among replicas; actual locality dispatch requires capture.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re

MODEL_KEY = 'HYV4VLForConditionalGeneration'
MODEL_PATH = '/data_hy4_w4a8/Hy4-testing-Channel-INT4-w4a8'
POLICY_SHA256 = '3b54229e6240d050480390f6b29d4744fd38437d23719f78a6decf7002296a29'
CONFIG_SHA256 = '738902ee7f32d7f4db373c7133e98a2b284dc395784144d04c677cf5bec419b7'
INDEX_SHA256 = 'b7019fa49f4c9eaf1a629fd6ac06daa463ce911efe9dd882232b9aa00568f4be'
INT64_MAX = (1 << 63)-1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def integer(value, name, minimum=0):
    require(type(value) is int and minimum <= value <= INT64_MAX,
            name + ' must be a nonnegative int64' if minimum == 0 else name + ' must be a positive int64')
    return value


def histogram(values, length, name='histogram'):
    require(isinstance(values, list) and len(values) == length, name+' has wrong length')
    for x in values:
        integer(x, name)
    require(sum(values) <= INT64_MAX, name+' total exceeds int64')
    return values


def validate_map(rows, physical, layers=77):
    require(physical in (256,272), 'physical count must be 256 or 272')
    require(isinstance(rows,list) and len(rows)==layers, 'wrong MoE row count')
    for row in rows:
        require(isinstance(row,list) and len(row)==physical, 'wrong physical row length')
        require(all(type(x) is int and 0 <= x < 256 for x in row), 'bad logical ID/type in map')
        require(set(row)==set(range(256)), 'uncovered logical expert')
        require(len(row)-len(set(row))==physical-256, 'wrong redundancy')
    return rows


def scatter_logical(physical_histogram, physical_to_logical):
    validate_map([physical_to_logical],len(physical_to_logical),layers=1)
    histogram(physical_histogram,len(physical_to_logical),'physical_histogram')
    result=[0]*256
    for count,logical in zip(physical_histogram,physical_to_logical):
        result[logical]+=count
    return result


def destination_loads(physical_histograms):
    require(bool(physical_histograms),'no layers')
    physical=len(physical_histograms[0])
    require(physical in (256,272),'wrong physical width')
    slots=physical//16
    return [[sum(histogram(row,physical)[r*slots:(r+1)*slots]) for r in range(16)]
            for row in physical_histograms]


def balance_metrics(loads):
    require(bool(loads),'no layers')
    ratios=[];cvs=[];nonzero=[];maxima=[];means=[];zeros=[]
    cumulative=[0.0]*16
    for layer,row in enumerate(loads):
        require(len(row)==16 and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in row),
                'rank loads must be 16 finite nonnegative counts')
        for rank,x in enumerate(row):cumulative[rank]+=x
        mean=math.fsum(row)/16
        if mean==0:
            zeros.append(layer);continue
        maximum=max(row);ratios.append(maximum/mean)
        cvs.append(math.sqrt(math.fsum((x-mean)**2 for x in row)/16)/mean)
        means.append(mean);maxima.append(maximum);nonzero.append(layer)
    mean=math.fsum(cumulative)/16
    worst=max(range(len(ratios)),key=ratios.__getitem__) if ratios else None
    return {'weighted_layer_max_over_mean':math.fsum(maxima)/math.fsum(means) if means else None,
            'worst_layer_max_over_mean':ratios[worst] if worst is not None else None,
            'worst_layer_index':nonzero[worst] if worst is not None else None,
            'mean_layer_cv':math.fsum(cvs)/len(cvs) if cvs else None,
            'cumulative_rank_cv':math.sqrt(math.fsum((x-mean)**2 for x in cumulative)/16)/mean if mean else None,
            'cumulative_rank_max_over_mean':max(cumulative)/mean if mean else None,
            'zero_layer_indexes':zeros,'executed_layer_count':len(nonzero)}


def validate_capture(data):
    require(isinstance(data,dict),'capture must be an object')
    rank=integer(data.get('rank'),'rank');require(rank<16,'rank out of range')
    require(isinstance(data.get('scope'),str) and 'PCP padding' in data['scope'], 'capture must label executed PCP padding scope')
    records=data.get('records');require(isinstance(records,list) and len(records)==77,'exactly 77 records required')
    indexed={}
    widths=set()
    for record in records:
        require(record.get('model_key')==MODEL_KEY,'wrong model key; text/MTP captures forbidden')
        index=integer(record.get('layer_index'),'layer_index')
        require(index<77 and index not in indexed,'missing/duplicate/out-of-range MoE row')
        require(record.get('num_logical_experts')==256 and type(record.get('num_logical_experts')) is int,'wrong logical expert count')
        require(record.get('top_k')==8 and type(record.get('top_k')) is int,'wrong top-k')
        calls=integer(record.get('calls'),'calls',1);rows=integer(record.get('rows'),'rows',1)
        require(calls<=rows,'calls cannot exceed routed rows')
        require(integer(record.get('mapping_calls'),'mapping_calls',1)==calls,'capture/mapping call mismatch')
        ignored=integer(record.get('ignored_invalid_slots'),'ignored_invalid_slots')
        logical=histogram(record.get('histogram'),256)
        require(ignored<=rows*8 and sum(logical)==rows*8-ignored,'logical total differs from rows*topk-ignored')
        physical=record.get('physical_to_logical');require(isinstance(physical,list),'missing physical map')
        validate_map([physical],len(physical),layers=1);widths.add(len(physical))
        counts=histogram(record.get('physical_histogram'),len(physical),'physical_histogram')
        require(record.get('physical_mapping_verified') is True,'physical per-selection validation missing')
        require(integer(record.get('physical_ignored_invalid_slots'),'physical_ignored_invalid_slots')==ignored,'invalid-slot mismatch')
        require(scatter_logical(counts,physical)==logical,'physical scatter does not reconstruct logical selections')
        indexed[index]=record
    require(len(widths)==1,'mixed physical widths within rank')
    return [indexed[i] for i in range(77)]


def load_capture_manifest(manifest):
    workers=manifest.get('workers')
    require(isinstance(workers,list) and len(workers)==16,'all 16 source ranks required')
    require(all(type(w.get('rank')) is int for w in workers),'rank must be integer')
    require({w['rank'] for w in workers}==set(range(16)),'duplicate/missing source ranks')
    logical=[[0]*256 for _ in range(77)]
    physical=None;source_files=[];bound_rows=None
    for worker in sorted(workers,key=lambda w:w['rank']):
        path=Path(worker['file']);raw=path.read_bytes()
        require(sha256(raw)==worker.get('sha256'),'capture SHA256 mismatch: '+str(path))
        data=json.loads(raw);require(data.get('rank')==worker['rank'],'manifest/capture rank mismatch')
        records=validate_capture(data)
        current_rows=[r['physical_to_logical'] for r in records]
        if bound_rows is None:bound_rows=current_rows
        require(bound_rows==current_rows,'source ranks have different bound maps')
        if physical is None:physical=[[0]*len(bound_rows[0]) for _ in range(77)]
        for layer,record in enumerate(records):
            logical[layer]=[a+b for a,b in zip(logical[layer],record['histogram'])]
            physical[layer]=[a+b for a,b in zip(physical[layer],record['physical_histogram'])]
            histogram(logical[layer],256);histogram(physical[layer],len(bound_rows[0]))
        source_files.append({'rank':worker['rank'],'file':str(path),'sha256':sha256(raw)})
    return {'logical':logical,'physical':physical,'map':bound_rows,'sources':source_files,
            'actual_count_balance':balance_metrics(destination_loads(physical))}


def ideal_equal_split(logical,rows):
    validate_map(rows,len(rows[0]),layers=len(logical))
    slots=len(rows[0])//16;loads=[]
    for weights,row in zip(logical,rows):
        histogram(weights,256)
        replicas=[row.count(i) for i in range(256)]
        loads.append([math.fsum(weights[e]/replicas[e] for e in row[r*slots:(r+1)*slots]) for r in range(16)])
    return {'label':'ideal equal-split prediction; not measured replica dispatch',
            'metrics':balance_metrics(loads),'per_layer_rank_loads':loads}


def validate_provenance(data):
    require(data.get('model_path')==MODEL_PATH and data.get('model_class')==MODEL_KEY,'wrong checkpoint identity')
    require(data.get('speculative_config') is None and 'speculative_config' in data,'noMTP must be explicit')
    for key in ('config_sha256','index_sha256','corpus_sha256','core_sha256','plugin_sha256'):
        require(isinstance(data.get(key),str) and re.fullmatch('[0-9a-f]{64}',data[key]),'missing/invalid '+key)
    require(data['config_sha256']==CONFIG_SHA256 and data['index_sha256']==INDEX_SHA256,'checkpoint metadata binding mismatch')
    for key in ('item_image_hashes','fit_item_ids','heldout_item_ids','preprocessing_limits','input_output_lengths',
                'concurrency','prefix_cache_state','prompt_templates','source_files'):
        require(key in data and data[key] is not None,'missing provenance '+key)
    fit=data['fit_item_ids'];heldout=data['heldout_item_ids']
    require(isinstance(fit,list) and isinstance(heldout,list) and fit and heldout,'nonempty corpus splits required')
    require(len(set(fit))==len(fit) and len(set(heldout))==len(heldout) and not set(fit)&set(heldout),'fit/heldout overlap or duplicate')
    require(isinstance(data['item_image_hashes'],dict) and set(fit+heldout)<=set(data['item_image_hashes']),'missing item/image hashes')
    for item in fit+heldout:
        item_hashes=data['item_image_hashes'][item]
        require(isinstance(item_hashes,dict) and item_hashes,'item/image hashes must be nonempty objects')
        require(all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in item_hashes.values()),'bad corpus item/image SHA256')
    require(isinstance(data['source_files'],list) and data['source_files'],'source files are required')
    for source in data['source_files']:
        require(sha256(Path(source['file']).read_bytes())==source.get('sha256'),'source SHA256 mismatch')
    return data


def load_policy(source, expected_sha):
    # Hide accelerators even if this helper is imported on the native D CPU lane.
    os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',VLLM_PLUGINS='')
    import torch
    if source:
        import numpy as np
        path=Path(source);raw=path.read_bytes()
        require(sha256(raw)==expected_sha,'policy SHA256 mismatch')
        tree=ast.parse(raw,filename=str(path))
        classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DefaultEplbPolicy']
        require(len(classes)==1,'native policy class missing')
        # Execute the pinned class body verbatim; the abstract base carries no policy
        # behavior. This avoids importing GPU/platform-dependent vLLM on CPU.
        namespace={'np':np,'torch':torch,'AbstractEplbPolicy':object,'__name__':'owned_pinned_native_policy'}
        exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),'exec'),namespace)
        policy=namespace['DefaultEplbPolicy']
    else:
        import inspect
        from vllm.distributed.eplb.policy.default import DefaultEplbPolicy as policy
        path=Path(inspect.getsourcefile(policy));raw=path.read_bytes()
        require(sha256(raw)==expected_sha,'installed policy SHA256 mismatch')
    return policy,{'file':str(path),'sha256':sha256(raw),'class':'DefaultEplbPolicy'}


def output_path(value):
    path=PurePosixPath(value)
    require(path.is_absolute() and '..' not in path.parts and path.is_relative_to('/workspace/hy4_mm_20261108') and str(path)!='/workspace/hy4_mm_20261108',
            'outputs must be absolute and inside /workspace/hy4_mm_20261108/')
    resolved=Path(value).resolve()
    require(resolved.is_relative_to(Path('/workspace/hy4_mm_20261108').resolve()),'output escapes experiment through symlink')
    return Path(value)


def generate(logical, redundancy, policy):
    import torch
    require(redundancy in (0,16),'redundancy must be 0 or 16')
    require(len(logical)==77,'77 logical rows required')
    for row in logical:histogram(row,256)
    physical=256+redundancy
    initial=torch.tensor([list(range(256))+list(range(redundancy))]*77,dtype=torch.int64,device='cpu')
    candidate=policy.rebalance_experts(torch.tensor(logical,dtype=torch.int64,device='cpu'),
                    num_replicas=physical,num_groups=1,num_nodes=1,num_ranks=16,
                    old_global_expert_indices=initial)
    require(candidate.device.type=='cpu' and candidate.dtype==torch.int64,'policy output must be CPU int64')
    rows=validate_map(candidate.tolist(),physical)
    return {'version':2,'model_maps':{MODEL_KEY:{'model_name':MODEL_PATH,'model_class':MODEL_KEY,
            'num_logical_experts':256,'num_physical_experts':physical,'num_redundant_experts':redundancy,
            'num_moe_layers':77,'physical_to_logical_map':rows}}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyze-manifest',help='Print verified actual physical count balance only; no policy or GPU imports')
    parser.add_argument('--fit-manifest')
    parser.add_argument('--heldout-manifest')
    parser.add_argument('--provenance')
    parser.add_argument('--policy-source',help='Pinned native default.py; execute only policy class on CPU')
    parser.add_argument('--policy-sha256',default=POLICY_SHA256)
    parser.add_argument('--redundancy',type=int,choices=[0,16],default=16)
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.analyze_manifest:
        raw=Path(args.analyze_manifest).read_bytes()
        captured=load_capture_manifest(json.loads(raw))
        print(json.dumps({'manifest_sha256':sha256(raw),'sources':captured['sources'],
            'actual_count_balance':captured['actual_count_balance'],
            'per_layer_rank_loads':destination_loads(captured['physical']),
            'total_selected_ids':sum(map(sum,captured['logical'])),
            'physical_experts':len(captured['physical'][0]),
            'scope':'actual selected-ID counts including executed PCP padding; -1 slots excluded; not FLOPs/utilization/TTFT'}))
        return
    for name in ('fit_manifest','heldout_manifest','provenance','output'):
        if not getattr(args,name):parser.error('--'+name.replace('_','-')+' required for generation')
    target=output_path(args.output);report_path=target.with_suffix('.report.json')
    require(not target.exists() and not report_path.exists(),'refuse map/report overwrite')
    raws={name:Path(getattr(args,name.replace('-','_'))).read_bytes() for name in ('fit-manifest','heldout-manifest','provenance')}
    provenance=validate_provenance(json.loads(raws['provenance']))
    fit=load_capture_manifest(json.loads(raws['fit-manifest']))
    heldout=load_capture_manifest(json.loads(raws['heldout-manifest']))
    require(not {x['sha256'] for x in fit['sources']}&{x['sha256'] for x in heldout['sources']},'fit/heldout capture reuse forbidden')
    policy,policy_info=load_policy(args.policy_source,args.policy_sha256)
    mapping=generate(fit['logical'],args.redundancy,policy)
    rows=mapping['model_maps'][MODEL_KEY]['physical_to_logical_map']
    report={'map_file':str(target),'map_sha256':sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()),
        'static_dispatch_policy':'nearest','fit_only':True,'heldout_used_for_fit':False,'provenance':provenance,
        'input_sha256':{k:sha256(v) for k,v in raws.items()},'policy':policy_info,
        'generator_sha256':sha256(Path(__file__).read_bytes()),'sources':{'fit':fit['sources'],'heldout':heldout['sources']},
        'actual_count_balance':{'fit':fit['actual_count_balance'],'heldout':heldout['actual_count_balance']},
        'prediction':{name:{'identity_r0':ideal_equal_split(capture['logical'],[list(range(256))]*77),
                            'candidate':ideal_equal_split(capture['logical'],rows)} for name,capture in [('fit',fit),('heldout',heldout)]},
        'scope':'Selected-ID counts include executed PCP padding, exclude -1 invalid slots; not GPU utilization, padded FLOPs, communication bytes or TTFT.'}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode())
    with report_path.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report))

if __name__=='__main__':main()
