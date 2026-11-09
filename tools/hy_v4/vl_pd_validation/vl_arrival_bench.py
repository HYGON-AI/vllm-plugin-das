"""Owned real-image Poisson-arrival client. Importing/testing sends no requests.

All dataset/image/request preparation precedes the timed origin. No API warmup,
prefix-cache reset, retry, or backend control is performed by this operator.
"""
import argparse
import asyncio
import base64
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random
import time
import uuid

REVISION='17e7f75e092e47169732462ea3cdfebe911105dd'


def require(condition,message):
    if not condition:raise ValueError(message)


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')


def sha256(raw):return hashlib.sha256(raw).hexdigest()


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


class SSEParser:
    """Incremental SSE event parsing with first *content* timestamp, not role."""
    def __init__(self):
        self.pending=[];self.raw_events=[];self.errors=[];self.parts=[]
        self.first_content_s=None;self.finish_reason=None;self.usage=None;self.done=False

    @property
    def content(self):return ''.join(self.parts)

    def feed_line(self,line,received_s):
        line=line.rstrip('\r\n')
        if not line:
            self._dispatch(received_s);return
        if line.startswith(':'):return
        field,_,value=line.partition(':')
        if value.startswith(' '):value=value[1:]
        if field=='data':self.pending.append(value)

    def _dispatch(self,received_s):
        if not self.pending:return
        raw='\n'.join(self.pending);self.pending=[]
        self.raw_events.append({'received_s':received_s,'data':raw})
        if raw=='[DONE]':self.done=True;return
        if self.done:
            self.errors.append('data received after [DONE]');return
        try:
            value=json.loads(raw)
            require(isinstance(value,dict),'SSE JSON must be an object')
            if value.get('error') is not None:
                self.errors.append('SSE error: '+json.dumps(value['error'],ensure_ascii=False));return
            if value.get('usage') is not None:
                require(isinstance(value['usage'],dict),'usage must be an object')
                self.usage=value['usage']
            choices=value.get('choices',[])
            require(isinstance(choices,list),'choices must be a list')
            for choice in choices:
                require(isinstance(choice,dict),'choice must be an object')
                if choice.get('index',0)!=0:continue
                delta=choice.get('delta') or {}
                require(isinstance(delta,dict),'delta must be an object')
                content=delta.get('content')
                if content is not None:
                    require(isinstance(content,str),'stream content must be text')
                    if content:
                        if self.first_content_s is None:self.first_content_s=received_s
                        self.parts.append(content)
                reason=choice.get('finish_reason')
                if reason is not None:
                    require(isinstance(reason,str),'finish_reason must be text')
                    if self.finish_reason is not None and self.finish_reason!=reason:
                        self.errors.append('conflicting finish reasons')
                    self.finish_reason=reason
        except (ValueError,TypeError) as error:
            self.errors.append(type(error).__name__+': '+str(error))

    def finish(self,received_s):self._dispatch(received_s)


def select_items(rows,corpus,*,count=512,seed=255):
    require(type(count) is int and count>=1,'count must be integer>=1')
    groups=corpus.get('groups',{});items=corpus.get('items',{})
    fit=groups.get('fit');heldout=groups.get('heldout')
    require(isinstance(fit,list) and isinstance(heldout,list) and fit and heldout,'fit and heldout exclusions required')
    excluded=fit+heldout
    require(all(type(i) is int and 0<=i<len(rows) for i in excluded),'calibration ID outside source dataset')
    require(len(set(excluded))==len(excluded),'calibration IDs duplicated or overlapping')
    hashes=[sha256(row['image']['bytes']) for row in rows]
    excluded_hashes=set()
    for index in excluded:
        item_id='realworldqa-'+str(index)
        require(item_id in items,'missing excluded corpus item '+item_id)
        expected=items[item_id].get('raw_image_sha256')
        require(hashes[index]==expected,'excluded item raw-image hash differs from source: '+item_id)
        excluded_hashes.add(expected)
    pool=[];seen=set()
    for index,row in enumerate(rows):
        raw_hash=hashes[index]
        if index in set(excluded) or raw_hash in excluded_hashes or raw_hash in seen:continue
        require(isinstance(row.get('question'),str) and row['question'].strip(),'source question missing')
        seen.add(raw_hash)
        pool.append({'index':index,'item_id':'realworldqa-'+str(index),'question':row['question'].strip(),
                     'raw_image_sha256':raw_hash,'image_bytes':row['image']['bytes']})
    require(count<=len(pool),f'count {count} exceeds unique available pool {len(pool)} after fit/heldout image exclusion')
    selected=random.Random(seed).sample(pool,count)
    return selected,{'unique_available':len(pool),'source_rows':len(rows),'excluded_ids':sorted(excluded),
        'excluded_raw_image_sha256':sorted(excluded_hashes),'fit_count':len(fit),'heldout_count':len(heldout),
        'selected_ids':[item['item_id'] for item in selected],'selection_seed':seed,
        'deduplication':'raw source image bytes SHA256; first source row retained per remaining image'}


def freeze_request(item,png_bytes,*,model='hy4-vl-w4a8',max_tokens=16):
    require(type(max_tokens) is int and max_tokens>=1,'max_tokens must be positive integer')
    question=item['question'].strip();require(bool(question),'empty original question')
    png_hash=sha256(png_bytes)
    item_payload={'index':item['index'],'question':question,'image_png_sha256':png_hash}
    body={'model':model,'messages':[{'role':'user','content':[
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png_bytes).decode('ascii')}},
        {'type':'text','text':question}]}],
        'max_tokens':max_tokens,'temperature':0,'top_p':1.0,'seed':0,
        'chat_template_kwargs':{'reasoning_effort':'no_think'},'stream':True,'stream_options':{'include_usage':True}}
    raw=encoded(body)
    return {'index':item['index'],'item_id':item['item_id'],'question':question,
        'raw_image_sha256':item['raw_image_sha256'],'image_png_sha256':png_hash,
        'item_sha256':sha256(encoded(item_payload)),'body_sha256':sha256(raw),
        'body_bytes':raw,'body_bytes_count':len(raw),'png_bytes_count':len(png_bytes)}


def poisson_offsets(count,rate,seed):
    require(type(count) is int and count>=1,'arrival count must be positive integer')
    require(type(rate) in (int,float) and math.isfinite(rate) and rate>0,'rate must be finite and positive')
    rng=random.Random(seed);result=[0.0]
    for _ in range(count-1):result.append(result[-1]+rng.expovariate(rate))
    return result


def event_rate(times):
    ordered=sorted(times)
    return (len(ordered)-1)/(ordered[-1]-ordered[0]) if len(ordered)>1 and ordered[-1]>ordered[0] else None


def distribution(values):
    values=sorted(values)
    def percentile(p):
        if not values:return None
        index=(len(values)-1)*p;lower=math.floor(index);upper=math.ceil(index)
        return values[lower]+(values[upper]-values[lower])*(index-lower)
    return {'count':len(values),'mean':math.fsum(values)/len(values) if values else None,
            'p50':percentile(.5),'p95':percentile(.95),'p99':percentile(.99)}


def completion_tokens(row):
    value=(row.get('usage') or {}).get('completion_tokens')
    return value if type(value) is int and value>=0 else None


def is_success(row):
    return (row.get('status')==200 and not row.get('exception') and not row.get('http_error')
        and not row.get('protocol_errors') and row.get('done') is True
        and bool((row.get('content') or '').strip()) and row.get('first_content_s') is not None
        and row.get('finish_reason') in ('stop','length'))


def summarize(records,*,duration_s,target_rps,max_tokens):
    require(math.isfinite(duration_s) and duration_s>0,'positive wall duration required')
    good=[row for row in records if is_success(row)]
    counts={'total':len(records),'successful':len(good),'error_requests':len(records)-len(good),
        'http_error_requests':sum(row.get('status') is not None and row['status']!=200 for row in records),
        'exception_requests':sum(bool(row.get('exception')) for row in records),
        'protocol_error_requests':sum(bool(row.get('protocol_errors')) for row in records),
        'incomplete_stream_requests':sum(row.get('status')==200 and not row.get('done') for row in records),
        'empty_content_requests':sum(row.get('status')==200 and not (row.get('content') or '').strip() for row in records),
        'truncated_requests':sum(row.get('finish_reason')=='length' for row in records),
        'unexpected_finish_reason_requests':sum(row.get('status')==200 and row.get('finish_reason') not in ('stop','length') for row in records),
        'missing_usage_requests':sum(completion_tokens(row) is None for row in records),
        'usage_hit_max_tokens_requests':sum(completion_tokens(row)==max_tokens for row in records),
        'successful_untruncated':sum(row.get('finish_reason')!='length' for row in good)}
    ms=lambda values:distribution([1000*value for value in values])
    tokens=sum(completion_tokens(row) or 0 for row in good)
    tpot=[(row['end_s']-row['first_content_s'])/(completion_tokens(row)-1) for row in good
          if completion_tokens(row) is not None and completion_tokens(row)>1]
    return {'counts':counts,'target_arrivals_rps':target_rps,
        'scheduled_arrivals_rps':event_rate([row['scheduled_arrival_s'] for row in records]),
        'actual_arrivals_rps':event_rate([row['actual_arrival_s'] for row in records]),
        'actual_http_starts_rps':event_rate([row['http_start_s'] for row in records if row.get('http_start_s') is not None]),
        'arrival_rate_definition':'(N-1)/(last-first) across arrival events; null for fewer than2 or zero span; first scheduled arrival at0 then exponential interarrivals',
        'successful_completion_rps':len(good)/duration_s,
        'successful_untruncated_completion_rps':counts['successful_untruncated']/duration_s,
        'successful_output_tokens':tokens,'successful_output_tokens_per_s':tokens/duration_s,
        'all_reported_output_tokens':sum(completion_tokens(row) or 0 for row in records),
        'wall_duration_s':duration_s,'metric_cohort':'successful HTTP200 complete SSE nonempty-content requests; length finishes included and counted as truncated',
        'client_ttft_ms':ms([row['first_content_s']-row['actual_arrival_s'] for row in good]),
        'scheduled_arrival_ttft_ms':ms([row['first_content_s']-row['scheduled_arrival_s'] for row in good]),
        'http_only_ttft_ms':ms([row['first_content_s']-row['http_start_s'] for row in good]),
        'client_e2e_ms':ms([row['end_s']-row['actual_arrival_s'] for row in good]),
        'http_only_e2e_ms':ms([row['end_s']-row['http_start_s'] for row in good]),
        'client_semaphore_queue_ms':ms([row['http_start_s']-row['before_semaphore_s'] for row in records if row.get('http_start_s') is not None]),
        'arrival_scheduler_lag_ms':ms([row['actual_arrival_s']-row['scheduled_arrival_s'] for row in records]),
        'usage_derived_tpot_ms':ms(tpot),
        'tpot_definition':'standard derived (stream_end-first_nonempty_content)/(usage.completion_tokens-1), tokens>1; includes tail/usage overhead; not measured per-token intervals',
        'latency_definition':'client starts at actual arrival before semaphore; HTTP-only starts after semaphore before opening HTTP stream; role/empty chunks never start TTFT',
        'percentile_definition':'linear interpolation at (N-1)*q',
        'http_status_counts':dict(Counter(str(row.get('status')) for row in records)),
        'finish_reason_counts':dict(Counter(str(row.get('finish_reason')) for row in records))}


def load_dataset(data_dir,dataset_manifest):
    import pyarrow.parquet as pq
    raw=dataset_manifest.read_bytes();manifest=json.loads(raw)
    require(manifest.get('dataset_revision')==REVISION,'dataset revision mismatch')
    expected={str(record['file']).replace('\\','/').split('/')[-1]:record for record in manifest['files']}
    files=sorted(data_dir.glob('*.parquet'));require(bool(files),'no parquet source files')
    rows=[];sources=[]
    for path in files:
        require(path.name in expected,'parquet not in frozen dataset manifest')
        actual=file_sha256(path)
        require(actual==expected[path.name]['sha256'],'dataset file SHA256 mismatch: '+str(path))
        sources.append({'file':str(path.resolve()),'bytes':path.stat().st_size,'sha256':actual})
        rows.extend(pq.read_table(path).to_pylist())
    require(len(rows)==765,'expected complete765-row RealWorldQA dataset')
    return rows,{'dataset_revision':REVISION,'dataset_manifest_file':str(dataset_manifest.resolve()),
                 'dataset_manifest_sha256':sha256(raw),'parquet_files':sources}


def prepare_requests(items,*,model,max_tokens,endpoint,run_id):
    from PIL import Image
    prepared=[]
    for ordinal,item in enumerate(items):
        with Image.open(io.BytesIO(item['image_bytes'])) as original:
            image=original.convert('RGB');buffer=io.BytesIO();image.save(buffer,format='PNG')
            size=list(image.size)
        frozen=freeze_request(item,buffer.getvalue(),model=model,max_tokens=max_tokens)
        headers={'content-type':'application/json','x-hy4-bench-case':'vl-arrival-'+run_id,
                 'x-request-id':'vl-arrival-'+run_id+'-'+str(ordinal)}
        frozen.update(ordinal=ordinal,image_size=size,headers=headers)
        frozen['request_sha256']=sha256(encoded({'method':'POST','url':endpoint,'headers':headers,'body_sha256':frozen['body_sha256']}))
        prepared.append(frozen)
    return prepared


def atomic_progress(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


async def run_arrivals(prepared,offsets,*,endpoint,concurrency,target_rps,max_tokens,output_dir):
    import httpx
    limits=httpx.Limits(max_connections=concurrency,max_keepalive_connections=1,keepalive_expiry=1.0)
    transport=httpx.AsyncHTTPTransport(retries=0,limits=limits,trust_env=False)
    timeout=httpx.Timeout(300.0,connect=300.0,read=300.0,write=300.0,pool=300.0)
    semaphore=asyncio.Semaphore(concurrency);records=[];finished=asyncio.Event()
    counters={'arrived':0,'http_started':0,'active_http':0,'peak_active_http':0}
    progress_path=output_dir/'progress.json'
    raw_path=output_dir/'responses.jsonl'
    started_unix=time.time()
    async with httpx.AsyncClient(transport=transport,timeout=timeout,trust_env=False) as client:
        with raw_path.open('x',encoding='utf-8') as raw_stream:
            origin=time.perf_counter()
            clock=lambda:time.perf_counter()-origin
            async def progress():
                while not finished.is_set():
                    atomic_progress(progress_path,{'phase':'running','elapsed_s':clock(),'completed':len(records),
                        'total':len(prepared),'counts':dict(counters),
                        'successful':sum(is_success(row) for row in records),'started_unix':started_unix})
                    try:await asyncio.wait_for(finished.wait(),timeout=5)
                    except asyncio.TimeoutError:pass
            async def one(item,scheduled):
                await asyncio.sleep(max(0,scheduled-clock()))
                arrival=clock();counters['arrived']+=1
                row={key:value for key,value in item.items() if key not in ('body_bytes','headers')}
                row.update(scheduled_arrival_s=scheduled,actual_arrival_s=arrival,before_semaphore_s=clock(),
                    http_start_s=None,first_content_s=None,end_s=None,status=None,exception=None,http_error=None,
                    content='',finish_reason=None,usage=None,done=False,protocol_errors=[],raw_sse_events=[],raw_http_body=None)
                parser=SSEParser()
                try:
                    async with semaphore:
                        row['http_start_s']=clock();counters['http_started']+=1;counters['active_http']+=1
                        counters['peak_active_http']=max(counters['peak_active_http'],counters['active_http'])
                        try:
                            async with client.stream('POST',endpoint,content=item['body_bytes'],headers=item['headers']) as response:
                                row['status']=response.status_code
                                row['response_headers']={key:response.headers.get(key) for key in ('content-type','x-request-id')}
                                if response.status_code!=200:
                                    row['raw_http_body']=(await response.aread()).decode('utf-8',errors='replace')
                                    row['http_error']='HTTP '+str(response.status_code)
                                else:
                                    async for line in response.aiter_lines():parser.feed_line(line,clock())
                                    parser.finish(clock())
                        finally:
                            row['end_s']=clock();counters['active_http']-=1
                except Exception as error:
                    row['exception']={'type':type(error).__name__,'message':str(error)}
                    row['end_s']=clock()
                    parser.finish(row['end_s'])
                row.update(first_content_s=parser.first_content_s,content=parser.content,finish_reason=parser.finish_reason,
                    usage=parser.usage,done=parser.done,protocol_errors=parser.errors,raw_sse_events=parser.raw_events)
                row['success']=is_success(row)
                records.append(row)
                raw_stream.write(json.dumps(row,ensure_ascii=False)+'\n');raw_stream.flush()
            reporter=asyncio.create_task(progress())
            try:await asyncio.gather(*(one(item,scheduled) for item,scheduled in zip(prepared,offsets)))
            finally:
                finished.set();await reporter
            duration=max(row['end_s'] for row in records)
    summary=summarize(records,duration_s=duration,target_rps=target_rps,max_tokens=max_tokens)
    summary.update(started_unix=started_unix,finished_unix=time.time(),peak_active_http=counters['peak_active_http'],
        responses_sha256=file_sha256(raw_path),arrival_schedule_sha256=sha256(encoded(offsets)))
    atomic_progress(progress_path,{'phase':'complete','completed':len(records),'total':len(prepared),
        'counts':dict(counters),'wall_duration_s':duration,'summary':summary})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--dataset-manifest',type=Path)
    parser.add_argument('--corpus',type=Path,required=True,help='Frozen fit64/heldout32 calibration corpus JSON')
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--model',default='hy4-vl-w4a8')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--count',type=int,default=512)
    parser.add_argument('--rps',type=float,default=2.55)
    parser.add_argument('--concurrency',type=int,default=146)
    parser.add_argument('--c1',action='store_true',help='Use concurrency1; combine --count 1 for one explicit trial')
    parser.add_argument('--selection-seed',type=int,default=255)
    parser.add_argument('--arrival-seed',type=int,default=255)
    parser.add_argument('--max-tokens',type=int,default=16)
    parser.add_argument('--server-options-json',type=Path,help='Optional frozen launch/config evidence to include verbatim')
    args=parser.parse_args()
    if args.c1:args.concurrency=1
    require(args.concurrency>=1,'concurrency must be>=1')
    require(not args.output_dir.exists(),'refuse existing output directory')
    offsets=poisson_offsets(args.count,args.rps,args.arrival_seed)
    corpus_raw=args.corpus.read_bytes();corpus=json.loads(corpus_raw)
    require(corpus.get('dataset_revision')==REVISION,'calibration corpus revision mismatch')
    require(len(corpus.get('groups',{}).get('fit',[]))==64 and len(corpus.get('groups',{}).get('heldout',[]))==32,
            'complete fit64 and heldout32 exclusions required')
    rows,dataset_info=load_dataset(args.data_dir,args.dataset_manifest or args.data_dir.parent/'MANIFEST.json')
    selected,pool_info=select_items(rows,corpus,count=args.count,seed=args.selection_seed)
    base=args.base_url.rstrip('/');endpoint=base+'/chat/completions' if base.endswith('/v1') else base+'/v1/chat/completions'
    run_id=uuid.uuid4().hex
    prepared=prepare_requests(selected,model=args.model,max_tokens=args.max_tokens,endpoint=endpoint,run_id=run_id)
    del rows,selected
    for item,offset in zip(prepared,offsets):item['scheduled_arrival_s']=offset
    versions={name:importlib.metadata.version(name) for name in ('httpx','Pillow','pyarrow')}
    server_options=None
    if args.server_options_json:
        raw=args.server_options_json.read_bytes()
        server_options={'file':str(args.server_options_json.resolve()),'sha256':sha256(raw),'content':json.loads(raw)}
    manifest={'run_id':run_id,'source_sha256':file_sha256(__file__),'prepared_unix':time.time(),
        'source_corpus_file':str(args.corpus.resolve()),'source_corpus_sha256':sha256(corpus_raw),
        'dataset':dataset_info,'pool':pool_info,'selected_ids':[item['item_id'] for item in prepared],
        'requests':[{key:value for key,value in item.items() if key!='body_bytes'} for item in prepared],
        'arrival_offsets_s':offsets,'arrival_schedule_sha256':sha256(encoded(offsets)),
        'options':{'count':args.count,'target_arrivals_rps':args.rps,'concurrency':args.concurrency,
            'selection_seed':args.selection_seed,'arrival_seed':args.arrival_seed,'generation_seed':0,
            'model':args.model,'endpoint':endpoint,'max_tokens':args.max_tokens,'temperature':0,'top_p':1.0,
            'reasoning_effort':'no_think','stream':True,'include_usage':True,'trust_env':False,
            'transport_retries':0,'max_keepalive_connections':1,'keepalive_expiry_s':1.0,
            'network_timeouts_s':{'connect':300,'read':300,'write':300,'pool':300},
            'warmup_requests':0,'prefix_cache_reset_requests':0,
            'prefix_cache_precondition':'root guarantees server prefix caching off; no verification/warmup/reset request made',
            'prompt':'original question.strip(); image-first RGB PNG; no labels or hints',
            'image_client_processing':'RGB conversion and PNG encoding only; original dimensions; all bodies serialized before timing'},
        'server_options':server_options,'client_versions':versions,
        'scope':'fixed real-image arrival workload; all preparation excluded from timed origin; no answer labels used'}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest_raw=encoded(manifest)
    with (args.output_dir/'MANIFEST.json').open('xb') as stream:stream.write(manifest_raw)
    atomic_progress(args.output_dir/'progress.json',{'phase':'prepared','total':len(prepared),'manifest_sha256':sha256(manifest_raw)})
    summary=asyncio.run(run_arrivals(prepared,offsets,endpoint=endpoint,concurrency=args.concurrency,
        target_rps=args.rps,max_tokens=args.max_tokens,output_dir=args.output_dir))
    summary['manifest_sha256']=sha256(manifest_raw)
    with (args.output_dir/'RESULT.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
