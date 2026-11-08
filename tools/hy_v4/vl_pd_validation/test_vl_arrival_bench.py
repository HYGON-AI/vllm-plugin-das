"""Offline tests only: no HTTP clients, network, dataset decode, or GPU use."""
import asyncio
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('vl_arrival_bench',HERE/'vl_arrival_bench.py')
bench=None
if Path(spec.origin).exists():
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

class ArrivalBenchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(bench,'frozen real-image arrival benchmark operator required')

    def event(self,parser,value,t):
        parser.feed_line('data: '+(value if isinstance(value,str) else json.dumps(value)),t)
        parser.feed_line('',t)

    def test_sse_role_and_empty_chunks_do_not_start_ttft_and_usage_survives(self):
        parser=bench.SSEParser()
        self.event(parser,{'choices':[{'index':0,'delta':{'role':'assistant','content':''}}]},1)
        self.assertIsNone(parser.first_content_s)
        self.event(parser,{'choices':[{'index':0,'delta':{'content':''}}]},2)
        self.event(parser,{'choices':[{'index':0,'delta':{'content':'hello'}}]},3)
        self.event(parser,{'choices':[{'index':0,'delta':{'content':' world'}}]},4)
        self.event(parser,{'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]},5)
        self.event(parser,{'choices':[],'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}},6)
        self.event(parser,'[DONE]',7)
        parser.finish(7)
        self.assertEqual(parser.first_content_s,3)
        self.assertEqual(parser.content,'hello world')
        self.assertEqual(parser.finish_reason,'stop')
        self.assertEqual(parser.usage['completion_tokens'],2)
        self.assertTrue(parser.done)
        self.assertEqual(parser.errors,[])

    def test_sse_comments_multiline_data_errors_and_incomplete_stream(self):
        parser=bench.SSEParser()
        parser.feed_line(': heartbeat',1)
        parser.feed_line('data: {"choices":',2)
        parser.feed_line('data: [{"delta":{"content":" "},"index":0}]}',2)
        parser.feed_line('',2)
        self.assertEqual(parser.first_content_s,2)
        self.event(parser,'not-json',3)
        self.event(parser,{'error':{'message':'owned failure'}},4)
        parser.finish(5)
        self.assertFalse(parser.done)
        self.assertEqual(len(parser.errors),2)
        self.assertEqual(parser.content,' ')

    def test_pool_excludes_fit_heldout_ids_hash_aliases_and_duplicate_images(self):
        images=[b'fit',b'heldout',b'fit',b'C',b'C',b'D',b'E']
        rows=[{'question':' question '+str(i)+' ','image':{'bytes':image},'answer':'DO_NOT_SEND'} for i,image in enumerate(images)]
        corpus={'groups':{'fit':[0],'heldout':[1]},'items':{
            'realworldqa-0':{'raw_image_sha256':hashlib.sha256(b'fit').hexdigest()},
            'realworldqa-1':{'raw_image_sha256':hashlib.sha256(b'heldout').hexdigest()}}}
        selected,info=bench.select_items(rows,corpus,count=3,seed=255)
        self.assertEqual({x['index'] for x in selected},{3,5,6})
        self.assertEqual(len({x['raw_image_sha256'] for x in selected}),3)
        self.assertEqual(info['unique_available'],3)
        with self.assertRaises(ValueError):bench.select_items(rows,corpus,count=4,seed=255)
        with self.assertRaises(ValueError):bench.select_items(rows,corpus,count=True,seed=255)
        bad=copy.deepcopy(corpus);bad['items']['realworldqa-0']['raw_image_sha256']='0'*64
        with self.assertRaises(ValueError):bench.select_items(rows,bad,count=1,seed=255)

    def test_body_frozen_before_timing_image_first_and_original_question_only(self):
        item={'index':8,'item_id':'realworldqa-8','question':'original question',
              'raw_image_sha256':'a'*64,'image_bytes':b'raw-image'}
        frozen=bench.freeze_request(item,b'fixed-png-bytes',model='hy4-vl-w4a8',max_tokens=16)
        body=json.loads(frozen['body_bytes'])
        content=body['messages'][0]['content']
        self.assertEqual([part['type'] for part in content],['image_url','text'])
        self.assertEqual(content[1]['text'],'original question')
        self.assertEqual(body['chat_template_kwargs'],{'reasoning_effort':'no_think'})
        self.assertTrue(body['stream_options']['include_usage'])
        self.assertEqual((body['temperature'],body['seed'],body['max_tokens']),(0,0,16))
        digest=frozen['body_sha256'];item['question']='changed later'
        self.assertEqual(hashlib.sha256(frozen['body_bytes']).hexdigest(),digest)
        self.assertEqual(json.loads(frozen['body_bytes'])['messages'][0]['content'][1]['text'],'original question')
        self.assertEqual(frozen['image_png_sha256'],hashlib.sha256(b'fixed-png-bytes').hexdigest())

    def test_poisson_offsets_and_arrival_rate_use_event_intervals(self):
        self.assertEqual(bench.poisson_offsets(1,2.55,255),[0.0])
        offsets=bench.poisson_offsets(10000,2.55,255)
        self.assertTrue(all(b>a for a,b in zip(offsets,offsets[1:])))
        self.assertAlmostEqual((offsets[-1]-offsets[0])/9999,1/2.55,delta=0.015)
        self.assertAlmostEqual(bench.event_rate([0,1,3]),2/3)
        self.assertIsNone(bench.event_rate([3]))
        for rate in (0,-1,float('inf')):
            with self.assertRaises(ValueError):bench.poisson_offsets(2,rate,255)

    def test_metrics_separate_client_queue_http_ttft_and_completion_throughput(self):
        row=successful_row()
        report=bench.summarize([row],duration_s=10,target_rps=2.55,max_tokens=16)
        self.assertEqual(report['successful_completion_rps'],0.1)
        self.assertEqual(report['successful_output_tokens_per_s'],0.3)
        self.assertEqual(report['client_ttft_ms']['mean'],4000)
        self.assertEqual(report['http_only_ttft_ms']['mean'],2000)
        self.assertEqual(report['client_e2e_ms']['mean'],8000)
        self.assertEqual(report['usage_derived_tpot_ms']['mean'],2000)
        self.assertIn('not measured per-token',report['tpot_definition'])
        self.assertAlmostEqual(bench.distribution([1,2,3,4])['p95'],3.85)

    def test_async_arrivals_queue_independently_of_c1_stream_and_log_all_results(self):
        class Response:
            status_code=200
            headers={'content-type':'text/event-stream'}
            async def __aenter__(self):return self
            async def __aexit__(self,*args):return None
            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}'
                yield ''
                await asyncio.sleep(.01)
                yield 'data: {"choices":[{"delta":{"content":"answer"},"index":0,"finish_reason":"stop"}],"usage":{"completion_tokens":2}}'
                yield ''
                yield 'data: [DONE]'
                yield ''
        class Client:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):return None
            def stream(self,method,endpoint,*,content,headers):
                if method!='POST' or content!=b'frozen-body':raise AssertionError('body changed or reserialized')
                return Response()
        fake=SimpleNamespace(AsyncClient=Client,Limits=lambda **kw:kw,
            AsyncHTTPTransport=lambda **kw:kw,Timeout=lambda *args,**kw:kw)
        prepared=[{'ordinal':i,'body_bytes':b'frozen-body','headers':{},'item_id':str(i)} for i in range(2)]
        with tempfile.TemporaryDirectory() as folder,patch.dict('sys.modules',{'httpx':fake}):
            output=Path(folder)
            result=asyncio.run(bench.run_arrivals(prepared,[0,0],endpoint='http://unused.invalid/v1/chat/completions',
                concurrency=1,target_rps=2.55,max_tokens=16,output_dir=output))
            records=[json.loads(line) for line in (output/'responses.jsonl').read_text().splitlines()]
            by_id={row['item_id']:row for row in records}
            self.assertEqual(result['counts']['successful'],2)
            self.assertEqual(result['peak_active_http'],1)
            self.assertLess(by_id['1']['actual_arrival_s'],by_id['0']['end_s'])
            self.assertGreater(by_id['1']['http_start_s'],by_id['1']['before_semaphore_s'])
            self.assertEqual(json.loads((output/'progress.json').read_text())['phase'],'complete')

    def test_errors_empty_length_and_missing_usage_are_all_accounted(self):
        good=successful_row()
        empty=successful_row();empty.update(content='',first_content_s=None,finish_reason='length')
        failed=successful_row();failed.update(status=500,content='',first_content_s=None,done=False,http_error='HTTP 500')
        report=bench.summarize([good,empty,failed],duration_s=10,target_rps=2.55,max_tokens=16)
        self.assertEqual(report['counts']['successful'],1)
        self.assertEqual(report['counts']['error_requests'],2)
        self.assertEqual(report['counts']['http_error_requests'],1)
        self.assertEqual(report['counts']['empty_content_requests'],1)
        self.assertEqual(report['counts']['truncated_requests'],1)
        self.assertEqual(report['client_ttft_ms']['count'],1)
        missing=successful_row();missing['usage']=None
        report=bench.summarize([missing],duration_s=10,target_rps=2.55,max_tokens=16)
        self.assertEqual(report['counts']['missing_usage_requests'],1)
        self.assertEqual(report['usage_derived_tpot_ms']['count'],0)


def successful_row():
    return {'scheduled_arrival_s':0,'actual_arrival_s':1,'before_semaphore_s':1,'http_start_s':3,
        'first_content_s':5,'end_s':9,'status':200,'content':'yes','finish_reason':'stop',
        'usage':{'prompt_tokens':10,'completion_tokens':3},'done':True,'protocol_errors':[],
        'exception':None,'http_error':None}

if __name__=='__main__':unittest.main()
