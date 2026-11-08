"""Untimed calibration requests from a frozen label-free image manifest."""
import argparse,base64,hashlib,io,json,time
from pathlib import Path
import requests
import pyarrow.parquet as pq
from PIL import Image

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--corpus',type=Path,required=True);parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--group',choices=['fit','heldout'],required=True);parser.add_argument('--base-url',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prefill-only',action='store_true');args=parser.parse_args()
    assert not args.output.exists();args.output.mkdir(parents=True)
    raw=args.corpus.read_bytes();manifest=json.loads(raw)
    rows=[row for file in sorted(args.dataset.glob('*.parquet')) for row in pq.read_table(file).to_pylist()];assert len(rows)==765
    session=requests.Session();session.trust_env=False;result=[]
    with (args.output/'responses.jsonl').open('x') as f:
        for index in manifest['groups'][args.group]:
            item=manifest['items']['realworldqa-'+str(index)];row=rows[index]
            assert hashlib.sha256(row['image']['bytes']).hexdigest()==item['raw_image_sha256']
            image=Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB');buffer=io.BytesIO();image.save(buffer,format='PNG');image_bytes=buffer.getvalue()
            assert hashlib.sha256(image_bytes).hexdigest()==item['image_png_sha256']
            assert row['question'].strip()==item['question']
            body={'model':'hy4-vl-w4a8','messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image_bytes).decode()}},{'type':'text','text':item['question']}]}],**manifest['generation']}
            if args.prefill_only:body['max_tokens']=1
            started=time.monotonic();response=session.post(args.base_url.rstrip('/')+'/v1/chat/completions',json=body,headers={'x-hy4-bench-case':'vl-calibration-'+args.group,'x-request-id':'vl-'+args.group+'-'+str(index)},timeout=300)
            row_result={'index':index,'item_sha256':item['item_sha256'],'image_sha256':item['image_png_sha256'],'status':response.status_code,'response':response.json(),'seconds':time.monotonic()-started}
            f.write(json.dumps(row_result,ensure_ascii=False)+'\n');f.flush();result.append(row_result)
            assert response.status_code==200,row_result
            choice=row_result['response']['choices'][0]
            if args.prefill_only:
                usage=row_result['response']['usage'];assert usage['prompt_tokens']>0 and usage['completion_tokens']<=1,row_result
            else:assert choice.get('message',{}).get('content'),row_result
            print(json.dumps({'group':args.group,'done':len(result),'total':len(manifest['groups'][args.group]),'status':response.status_code}),flush=True)
    value={'group':args.group,'requests':len(result),'http_errors':sum(x['status']!=200 for x in result),'corpus_sha256':hashlib.sha256(raw).hexdigest(),'answers_used':False,'timed_benchmark':False,'prefill_only':args.prefill_only,'P_requested_max_tokens':1 if args.prefill_only else manifest['generation']['max_tokens'],'usage':[x['response'].get('usage') for x in result]}
    (args.output/'RESULT.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
if __name__=='__main__':main()
