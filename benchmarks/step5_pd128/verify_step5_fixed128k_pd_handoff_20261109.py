"""Verify a 128K handoff through the same proxy used by the SLO run."""
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request


def through_eos(tokens, eos_token_id):
    return tokens[:tokens.index(eos_token_id)+1] if eos_token_id in tokens else tokens


def verify(groups, endpoint, output_dir, port, model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    eos = tokenizer.eos_token_id
    assert eos is not None
    marker = 'UniqueContentMarker729136'
    rendered = tokenizer.apply_chat_template([{'role':'user','content':marker}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False, thinking_mode=False)
    head, tail = rendered.split(marker)
    head = tokenizer.encode(head, add_special_tokens=False)
    tail = tokenizer.encode(tail+'</think>', add_special_tokens=False)
    filler = tokenizer.encode('Reference note: the background records contain ordinary neutral information. ',
                              add_special_tokens=False)

    def ask(base, prompt, count, extra_headers):
        payload = json.dumps({'model':'step5','prompt':prompt,'max_tokens':count,'temperature':0,
                              'ignore_eos':True,'stream':False,'return_token_ids':True}).encode()
        request = urllib.request.Request(base+'/v1/completions', data=payload,
                                         headers={'Content-Type':'application/json', **extra_headers})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=600) as response:
            result = json.load(response)
        tokens = result['choices'][0]['token_ids']
        assert len(tokens)==count and result['usage']['prompt_tokens']==131072
        return tokens

    def group_check(item):
        group, word = item
        suffix = tokenizer.encode('\nIgnore the reference notes. Reply with exactly the word '+word+
                                  ' and nothing else.', add_special_tokens=False)
        room = 131072-len(head)-len(tail)-len(suffix)
        prompt = head+(filler*((room+len(filler)-1)//len(filler)))[:room]+suffix+tail
        reference = ask(group['url'], prompt, 1, {'X-data-parallel-rank':'0'})
        outputs = [ask(endpoint, prompt, 8, {'X-Step5-Validation-Group':group['name']}) for _ in range(2)]
        natural = [through_eos(tokens, eos) for tokens in outputs]
        record = {'input':len(prompt), 'prefill_first_token':reference, 'actual':outputs,
                  'eos_token_id':eos, 'through_eos':natural,
                  'exact':natural[0]==natural[1] and all(row[:1]==reference for row in natural)}
        (output_dir/f"golden-{port}-{group['name']}.json").write_text(json.dumps([record],indent=2))
        assert record['exact'], (group['name'],record)
        return group['name'], {'prompts':1,'pd_runs':2,'exact':True}

    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(pool.map(group_check, zip(groups, ('GREEN','RED','BLUE'))))
