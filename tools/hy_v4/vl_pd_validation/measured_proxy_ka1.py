"""Official Mooncake proxy with phase timing; transport and scheduling unchanged."""
import contextvars
import importlib.util
import json
import os
from pathlib import Path
import time

from fastapi import Request
from fastapi.responses import Response
import uvicorn

B = Path('/workspace/hy4_mm_20261108/pd-r16-inspect-v1')
spec = importlib.util.spec_from_file_location('official_mooncake_proxy', B / 'mooncake_connector_proxy_ka1.py')
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
context = contextvars.ContextVar('hy4_request', default={})
events = (B / 'proxy-requests.jsonl').open('a', buffering=1)

def emit(event, request_id=None, **data):
    row = dict(context.get())
    row.update(event=event, request_id=request_id, wall_time=time.time(), perf_time=time.perf_counter(), **data)
    events.write(json.dumps(row) + '\n')

original_handle = proxy._handle_completions
original_prefill = proxy.send_request_to_service
original_decode = proxy.stream_service_response

async def tagged_handle(api, request):
    payload = await request.json()
    context.set(dict(case=request.headers.get('x-hy4-bench-case', 'smoke'), client_request_id=request.headers.get('x-request-id'), arrival_wall=time.time(), arrival_perf=time.perf_counter(), requested_output_tokens=payload.get('max_tokens')))
    return await original_handle(api, request)

async def timed_prefill(client_info, dp_rank, endpoint, req_data, request_id):
    start = time.perf_counter()
    emit('p_start', request_id)
    try:
        await original_prefill(client_info, dp_rank, endpoint, req_data, request_id)
        emit('p_done', request_id, elapsed_ms=1000 * (time.perf_counter() - start))
    except Exception as error:
        emit('p_error', request_id, error=repr(error))
        raise

async def timed_decode(prefill_client_info, prefill_dp_rank, decode_client_info, endpoint, req_data, request_id):
    start = time.perf_counter()
    first = True
    emit('d_start', request_id)
    try:
        async for chunk in original_decode(prefill_client_info, prefill_dp_rank, decode_client_info, endpoint, req_data, request_id):
            if first:
                emit('d_first_bytes', request_id, elapsed_ms=1000 * (time.perf_counter() - start))
                first = False
            yield chunk
        emit('d_done', request_id, elapsed_ms=1000 * (time.perf_counter() - start))
    except Exception as error:
        emit('d_error', request_id, error=repr(error))
        raise

proxy._handle_completions = tagged_handle
proxy.send_request_to_service = timed_prefill
proxy.stream_service_response = timed_decode

@proxy.app.api_route('/metrics', methods=['GET'])
@proxy.app.api_route('/v1/models', methods=['GET'])
@proxy.app.api_route('/tokenize', methods=['POST'])
@proxy.app.api_route('/detokenize', methods=['POST'])
async def forward_auxiliary(request: Request):
    client = request.app.state.decode_clients[0]['client']
    response = await client.request(request.method, request.url.path, content=await request.body(), headers={'content-type': request.headers.get('content-type', 'application/json')})
    return Response(content=response.content, status_code=response.status_code, headers={'content-type': response.headers.get('content-type', 'application/json')})

@proxy.app.get('/health')
async def health(request: Request):
    if not request.app.state.ready.is_set():
        return Response(status_code=503)
    verified = []
    try:
        for info in request.app.state.prefill_clients:
            response = await info['client'].get(info['bootstrap_addr'] + '/query')
            response.raise_for_status()
            current = {int(rank): entry['engine_id'] for rank, entry in response.json().items()}
            cached = info['dp_engine_id']
            if current != cached:
                return Response(content='Producer engine_id changed; restart proxy', status_code=503)
            verified.append(dict(url=info['url'], engine_ids=current))
    except Exception:
        return Response(status_code=503)
    (B / 'proxy-engine-validation.json').write_text(json.dumps(dict(verified_at=time.time(), prefillers=verified), indent=2) + '\n')
    response = await request.app.state.decode_clients[0]['client'].get('/health')
    return Response(status_code=response.status_code)


# Record the actual instantiated pools; model requests are never retried.
from contextlib import asynccontextmanager
original_lifespan = proxy.app.router.lifespan_context
@asynccontextmanager
async def transport_lifespan(app):
    async with original_lifespan(app):
        import httpx,httpcore
        pools={role:[{'keepalive_expiry':x['client']._transport._pool._keepalive_expiry,
                      'retries':x['client']._transport._pool._retries}
                     for x in infos]
               for role,infos in [('P',app.state.prefill_clients),('D',app.state.decode_clients)]}
        assert all(x['keepalive_expiry']==1.0 and x['retries']==0 for xs in pools.values() for x in xs)
        proof={'pid':os.getpid(),'pools':pools,'httpx':httpx.__version__,'httpcore':httpcore.__version__,'verified_at':time.time()}
        (B/'proxy-transport-ka1.json').write_text(json.dumps(proof,indent=2))
        yield
proxy.app.router.lifespan_context=transport_lifespan

if __name__ == '__main__':
    proxy.global_args = proxy.parse_args()
    uvicorn.run(proxy.app, host=proxy.global_args.host, port=proxy.global_args.port, access_log=False)
