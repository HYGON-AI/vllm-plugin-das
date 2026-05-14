# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import threading
import uuid

import aiohttp
import msgpack
import zmq
from typing import Any
from quart import Quart, make_response, request
from dataclasses import dataclass, field
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
import time
import asyncio
from collections import deque, defaultdict
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class Request:
    request_id: str
    p_http_address: str = ""
    p_dp_rank: int = -1
    d_http_address: str = ""
    d_dp_rank: int = -1

@dataclass
class Instance:
    ins_type: str = "P"
    http_address: str = ""
    zmq_address: str = ""
    p_unique_id: bytes = b""
    dp_size: int = 0
    pp_size: int = 0
    tp_size: int = 0
    # [dp, pp, tp] : zmq_address
    rank_table: dict[int, dict[int, dict[int, str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    # [dp, pp, tp] : global rank
    comm_rank_table: dict[int, dict[int, dict[int, int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )

    def count_rank_table_elements(self):
        count = 0
        for first_dict in self.rank_table.values():
            for second_dict in first_dict.values():
                count += len(second_dict)
        return count

    def is_ready(self):
        world_size = self.dp_size * self.pp_size * self.tp_size
        inited_rank = self.count_rank_table_elements()
        all_ranks_ready = world_size and inited_rank == world_size
        if self.ins_type == "P" :
            logger.info(f"""[Router] P is_ready? : {self.http_address} world_size = {world_size} inited_rank = {inited_rank}""")
            return all_ranks_ready
        else :
            logger.info(f"""[Router] D is_ready? : {self.http_address} world_size = {world_size} inited_rank = {inited_rank}""")
            return all_ranks_ready 

count = 0
prefill_instances: dict[str, Instance] = {} 
decode_instances: dict[str, Instance] = {} 
running_requests: dict[str, Request] = {}
healthy_instances: dict[str, float] = {}

pending_prefill_ins: list[str] = []
pending_decode_ins: list[str] = []
ready_prefill_ins: list[str] = []
ready_decode_ins: list[str] = []

pd_pair : dict[str, bytes] = {}
router_nccl = NCCLLibrary()

instance_cv = threading.Condition()

request_cv = threading.Condition()
health_cv = threading.Condition()

request_queue_cv = threading.Condition()
request_queue: deque[list[Any]] = deque()
sock_cache : dict[str, Any] = {} 

def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            # data: {"type": "P", "http_address": "ip:port",
            #        "zmq_address": "ip:port"}
            data = msgpack.loads(message)
            global prefill_instances
            global instance_cv
            global decode_instances
            if data["type"] == "P":
                 with instance_cv:
                    if data["http_address"] not in prefill_instances:
                        prefill_instances[data["http_address"]] = Instance(http_address=data["http_address"])
                    p_instance = prefill_instances[data["http_address"]]
                    p_instance.rank_table[int(data["dp_rank"])][int(data["pp_rank"])][int(data["tp_rank"])] = data["zmq_address"]
                    if p_instance.is_ready():
                        pending_prefill_ins.append(p_instance.http_address)
                        logger.info(f"""[Router] pending_prefill_ins appended {p_instance.http_address} ZMQ:{p_instance.zmq_address}""")
                        instance_cv.notify()
                    logger.info(f"""[Router] add P rank [{data["dp_rank"]}, {data["pp_rank"]}, {data["tp_rank"]}] : {data["zmq_address"]}""")
            elif data["type"] == "D":
                with instance_cv:
                    if data["http_address"] not in decode_instances:
                        decode_instances[data["http_address"]] = Instance(ins_type="D", http_address=data["http_address"])
                    d_instance = decode_instances[data["http_address"]]
                    d_instance.rank_table[int(data["dp_rank"])][int(data["pp_rank"])][int(data["tp_rank"])] = data["zmq_address"]
                    if d_instance.is_ready():
                        pending_decode_ins.append(d_instance.http_address)
                        logger.info(f"""[Router] pending_decode_ins appended {d_instance.http_address} ZMQ:{d_instance.zmq_address}""")
                        instance_cv.notify()
                    logger.info(f"""[Router] add D rank [{data["dp_rank"]}, {data["pp_rank"]}, {data["tp_rank"]}] : {data["zmq_address"]}""")
            elif data["type"] == "P_init":
                with instance_cv:
                    if data["http_address"] not in prefill_instances:
                        prefill_instances[data["http_address"]] = Instance(http_address=data["http_address"], dp_size=int(data["dp_size"]), pp_size=int(data["pp_size"]), tp_size=int(data["tp_size"]))
                        prefill_instances[data["http_address"]].zmq_address = data["zmq_address"]
                        continue
                    p_instance = prefill_instances[data["http_address"]]
                    p_instance.dp_size=int(data["dp_size"])
                    p_instance.pp_size=int(data["pp_size"])
                    p_instance.tp_size=int(data["tp_size"])
                    p_instance.zmq_address=data["zmq_address"]
                    if p_instance.is_ready():
                        pending_prefill_ins.append(p_instance.http_address)
                        logger.info(f"""[Router] pending_prefill_ins appended {p_instance.http_address} ZMQ:{p_instance.zmq_address}""")
                        instance_cv.notify()
            elif data["type"] == "D_init":
                with instance_cv:
                    if data["http_address"] not in decode_instances:
                        decode_instances[data["http_address"]] = Instance(ins_type="D", http_address=data["http_address"], dp_size=int(data["dp_size"]), pp_size=int(data["pp_size"]), tp_size=int(data["tp_size"]))
                        decode_instances[data["http_address"]].zmq_address = data["zmq_address"]
                        continue
                    d_instance = decode_instances[data["http_address"]]
                    d_instance.dp_size=int(data["dp_size"])
                    d_instance.pp_size=int(data["pp_size"])
                    d_instance.tp_size=int(data["tp_size"])
                    d_instance.zmq_address=data["zmq_address"]
                    if d_instance.is_ready():
                        pending_decode_ins.append(d_instance.http_address)
                        logger.info(f"""[Router] pending_decode_ins appended {d_instance.http_address} ZMQ:{d_instance.zmq_address}""")
                        instance_cv.notify()
            elif data["type"] == "heartbeat":
                global healthy_instances
                global health_cv
                with health_cv:
                    healthy_instances[data["http_address"]] = time.time()
            elif data["type"] == "Req":
                # logger.info(f"""[Router] recv Request {data["request_id"]} : {data["instance_type"]}""")
                global running_requests
                global request_cv
                with request_cv:
                    if data["request_id"] in running_requests:
                        request = running_requests[data["request_id"]]
                        if data["instance_type"] == "P":
                            request.p_http_address = data["http_address"]
                            request.p_dp_rank = int(data["dp_rank"])
                        elif data["instance_type"] == "D":
                            request.d_http_address = data["http_address"]
                            request.d_dp_rank = int(data["dp_rank"])
                        assert(request.p_dp_rank >= 0 and request.d_dp_rank >=0)
                        with request_queue_cv:
                            request_queue.append(request)
                            # logger.info(f"""[Router] add Request {data["request_id"]} [{request.p_http_address}:{request.p_dp_rank}, {request.d_http_address}:{request.d_dp_rank}]""")
                            request_queue_cv.notify()
                    else:
                        if data["instance_type"] == "P":
                            running_requests[data["request_id"]] = Request(request_id=data["request_id"], p_http_address=data["http_address"], p_dp_rank=int(data["dp_rank"]))
                        elif data["instance_type"] == "D":
                            running_requests[data["request_id"]] = Request(request_id=data["request_id"], d_http_address=data["http_address"], d_dp_rank=int(data["dp_rank"]))

            else:
                print(
                    "Unexpected, Received message from %s, data: %s",
                    remote_address,
                    data,
                )

zmq_context = None
tp_mapping_of_pd_pair : dict[str, dict[int, list[str]]] = {}
tp_comm_mapping_of_pd_pair : dict[str, dict[int, list[int]]] = {}
active_p_tp_rank_of_pd_pair : dict[str, set[int]] = {}

def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")
    
    # context = zmq.Context()
    # router_socket = context.socket(zmq.ROUTER)
    global zmq_context
    zmq_context = zmq.Context()
    router_socket = zmq_context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    _listener_thread = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    _listener_thread.start()
    return _listener_thread

def dispatch_to_P(request : Request):
    global prefill_instances
    global decode_instances
    p_ins = prefill_instances[request.p_http_address]
    d_ins = decode_instances[request.d_http_address]

    global zmq_context
    global sock_cache

    pd_pair_id = p_ins.http_address + "_" + d_ins.http_address
    p_dp_rank = request.p_dp_rank
    d_dp_rank = request.d_dp_rank
    tp_dst_id = pd_pair_id + "_" + str(d_dp_rank)
    assert(d_ins.pp_size == 1)
    d_pp_rank = 0

    global tp_mapping_of_pd_pair
    global tp_comm_mapping_of_pd_pair
    global active_p_tp_rank_of_pd_pair

    if tp_dst_id not in active_p_tp_rank_of_pd_pair:
        p_active_tp_rank = set()
        p_tp_rank_to_dst : dict[int, list[str]] = defaultdict(list)
        p_tp_rank_to_dst_comm : dict[int, list[int]] = defaultdict(list)
        for d_tp_rank in range(d_ins.tp_size):
            p_tp_rank = d_tp_rank % p_ins.tp_size
            p_active_tp_rank.add(p_tp_rank)
            p_tp_rank_to_dst[p_tp_rank].append(d_ins.rank_table[d_dp_rank][d_pp_rank][d_tp_rank])
            p_tp_rank_to_dst_comm[p_tp_rank].append(d_ins.comm_rank_table[d_dp_rank][d_pp_rank][d_tp_rank])
        tp_mapping_of_pd_pair[tp_dst_id] = p_tp_rank_to_dst
        tp_comm_mapping_of_pd_pair[tp_dst_id] = p_tp_rank_to_dst_comm
        active_p_tp_rank_of_pd_pair[tp_dst_id] = p_active_tp_rank
    
    p_active_tp_rank = active_p_tp_rank_of_pd_pair[tp_dst_id]
    p_tp_rank_to_dst = tp_mapping_of_pd_pair[tp_dst_id]
    p_tp_rank_to_dst_comm = tp_comm_mapping_of_pd_pair[tp_dst_id]

    for p_pp_rank in range(p_ins.pp_size):
        for p_tp_rank in p_active_tp_rank:
            if p_ins.rank_table[p_dp_rank][p_pp_rank][p_tp_rank] not in sock_cache:
                sock = zmq_context.socket(zmq.DEALER)
                sock.setsockopt_string(zmq.IDENTITY, "router")
                sock.connect(f"tcp://{p_ins.rank_table[p_dp_rank][p_pp_rank][p_tp_rank]}")
                sock_cache[p_ins.rank_table[p_dp_rank][p_pp_rank][p_tp_rank]] = sock
            data = {
                "cmd": "req_to_transfer",
                "request_id": request.request_id,
                "dst_num": len(p_tp_rank_to_dst[p_tp_rank]),
                "pd_pair_id": pd_pair_id,
                "remote_address": p_tp_rank_to_dst[p_tp_rank],
                "remote_rank": p_tp_rank_to_dst_comm[p_tp_rank],
            }
            sock_cache[p_ins.rank_table[p_dp_rank][p_pp_rank][p_tp_rank]].send(msgpack.dumps(data))
            logger.info(f"""[Router] dispatch Request {request.request_id} [{p_dp_rank}, {p_pp_rank}, {p_tp_rank}] -> [{d_dp_rank}, {d_pp_rank}]""")
    
    for p_tp_rank in range(p_ins.tp_size):
        if p_tp_rank not in p_active_tp_rank:
            for p_pp_rank in range(p_ins.pp_size):
                if p_ins.rank_table[p_dp_rank][p_pp_rank][p_tp_rank] not in sock_cache:
                    sock = zmq_context.socket(zmq.DEALER)
                    sock.setsockopt_string(zmq.IDENTITY, "router")
                    sock.connect(f"tcp://{p_ins.rank_table[p_dp_rank][p_pp_rank][p_tp_rank]}")
                    sock_cache[p_ins.rank_table[p_dp_rank][p_pp_rank][p_tp_rank]] = sock
                data = {
                    "cmd": "req_not_to_transfer",
                    "request_id": request.request_id,
                }
                sock_cache[p_ins.rank_table[p_dp_rank][p_pp_rank][p_tp_rank]].send(msgpack.dumps(data))

def dp_dispatch():
    global request_queue_cv
    global request_queue
    while True:
        with request_queue_cv:
            while not request_queue:
                request_queue_cv.wait()
            request = request_queue.pop()
        dispatch_to_P(request)

def start_dp_dispatch():
    _thread = threading.Thread(
        target=dp_dispatch, daemon=True
    )
    _thread.start()
    return _thread

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                if True:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


def unique_id_dispatch(prefill_instance : Instance,
                       decode_instance : Instance) :
    global zmq_context
    global sock_cache
    global router_nccl
    global pd_pair

    pd_pair_id = prefill_instance.http_address + "_" + decode_instance.http_address
    
    if pd_pair_id in pd_pair:
        logger.info(f"""[Router] pd pair {pd_pair_id} already exist""")
        return

    logger.info(f"""[Router] initing pd pair {pd_pair_id}""")

    unique_id = router_nccl.ncclGetUniqueId()
    unique_id = bytes(unique_id.internal)
    rank = 0
    p_rank_num = prefill_instance.dp_size * prefill_instance.pp_size * prefill_instance.tp_size
    d_rank_num = decode_instance.dp_size * decode_instance.pp_size * decode_instance.tp_size
    world_size = p_rank_num + d_rank_num

    for dp_rank in range(prefill_instance.dp_size):
        for pp_rank in range(prefill_instance.pp_size):
            for tp_rank in range(prefill_instance.tp_size): 
                if prefill_instance.rank_table[dp_rank][pp_rank][tp_rank] not in sock_cache:
                    sock = zmq_context.socket(zmq.DEALER)
                    sock.setsockopt_string(zmq.IDENTITY, "router")
                    sock.connect(f"tcp://{prefill_instance.rank_table[dp_rank][pp_rank][tp_rank]}")
                    sock_cache[prefill_instance.rank_table[dp_rank][pp_rank][tp_rank]] = sock
                data = {
                    "cmd": "comm_init",
                    "pd_pair_id": pd_pair_id,
                    "unique_id" : unique_id,
                    "world_size": world_size,
                    "rank": rank
                }
                sock_cache[prefill_instance.rank_table[dp_rank][pp_rank][tp_rank]].send(msgpack.dumps(data))
                prefill_instance.comm_rank_table[dp_rank][pp_rank][tp_rank] = rank
                rank += 1
                logger.info(f"""[Router] dispatch unique_id of pd pair {pd_pair_id} to [P] [{dp_rank}, {pp_rank}, {tp_rank}]""")
    
    for dp_rank in range(decode_instance.dp_size):
        for pp_rank in range(decode_instance.pp_size):
            for tp_rank in range(decode_instance.tp_size):
                if decode_instance.rank_table[dp_rank][pp_rank][tp_rank] not in sock_cache:
                    sock = zmq_context.socket(zmq.DEALER)
                    sock.setsockopt_string(zmq.IDENTITY, "router")
                    sock.connect(f"tcp://{decode_instance.rank_table[dp_rank][pp_rank][tp_rank]}")
                    sock_cache[decode_instance.rank_table[dp_rank][pp_rank][tp_rank]] = sock
                data = {
                    "cmd": "comm_init",
                    "pd_pair_id": pd_pair_id,
                    "unique_id" : unique_id,
                    "world_size": world_size,
                    "rank": rank
                }
                sock_cache[decode_instance.rank_table[dp_rank][pp_rank][tp_rank]].send(msgpack.dumps(data))
                decode_instance.comm_rank_table[dp_rank][pp_rank][tp_rank] = rank
                rank += 1
                logger.info(f"""[Router] dispatch unique_id of pd pair {pd_pair_id} to [D] [{dp_rank}, {pp_rank}, {tp_rank}]""")
    
    pd_pair[pd_pair_id] = unique_id


def pd_pair_init():
    global prefill_instances
    global decode_instances
    global pending_prefill_ins
    global pending_decode_ins
    global ready_prefill_ins
    global ready_decode_ins
    global instance_cv

    while True:
        with instance_cv:
            while len(pending_prefill_ins) == 0 and len(pending_decode_ins) == 0:
                logger.info(f"""[Router] pd_pair_init: waiting for instance_cv""")
                instance_cv.wait()
            logger.info(f"""[Router] pd_pair_init: instance_cv finished waiting""")
            while pending_prefill_ins:
                p_ins = pending_prefill_ins[0]
                logger.info(f"""[Router] pd_pair_init: processing {p_ins} from pending_prefill_ins""")
                for d_ins in ready_decode_ins:
                    unique_id_dispatch(prefill_instances[p_ins], decode_instances[d_ins])
                ready_prefill_ins.append(p_ins)
                pending_prefill_ins.remove(p_ins)
            while pending_decode_ins:
                d_ins = pending_decode_ins[0]
                logger.info(f"""[Router] pd_pair_init: processing {d_ins} from pending_decode_ins""")
                for p_ins in ready_prefill_ins:
                    unique_id_dispatch(prefill_instances[p_ins], decode_instances[d_ins])
                ready_decode_ins.append(d_ins)
                pending_decode_ins.remove(d_ins)

def start_pd_pair_init():
    _thread = threading.Thread(
        target=pd_pair_init, daemon=True
    )
    _thread.start()
    return _thread


@app.route("/v1/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1

        global count
        global prefill_instances
        global instance_cv
        with instance_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr, prefill_instance = prefill_list[count % len(prefill_list)]

        global decode_instances
        with instance_cv:
            decode_list = list(decode_instances.items())
            decode_addr, decode_instance = decode_list[count % len(decode_list)]

        global pd_pair
        if prefill_instance.http_address + "_" + decode_instance.http_address not in pd_pair:
            raise RuntimeError("Selected PD pair was not inited")
        logger.info(
            f"handle_request count: {count}, [HTTP:{prefill_addr}, 👉 HTTP:{decode_addr}]"
        )
        count += 1

        request_id = f"{random_uuid()}"


        async def run_prefill():
            async for _ in forward_request(
                f"http://{prefill_addr}/v1/completions", prefill_request, request_id
            ):
                pass 
        
        prefill_task = asyncio.create_task(run_prefill())

        # return decode
        generator = forward_request(
            f"http://{decode_addr}/v1/completions", original_request_data, request_id
        )
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == "__main__":
    t = start_service_discovery("0.0.0.0", 30001)
    t_1 = start_pd_pair_init()
    t_2 = start_dp_dispatch()
    app.run(host="0.0.0.0", port=10001)
    t.join()
    t_1.join()
    t_2.join()
