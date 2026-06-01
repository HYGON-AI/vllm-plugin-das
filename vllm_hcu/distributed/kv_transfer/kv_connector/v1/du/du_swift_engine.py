# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
import threading
import time
import typing
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING, Any, Optional

import msgpack
import torch
import zmq
import regex

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
)
from vllm_hcu.distributed.kv_transfer.kv_connector.v1.du.tensor_memory_pool import (  # noqa: E501
    TensorMemoryPool,
)
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import current_stream

import vllm_hcu.platforms.envs as henvs
from vllm.distributed.parallel_state import get_pp_group, get_tp_group

from dataclasses import dataclass
from vllm.model_executor.models.utils import extract_layer_index
from vllm.distributed.utils import get_pp_indices
from vllm.config import ModelConfig
import vllm.compilation.monitor as monitor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32


@contextmanager
def set_du_swift_context(num_channels: str):
    original_values: dict[str, Any] = {}
    env_vars = [
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CUMEM_ENABLE",
        "NCCL_BUFFSIZE",
        "NCCL_PROTO",  # LL,LL128,SIMPLE
        "NCCL_ALGO",  # RING,TREE
    ]

    for var in env_vars:
        original_values[var] = os.environ.get(var)

    logger.info("set_du_swift_context, original_values: %s", original_values)

    try:
        os.environ["NCCL_MAX_NCHANNELS"] = num_channels
        os.environ["NCCL_MIN_NCHANNELS"] = num_channels
        os.environ["NCCL_CUMEM_ENABLE"] = "1"
        yield
    finally:
        for var in env_vars:
            if original_values[var] is not None:
                os.environ[var] = original_values[var]
            else:
                os.environ.pop(var, None)


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor


@dataclass
class RemoteAddr:
    pd_pair_id: str = ""
    zmq_address: str = ""
    comm_rank: int = 0

class DuSwiftEngine:
    def __init__(self,
                local_rank: int,
                port_offset: int,
                config: KVTransferConfig,
                model_config: ModelConfig,
                dp_rank: int = 0,
                pp_rank: int = 0,
                tp_rank: int = 0,
                dp_size: int = 0,
                pp_size: int = 0,
                tp_size: int = 0,
                library_path: Optional[str] = None) -> None:
        self.config = config
        self.model_config = model_config
        self.rank = port_offset
        self.local_rank = local_rank
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.dp_size = dp_size
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        self.total_num_hidden_layers = getattr(self.model_config.hf_text_config,
                                              "num_hidden_layers", 0)
        self.pp_rank = get_pp_group().rank_in_group
        self.tp_rank = get_tp_group().rank_in_group
        self.pp_size = get_pp_group().world_size 
        self.tp_size = get_tp_group().world_size
        if config.is_kv_producer:
            self.remote_tp_size = self.config.get_from_extra_config(
                "remote_tp_size", self.tp_size)
            self.remote_pp_size = self.config.get_from_extra_config(
                "remote_pp_size", self.pp_size)
            self.enable_asymmetric_p2p = self.config.get_from_extra_config(
                "enable_asymmetric_p2p", False)

            if self.remote_tp_size % self.tp_size != 0:
                logger.warning(" the Prefill TP size must be less than or equal to the Decode TP size!!!!")
            self.multp = int(self.remote_tp_size / self.tp_size)
        self.multiple_machines = self.config.get_from_extra_config(
                "enable_multiple_machines", False)
        
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self.hostname = get_ip()
        self.port = port

        # Each card corresponds to a ZMQ address.
        self.zmq_address = f"{self.hostname}:{self.port}"

        # If `proxy_ip` or `proxy_port` is `""`,
        # then the ping thread will not be enabled.
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
            self.http_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port
            # the `http_port` must be consistent with the port of OpenAI.
            http_port = self.config.get_from_extra_config("http_port", None)
            if http_port is None:
                example_cfg = {
                    "kv_connector": "P2pNcclConnector",
                    "kv_connector_extra_config": {"http_port": 8000},
                }
                example = (
                    f"--port=8000 --kv-transfer-config='{json.dumps(example_cfg)}'"
                )
                raise ValueError(
                    "kv_connector_extra_config.http_port is required. "
                    f"Example: {example}"
                )
            self.http_address = f"{self.hostname}:{http_port}"

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        mem_pool_size_gb = float(
            self.config.get_from_extra_config(
                "mem_pool_size_gb", DEFAULT_MEM_POOL_SIZE_GB
            )
        )
        self.pool = TensorMemoryPool(
            max_block_size=int(mem_pool_size_gb * 1024**3)
        )  # GB

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC.
        self.send_type = self.config.get_from_extra_config("send_type", "PUT_ASYNC")
        if self.send_type == "GET":
            # tensor_id: torch.Tensor
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            # PUT or PUT_ASYNC
            # tensor_id: torch.Tensor
            self.send_queue: deque[SendQueueItem] = deque()
            if self.send_type == "PUT_ASYNC":
                self.send_thread = threading.Thread(
                    target=self.send_async, daemon=True
                )
                self.send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.comms: dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)

        self.nccl_num_channels = self.config.get_from_extra_config(
            "nccl_num_channels", "8"
        )

        self.listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True
        )
        self.listener_thread.start()

        self.ping_thread = None
        if self.multiple_machines:
            if port_offset == 0 and self.proxy_address != "":
                self.ping_thread = threading.Thread(target=self.ping,
                                                    daemon=True)
                self.ping_thread.start()
        else:
            if self.proxy_address != "":
                self.ping_thread = threading.Thread(target=self.ping_new,
                                                    daemon=True)
                self.ping_thread.start()
        logger.info(
            "💯P2pNcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, nccl_num_channels:%s",
            self.rank,
            self.local_rank,
            self.http_address,
            self.zmq_address,
            self.proxy_address,
            self.send_type,
            self.buffer_size_threshold,
            self.nccl_num_channels,
        )

    def create_connect_new(self, remote_address: typing.Optional[str] = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt(zmq.SNDHWM, 10000)
            sock.setsockopt(zmq.RCVHWM, 5000) 
            sock.setsockopt(zmq.LINGER, 0)  
            sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
            sock.setsockopt_string(zmq.IDENTITY, f"P-{self.zmq_address}")
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock

        return self.socks[remote_address]
    
    def create_connect(self, remote_address: str | None = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                logger.info(
                    "👋comm exists, remote_address:%s, comms:%s",
                    remote_address,
                    self.comms,
                )
                return sock, self.comms[remote_address]

            unique_id = self.nccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            with torch.accelerator.device_index(self.device.index):
                rank = 0
                with set_du_swift_context(self.nccl_num_channels):
                    comm: ncclComm_t = self.nccl.ncclCommInitRank(2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info(
                    "🤝ncclCommInitRank Success, %sğ%s, MyRank:%s",
                    self.zmq_address,
                    remote_address,
                    rank,
                )

        return self.socks[remote_address], self.comms[remote_address]

    def get_send_queue_items(self, request_id: str, layer_name: str,
                             tensor: torch.Tensor,
                             is_mla: bool) -> list[any]:
        tensor_id = self.get_tensor_id(request_id, layer_name)
        remote_ip, remote_port = self.parse_request_id(request_id, True)
        
        p_ip, p_port = self.parse_request_id(request_id, False)
        pd_pair_id = p_ip + ":" + str(p_port) + "_" + remote_ip + ":" + str(remote_port)

        items: SendQueueItem = []

        if not self.enable_asymmetric_p2p:
                remote_address = remote_ip + ":" + str(remote_port + self.rank)
                remote_addr = RemoteAddr(pd_pair_id, remote_address, self.rank + self.pp_size * self.tp_size)
                # logger.info(f"""+++++xiabo tensor_id:{tensor_id} request_id:{request_id} remote_address:{remote_address}""")
                item_tmp = SendQueueItem(
                    tensor_id=tensor_id, remote_address=remote_addr, tensor=tensor
                )
                items.append(item_tmp)
                return items
        
        if not is_mla:
            logger.error(" DuSwift only support mla model symmetric PP/TP!!!!")
        
        remote_pp_rank = self.compute_remote_pp_rank(layer_name)
        

        for d_tp_rank in range(self.remote_tp_size):
            for mul_tp in range(self.multp):
                if self.tp_rank + mul_tp * self.tp_size == d_tp_rank:
                    remote_port_offset = remote_pp_rank * self.remote_tp_size + d_tp_rank
                    remote_address = remote_ip + ":" + str(remote_port + remote_port_offset)
                    remote_addr = RemoteAddr(pd_pair_id, remote_address, remote_port_offset + self.pp_size * self.tp_size)
                    logger.debug(
                        "Wait to send::%s, tensor_shape:%s, "
                        "(pp=%d, tp=%d) -> remote_address=%s(pp=%d, tp=%d) comm_rank (%d -> %d)", tensor_id,
                        tensor.shape, self.pp_rank, self.tp_rank, remote_address,
                        remote_pp_rank, self.rank * mul_tp + self.rank, self.rank, remote_port_offset + self.pp_size * self.tp_size)
                    item_tmp = SendQueueItem(
                        tensor_id=tensor_id, remote_address=remote_addr, tensor=tensor
                    )
                    items.append(item_tmp)
        return items

    def send_tensor_new(
        self,
        request_id: str,
        layer_name: str,
        tensor: torch.Tensor,
        is_mla: bool = False,
    ) -> bool:
        tensor_id = self.get_tensor_id(request_id, layer_name)

        if self.send_type == "PUT":
            return all(
                self.send_sync_new(item) for item in self.get_send_queue_items(
                    request_id, layer_name, tensor, is_mla))

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                for item in self.get_send_queue_items(request_id, layer_name,
                                                      tensor, is_mla):
                    self.send_queue.append(item)
                self.send_queue_cv.notify()
            return True
        if self.send_type == "GET":
            logger.error(" DuSwift new not support GET model, please set VLLM_P2PNCCL_NEW=0 use defalut model!!!!")

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str | None = None,
    ) -> bool:
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True

        item = SendQueueItem(
            tensor_id=tensor_id, remote_address=remote_address, tensor=tensor
        )

        if self.send_type == "PUT":
            return self.send_sync(item)

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                self.send_queue_cv.notify()
            return True

        # GET
        with self.send_store_cv:
            tensor_size = tensor.element_size() * tensor.numel()
            if tensor_size > self.buffer_size_threshold:
                logger.warning(
                    "⛔[GET]tensor_id:%s, tensor_size:%d, is greater than"
                    "buffer size threshold :%d, skip send to %s, rank:%d",
                    tensor_id,
                    tensor_size,
                    self.buffer_size_threshold,
                    remote_address,
                    self.rank,
                )
                return False
            while self.buffer_size + tensor_size > self.buffer_size_threshold:
                assert len(self.send_store) > 0
                oldest_tensor_id = next(iter(self.send_store))
                oldest_tensor = self.send_store.pop(oldest_tensor_id)
                oldest_tensor_size = (
                    oldest_tensor.element_size() * oldest_tensor.numel()
                )
                self.buffer_size -= oldest_tensor_size
                logger.debug(
                    "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d,"
                    " buffer_size:%d, oldest_tensor_size:%d, rank:%d",
                    remote_address,
                    tensor_id,
                    tensor_size,
                    self.buffer_size,
                    oldest_tensor_size,
                    self.rank,
                )

            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
            logger.debug(
                "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                "shape:%s, rank:%d, buffer_size:%d(%.2f%%)",
                remote_address,
                tensor_id,
                tensor_size,
                tensor.shape,
                self.rank,
                self.buffer_size,
                self.buffer_size / self.buffer_size_threshold * 100,
            )
        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: str | None = None,
    ) -> torch.Tensor:
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    self.recv_store_cv.wait()
                tensor = self.recv_store[tensor_id]

            if tensor is not None:
                if isinstance(tensor, tuple):
                    addr, dtype, shape = tensor
                    tensor = self.pool.load_tensor(addr, dtype, shape, self.device)
                else:
                    self.buffer_size -= tensor.element_size() * tensor.numel()
            else:
                duration = time.time() - start_time
                logger.warning(
                    "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, rank:%d",
                    remote_address,
                    tensor_id,
                    duration * 1000,
                    self.rank,
                )
            return tensor

        # GET
        if remote_address is None:
            return None

        if remote_address not in self.socks:
            self.create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]

        data = {"cmd": "GET", "tensor_id": tensor_id}
        sock.send(msgpack.dumps(data))

        message = sock.recv()
        data = msgpack.loads(message)
        if data["ret"] != 0:
            logger.warning(
                "🔴[GET]Recv From %s, tensor_id: %s, ret: %d",
                remote_address,
                tensor_id,
                data["ret"],
            )
            return None

        with torch.cuda.stream(self.recv_stream):
            tensor = torch.empty(
                data["shape"], dtype=getattr(torch, data["dtype"]), device=self.device
            )

        self.recv(comm, tensor, rank ^ 1, self.recv_stream)

        return tensor

    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue

            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["cmd"] == "NEW":
                unique_id = self.nccl.unique_id_from_bytes(bytes(data["unique_id"]))
                with torch.accelerator.device_index(self.device.index):
                    rank = 1
                    with set_du_swift_context(self.nccl_num_channels):
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank
                        )
                    self.comms[remote_address.decode()] = (comm, rank)
                    logger.info(
                        "🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        rank,
                    )
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                try:
                    with torch.cuda.stream(self.recv_stream):
                        tensor = torch.empty(
                            data["shape"],
                            dtype=getattr(torch, data["dtype"]),
                            device=self.device,
                        )
                    self.router_socket.send_multipart([remote_address, b"0"])
                    comm, rank = self.comms[remote_address.decode()]
                    self.recv(comm, tensor, rank ^ 1, self.recv_stream)
                    tensor_size = tensor.element_size() * tensor.numel()
                    if self.buffer_size + tensor_size > self.buffer_size_threshold:
                        # Store Tensor in memory pool
                        addr = self.pool.store_tensor(tensor)
                        tensor = (addr, tensor.dtype, tensor.shape)
                        logger.warning(
                            "🔴[PUT]Recv Tensor, Out Of Threshold, "
                            "%👈%s, data:%s, addr:%d",
                            self.zmq_address,
                            remote_address.decode(),
                            data,
                            addr,
                        )
                    else:
                        self.buffer_size += tensor_size

                except torch.cuda.OutOfMemoryError:
                    self.router_socket.send_multipart([remote_address, b"1"])
                    tensor = None
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Memory, %sğ%s, data:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                    )

                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()
            elif data["cmd"] == "PUT_NEW":
                    tensor_id = data["tensor_id"]
                    if "tensor_split_num" in data:
                        self.tensor_split_num = data["tensor_split_num"]
                    try:
                        with torch.cuda.stream(self.recv_stream):
                            tensor = torch.empty(data["shape"],
                                                 dtype=getattr(
                                                     torch, data["dtype"]),
                                                 device=self.device)
                        self.router_socket.send_multipart(
                            [remote_address, b"0"])
                        # comm, rank = self.comms[remote_address.decode()]
                        # self.recv(comm, tensor, rank ^ 1, self.recv_stream)
                        comm, rank = self.comms[data["pd_pair_id"]]
                        self.recv(comm, tensor, int(data["comm_rank"]), self.recv_stream)
                        tensor_size = tensor.element_size() * tensor.numel()
                        if (self.buffer_size + tensor_size
                                > self.buffer_size_threshold):
                            # Store Tensor in memory pool
                            addr = self.pool.store_tensor(tensor)
                            tensor = (addr, tensor.dtype, tensor.shape)
                        else:
                            self.buffer_size += tensor_size

                    except torch.cuda.OutOfMemoryError:
                        self.router_socket.send_multipart(
                            [remote_address, b"1"])
                        tensor = None
                        logger.warning(
                            "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, "
                            "data:%s", self.zmq_address,
                            remote_address.decode(), data)
                    with self.recv_store_cv:
                        self.recv_store[tensor_id] = tensor
                        self.have_received_tensor_id(tensor_id)
                        self.recv_store_cv.notify()
            elif data["cmd"] == "comm_init":
                    unique_id = self.nccl.unique_id_from_bytes(
                        bytes(data["unique_id"]))
                    with torch.cuda.device(self.device):
                        rank = int(data["rank"])
                        world_size = int(data["world_size"])
                        with set_du_swift_context(self.nccl_num_channels):
                            while monitor.cudagraph_capturing_enabled:
                                time.sleep(1)
                            comm: ncclComm_t = self.nccl.ncclCommInitRank(
                                    world_size, unique_id, rank)
                        self.comms[data["pd_pair_id"]] = (comm, rank)
                        logger.info(
                            "🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                            self.zmq_address, data["pd_pair_id"], rank)
            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                with self.send_store_cv:
                    tensor = self.send_store.pop(tensor_id, None)
                    if tensor is not None:
                        data = {
                            "ret": 0,
                            "shape": tensor.shape,
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                        # LRU
                        self.send_store[tensor_id] = tensor
                        self.have_sent_tensor_id(tensor_id)
                    else:
                        data = {"ret": 1}

                self.router_socket.send_multipart([remote_address, msgpack.dumps(data)])

                if data["ret"] == 0:
                    comm, rank = self.comms[remote_address.decode()]
                    self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)
            else:
                logger.warning(
                    "ğ§Unexpected, Received message from %s, data:%s",
                    remote_address,
                    data,
                )

    def have_sent_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        if request_id not in self.send_request_id_to_tensor_ids:
            self.send_request_id_to_tensor_ids[request_id] = set()
        self.send_request_id_to_tensor_ids[request_id].add(tensor_id)

    def have_received_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)

    def send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                if not self.send_queue:
                    self.send_queue_cv.notify()
            if self.multiple_machines:
                self.send_sync(item)
            else:
                self.send_sync_new(item)

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.send_queue_cv:
                while self.send_queue:
                    self.send_queue_cv.wait()
            duration = time.time() - start_time
            logger.debug(
                "ğ§[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d",
                duration * 1000,
                self.rank,
            )

    def send_sync_new(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        
        if item.remote_address.zmq_address not in self.socks:
            # logger.info(f"""=============xiabo remote_address.zmq_address:{remote_address.zmq_address}""")
            self.create_connect_new(item.remote_address.zmq_address)

        sock = self.socks[item.remote_address.zmq_address]
        comm, rank = self.comms[item.remote_address.pd_pair_id]
        data = {
            "cmd": "PUT_NEW",
            "tensor_id": item.tensor_id,
            "shape": item.tensor.shape,
            "dtype": str(item.tensor.dtype).replace("torch.", ""),
            "pd_pair_id": item.remote_address.pd_pair_id,
            "comm_rank": rank
        }
        #logger.info(f"""_send_sync_new:{data}""")
        sock.send(msgpack.dumps(data))

        response = sock.recv()
        if response != b"0":
            logger.error(
                "🔴Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "MyRank:%s, data:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address, item.remote_address.zmq_address, rank, data, item.tensor.shape,
                item.tensor.element_size() * item.tensor.numel() / 1024**3,
                response.decode())
            return False

        self.send(comm, item.tensor.to(self.device), item.remote_address.comm_rank, self.send_stream)
        if self.send_type == "PUT_ASYNC":
            self.have_sent_tensor_id(item.tensor_id)

        return True

    def send_sync(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        if item.remote_address not in self.socks:
            self.create_connect(item.remote_address)

        tensor = item.tensor

        sock = self.socks[item.remote_address]
        comm, rank = self.comms[item.remote_address]
        data = {
            "cmd": "PUT",
            "tensor_id": item.tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        sock.send(msgpack.dumps(data))

        response = sock.recv()
        if response != b"0":
            logger.error(
                "ğ´Send Tensor, Peer Out Of Memory/Threshold, %s ğ %s, "
                "MyRank:%s, data:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address,
                item.remote_address,
                rank,
                data,
                tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode(),
            )
            return False

        self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)

        if self.send_type == "PUT_ASYNC":
            self.have_sent_tensor_id(item.tensor_id)

        return True

    def get_finished(
        self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        # Clear the buffer upon request completion.
        for request_id in finished_req_ids:
            for layer_name in no_compile_layers:
                tensor_id = request_id + "#" + layer_name
                if tensor_id in self.recv_store:
                    with self.recv_store_cv:
                        tensor = self.recv_store.pop(tensor_id, None)
                        self.send_request_id_to_tensor_ids.pop(request_id, None)
                        self.recv_request_id_to_tensor_ids.pop(request_id, None)
                    if isinstance(tensor, tuple):
                        addr, _, _ = tensor
                        self.pool.free(addr)

        # TODO:Retrieve requests that have already sent the KV cache.
        finished_sending: set[str] = set()

        # TODO:Retrieve requests that have already received the KV cache.
        finished_recving: set[str] = set()

        return finished_sending or None, finished_recving or None

    def ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address,
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def ping_new(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")

        if self.rank == 0:
            data = {
                "type": "P_init" if self.config.is_kv_producer else "D_init",
                "http_address": self.http_address,
                "zmq_address": self.zmq_address,
                "dp_size" : self.dp_size,
                "pp_size" : self.pp_size,
                "tp_size" : self.tp_size
            }
            # logger.info(f"""_ping data:{data}""")
            sock.send(msgpack.dumps(data))
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "dp_rank" : self.dp_rank,
            "pp_rank" : self.pp_rank,
            "tp_rank" : self.tp_rank,
            "zmq_address": self.zmq_address
        }
        # while True:
        # logger.info(f"""_ping data:{data}""")
        sock.send(msgpack.dumps(data))
            # time.sleep(3)

    def send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                dst,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
        stream.synchronize()

    def recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclRecv(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                src,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
        stream.synchronize()

    def close(self) -> None:
        self.listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self.send_thread.join()
        if self.ping_thread is not None:
            self.ping_thread.join()
    
    def get_pp_indices_d(self, num_hidden_layers: int, pp_rank: int,
                   pp_size: int) -> tuple[int, int]:
        partition_list_str = henvs.VLLM_PP_LAYER_PARTITION_D
        if partition_list_str is not None:
            try:
                partitions = [
                    int(layer) for layer in partition_list_str.split(",")
                ]
            except ValueError as err:
                raise ValueError("Invalid partition string: {}".format(
                    partition_list_str)) from err
            if len(partitions) != pp_size:
                raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
            if sum(partitions) != num_hidden_layers:
                raise ValueError(
                    f"{sum(partitions)=} does not match {num_hidden_layers=}.")
        else:
            layers_per_partition = num_hidden_layers // pp_size
            partitions = [layers_per_partition for _ in range(pp_size)]

            if remaining_layers := num_hidden_layers % pp_size:
                for i in range(2, remaining_layers + 2):
                    partitions[-i] += 1
                logger.info(
                    "Hidden layers were unevenly partitioned: [%s]. "
                    "This can be manually overridden using the "
                    "VLLM_PP_LAYER_PARTITION_D environment variable",
                    ",".join(str(p) for p in partitions))

        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]

        return (start_layer, end_layer)

    def compute_remote_pp_rank(self, layer_name: str) -> int:
        current_layer_idx = extract_layer_index(layer_name)
        for d_pp_rank in range(self.remote_pp_size):
            start, end = self.get_pp_indices_d(self.total_num_hidden_layers, d_pp_rank, self.remote_pp_size)
            # logger.info(f"""compute_remote_pp_rank : current_layer_idx:{current_layer_idx} start:{start} end:{end}""")
            if (current_layer_idx == self.total_num_hidden_layers):
                return self.remote_pp_size - 1
            if start <= current_layer_idx < end:
                return d_pp_rank
        return -1


    @staticmethod
    def get_tensor_id(request_id: str, layer_name: str) -> str:
        return request_id + "#" + layer_name

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = regex.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            return ip, port
        raise ValueError(
            f"Request id {request_id} does not contain hostname and port")
