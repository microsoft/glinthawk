#!/usr/bin/env python3
import sys
from typing import List, Dict

import settings

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required to run this program.")

import enum
import click
import socket
import asyncio
import logging
import datetime
import itertools

from enum import Enum
from dataclasses import dataclass, field

from common.message import Message
from protobuf import glinthawk_pb2 as glinthawk_pb

from rich.logging import RichHandler

logging.basicConfig(level=logging.NOTSET, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])


class Stage(Enum):
    PREATT = 0
    ATT = 1
    POSTATT = 2
    CLS = 3


@dataclass
class ModelInfo:
    name: str
    n_layers: int
    layers_per_worker: int


@dataclass
class WorkerStats:
    # no stats for now

    def __add__(self, other):
        pass

    def combine_max(self, other):
        pass


@dataclass
class Worker:
    class State(enum.Enum):
        Connected = enum.auto()
        Disconnected = enum.auto()

    id: int = field(default_factory=itertools.count().__next__)
    state: State = State.Connected
    ip: bytes = None
    port: int = None
    reader: asyncio.StreamReader = None
    writer: asyncio.StreamWriter = None
    model_name: str = ""
    start_layer: int = 0
    start_stage: Stage = Stage.PREATT
    end_layer: int = 0
    end_stage: Stage = Stage.CLS

    max_concurrency_size_pre: int = 16
    max_concurrency_size_att: int = 16
    max_concurrency_size_post: int = 16
    max_concurrency_size_cls: int = 16

    current_stats: WorkerStats = field(default_factory=WorkerStats)
    current_stats_time: datetime.datetime = None
    last_stats_time: datetime.datetime = None

    def work_rate(self):
        pass


class Coordinator:
    def __init__(self, **kwargs):
        self.workers: List[Worker] = []
        self.cls_gpu_worker: Worker = None
        self.ip_port_to_index: Dict[bytes, int] = {}
        self.layer_workers: Dict[int, List[Worker, Worker]] = {}
        self.incoming_messages = asyncio.Queue()
        self.outgoing_messages = asyncio.Queue()
        self.model = ModelInfo(
            name="llama2-70b-chat",
            n_layers=kwargs.get("n_layers", 80),
            layers_per_worker=kwargs.get("layers_per_worker", 1),
        )
        # self.model = ModelInfo(
        #     name="llama2-7b-chat",
        #     n_layers=kwargs.get("n_layers", 12),
        #     layers_per_worker=kwargs.get("layers_per_worker", 6),
        # )
        # self.model = ModelInfo(
        #     name="stories-110m",
        #     n_layers=kwargs.get("n_layers", 12),
        #     layers_per_worker=kwargs.get("layers_per_worker", 6),
        # )

        # Dummy prompt generation
        self.initial_dummy_count = kwargs.get("dummy_count", 0)
        self.generated_dummies = 0
        self.completed_dummies = 0

        # Concurrency sizes
        self.concurrency_size_pre = kwargs.get("concurrency_size_pre", 16)
        self.concurrency_size_att = kwargs.get("concurrency_size_att", 16)
        self.concurrency_size_post = kwargs.get("concurrency_size_post", 16)
        self.concurrency_size_cls = kwargs.get("concurrency_size_cls", 16)

        self.cpu_context_count = kwargs.get("cpu_context_count", 36*81*2)
        self.gpu_context_count = kwargs.get("gpu_context_count", 18*81)

        self.logger = logging.getLogger("coordinator")
        self.logger.setLevel(logging.INFO)

    def create_routing_message(self):
        message = glinthawk_pb.SetRoute()

        for newest_layer in range(self.model.n_layers):
            layer = max([i for i in self.layer_workers.keys() if i <= newest_layer])
            # Pre Attention
            sub_msg = glinthawk_pb.SetRoute.LayerToAddress()
            sub_msg.layer_num = newest_layer
            sub_msg.stage = glinthawk_pb.SetRoute.LayerToAddress.ProtoStage.PreAttention
            sub_msg.ip = socket.inet_ntoa(self.layer_workers[layer][0].ip)
            sub_msg.port = self.layer_workers[layer][0].port
            message.layer_to_address.append(sub_msg)
            # Attention
            sub_msg = glinthawk_pb.SetRoute.LayerToAddress()
            sub_msg.layer_num = newest_layer
            sub_msg.stage = glinthawk_pb.SetRoute.LayerToAddress.ProtoStage.Attention
            sub_msg.ip = socket.inet_ntoa(self.layer_workers[layer][1].ip)
            sub_msg.port = self.layer_workers[layer][1].port
            message.layer_to_address.append(sub_msg)
            # Post Attention
            sub_msg = glinthawk_pb.SetRoute.LayerToAddress()
            sub_msg.layer_num = newest_layer
            sub_msg.stage = glinthawk_pb.SetRoute.LayerToAddress.ProtoStage.PostAttention
            sub_msg.ip = socket.inet_ntoa(self.layer_workers[layer][0].ip)
            sub_msg.port = self.layer_workers[layer][0].port
            message.layer_to_address.append(sub_msg)
        # Classification
        sub_msg = glinthawk_pb.SetRoute.LayerToAddress()
        sub_msg.layer_num = self.model.n_layers - 1
        sub_msg.stage = glinthawk_pb.SetRoute.LayerToAddress.ProtoStage.Classification
        sub_msg.ip = socket.inet_ntoa(self.cls_gpu_worker.ip)
        sub_msg.port = self.cls_gpu_worker.port
        message.layer_to_address.append(sub_msg)

        return message

    async def handle_worker(self, reader, writer):
        addr = writer.get_extra_info("peername")
        self.logger.info(f"New connection from {addr!r}.")

        worker = Worker(reader=reader, writer=writer)
        self.workers += [worker]

        while True:
            try:
                message_header = await reader.readexactly(5)
                payload_length, opcode = Message.parse_header(message_header)
                message_payload = await reader.readexactly(payload_length)
                message = Message(opcode=opcode, payload=message_payload)
                await self.incoming_messages.put([worker, message])
            except:
                worker.state = Worker.State.Disconnected
                return

    @staticmethod
    async def send_message(worker, message):
        worker.writer.write(message.serialize())
        await worker.writer.drain()

    async def handle_outgoing_messages(self):
        while True:
            worker, message = await self.outgoing_messages.get()
            self.logger.info(f'Sending "{message!r}" to {worker.id}.')
            await self.send_message(worker, message)

    async def message_processor(self):
        while True:
            worker, message = await self.incoming_messages.get()
            # self.logger.info(f'Received "{message!r}" from {worker.id}.')

            if message.opcode in [Message.OpCode.HeyCPU, Message.OpCode.HeyGPU]:
                address = message.payload.decode()
                ip, port = address.split(":")
                worker.ip = socket.inet_aton(ip)
                worker.port = int(port)
                if worker.ip not in self.ip_port_to_index:
                    self.ip_port_to_index[worker.ip] = len(self.ip_port_to_index)
                self.logger.info(f"Worker {worker.id} is at {ip}:{port}, and sent {message.opcode}.")

                # assigning layers to this worker
                context_count = 0
                if message.opcode == Message.OpCode.HeyCPU:
                    if self.ip_port_to_index[worker.ip] < self.model.n_layers / self.model.layers_per_worker:
                        worker.start_layer = self.ip_port_to_index[worker.ip] * self.model.layers_per_worker
                        worker.end_layer = (self.ip_port_to_index[worker.ip] + 1) * self.model.layers_per_worker - 1
                        worker.max_concurrency_size_pre = 0
                        worker.max_concurrency_size_att = self.concurrency_size_att
                        worker.max_concurrency_size_post = 0
                        context_count = self.cpu_context_count
                    else:
                        worker.start_layer = self.model.n_layers - 1
                        worker.end_layer = self.model.n_layers - 1
                        worker.max_concurrency_size_cls = self.concurrency_size_cls
                        context_count = 0
                else:
                    if self.ip_port_to_index[worker.ip] < self.model.n_layers / self.model.layers_per_worker:
                        worker.start_layer = self.ip_port_to_index[worker.ip] * self.model.layers_per_worker
                        worker.end_layer = (self.ip_port_to_index[worker.ip] + 1) * self.model.layers_per_worker - 1
                        worker.max_concurrency_size_cls = 0
                        context_count = self.gpu_context_count
                    else:
                        worker.start_layer = self.model.n_layers - 1
                        worker.end_layer = self.model.n_layers - 1
                        worker.max_concurrency_size_cls = self.concurrency_size_cls
                        context_count = 0
                    worker.max_concurrency_size_pre = self.concurrency_size_pre
                    worker.max_concurrency_size_att = self.concurrency_size_att
                    worker.max_concurrency_size_post = self.concurrency_size_post

                initialization_message = glinthawk_pb.InitializeWorker(
                    model_name=self.model.name,
                    start_layer=worker.start_layer,
                    end_layer=worker.end_layer,
                    concurrency_pre_att_size=worker.max_concurrency_size_pre,
                    concurrency_att_size=worker.max_concurrency_size_att,
                    concurrency_post_att_size=worker.max_concurrency_size_post,
                    concurrency_cls_size=worker.max_concurrency_size_cls,
                    max_context_count=context_count,
                    randomize=message.opcode == Message.OpCode.HeyCPU,
                    blobstore_uri=settings.GLINTHAWK_PROMPT_BLOBSTORE,
                )

                self.outgoing_messages.put_nowait(
                    [
                        worker,
                        Message(
                            Message.OpCode.InitializeWorker,
                            initialization_message.SerializeToString(),
                        ),
                    ]
                )

                ignore = False
                if message.opcode == Message.OpCode.HeyCPU:
                    if self.ip_port_to_index[worker.ip] < self.model.n_layers / self.model.layers_per_worker:
                        if worker.start_layer not in self.layer_workers:
                            self.layer_workers[worker.start_layer] = [None, None]
                        self.layer_workers[worker.start_layer][1] = worker
                    else:
                        ignore = True
                else:
                    if self.ip_port_to_index[worker.ip] < self.model.n_layers / self.model.layers_per_worker:
                        if worker.start_layer not in self.layer_workers:
                            self.layer_workers[worker.start_layer] = [None, None]
                        self.layer_workers[worker.start_layer][0] = worker
                    else:
                        self.cls_gpu_worker = worker

                if len(self.layer_workers) == self.model.n_layers / self.model.layers_per_worker and \
                        all(self.layer_workers[key][0] is not None and self.layer_workers[key][1] is not None for key in self.layer_workers) and \
                        self.cls_gpu_worker is not None and not ignore:
                    # all layers have been assigned
                    # setting the route for the first worker
                    self.outgoing_messages.put_nowait(
                        [
                            self.layer_workers[0][0],
                            Message(
                                Message.OpCode.SetRoute,
                                self.create_routing_message().SerializeToString(),
                            ),
                        ]
                    )

                    self.logger.info(
                        f"Layer 0 is at {socket.inet_ntoa(self.layer_workers[0][0].ip)}:{self.layer_workers[0][0].port};"
                        "completions can be found there."
                    )

                    # telling the first worker to generate dummy prompts
                    if self.initial_dummy_count:
                        self.outgoing_messages.put_nowait(
                            [
                                self.layer_workers[0][0],
                                Message(
                                    Message.OpCode.PushDummyPrompts,
                                    str(self.initial_dummy_count).encode(),
                                ),
                            ]
                        )

                        self.generated_dummies += self.initial_dummy_count
                        asyncio.create_task(self.maybe_generate_dummies())

            elif message.opcode == Message.OpCode.InferenceState:
                self.logger.error("Received InferenceState message from a worker.")

            elif message.opcode == Message.OpCode.PromptCompleted:
                count = int(message.payload.decode())
                self.logger.info(f"Worker {worker.id} completed {count} prompts.")
                self.completed_dummies += count

            elif message.opcode == Message.OpCode.WorkerStats:
                pass

    async def maybe_generate_dummies(self):
        while True:
            if self.generated_dummies - self.completed_dummies < self.initial_dummy_count // 2:
                gen_count = (self.initial_dummy_count // 2) - (self.generated_dummies - self.completed_dummies)
                if gen_count <= 0:
                    continue

                self.logger.warning(f"Generating {gen_count} dummy prompts.")
                self.outgoing_messages.put_nowait(
                    [
                        self.layer_workers[0][0],
                        Message(
                            Message.OpCode.PushDummyPrompts,
                            str(gen_count).encode(),
                        ),
                    ]
                )

                self.generated_dummies += gen_count

            await asyncio.sleep(30)

    async def main(self, listen_address, listen_port):
        server = await asyncio.start_server(self.handle_worker, listen_address, listen_port)

        async with server:
            asyncio.create_task(self.message_processor())
            asyncio.create_task(self.handle_outgoing_messages())
            await server.serve_forever()


@click.command()
@click.option("--n-layers", "-N", help="Total layers of the model", required=True, type=click.INT)
@click.option("--layers-per-worker", "-L", required=True, type=click.INT)
@click.option("--listen-address", required=True)
@click.option("--listen-port", required=True)
@click.option(
    "--dummy-count",
    help="Number of dummy prompts to generate",
    type=click.INT,
    default=0,
)
@click.option("--cpu_context_count", "-NCPU", type=click.INT, default=0)
@click.option("--gpu_context_count", "-NGPU", type=click.INT, default=0)
@click.option("--concurrency-size-pre", "-C1", type=click.INT, default=1)
@click.option("--concurrency-size-att", "-C2", type=click.INT, default=1)
@click.option("--concurrency-size-post", "-C3", type=click.INT, default=1)
@click.option("--concurrency-size-cls", "-C4", type=click.INT, default=1)
def main(listen_address, listen_port, **kwargs):
    coordinator = Coordinator(**kwargs)
    asyncio.run(coordinator.main(listen_address, listen_port))


if __name__ == "__main__":
    main()
