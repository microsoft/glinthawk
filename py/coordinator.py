#!/usr/bin/env python3

import sys

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

from coordinator.ui import CoordinatorUI


class Stage(Enum):
    PREATT = 0
    ATT = 1
    POSTATT = 2


@dataclass
class ModelInfo:
    name: str
    n_layers: int
    layers_per_worker: int


@dataclass
class WorkerStats:
    states_sent: int = 0
    states_received: int = 0
    states_processed: int = 0

    tokens_processed: int = 0
    tokens_generated: int = 0
    prompts_completed: int = 0

    bytes_sent: int = 0
    bytes_received: int = 0

    def __add__(self, other):
        return WorkerStats(
            states_sent=self.states_sent + other.states_sent,
            states_received=self.states_received + other.states_received,
            states_processed=self.states_processed + other.states_processed,
            tokens_processed=self.tokens_processed + other.tokens_processed,
            tokens_generated=self.tokens_generated + other.tokens_generated,
            prompts_completed=self.prompts_completed + other.prompts_completed,
            bytes_sent=self.bytes_sent + other.bytes_sent,
            bytes_received=self.bytes_received + other.bytes_received,
        )

    def combine_max(self, other):
        return WorkerStats(
            states_sent=max(self.states_sent, other.states_sent),
            states_received=max(self.states_received, other.states_received),
            states_processed=max(self.states_processed, other.states_processed),
            tokens_processed=max(self.tokens_processed, other.tokens_processed),
            tokens_generated=max(self.tokens_generated, other.tokens_generated),
            prompts_completed=max(self.prompts_completed, other.prompts_completed),
            bytes_sent=max(self.bytes_sent, other.bytes_sent),
            bytes_received=max(self.bytes_received, other.bytes_received),
        )


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
    end_stage: Stage = Stage.POSTATT
    max_concurrency_size_pre: int = 16
    max_concurrency_size_att: int = 16
    max_concurrency_size_posy: int = 16

    current_stats: WorkerStats = field(default_factory=WorkerStats)
    current_stats_time: datetime.datetime = None
    last_stats_time: datetime.datetime = None

    def work_rate(self):
        if self.current_stats_time is None or self.last_stats_time is None:
            return WorkerStats()

        elapsed_time = (self.current_stats_time - self.last_stats_time).total_seconds()

        if elapsed_time == 0:
            return WorkerStats()

        return WorkerStats(
            states_sent=(self.current_stats.states_sent) / elapsed_time,
            states_received=(self.current_stats.states_received) / elapsed_time,
            states_processed=(self.current_stats.states_processed) / elapsed_time,
            tokens_processed=(self.current_stats.tokens_processed) / elapsed_time,
            tokens_generated=(self.current_stats.tokens_generated) / elapsed_time,
            prompts_completed=(self.current_stats.prompts_completed) / elapsed_time,
            bytes_sent=(self.current_stats.bytes_sent) / elapsed_time,
            bytes_received=(self.current_stats.bytes_received) / elapsed_time,
        )


class Coordinator:
    def __init__(self, **kwargs):
        self.workers = []
        self.layer_gpu_workers = {}
        self.layer_cpu_workers = {}
        self.incoming_messages = asyncio.Queue()
        self.outgoing_messages = asyncio.Queue()
        # self.model = ModelInfo(
        #     name="llama2-7b-chat",
        #     n_layers=kwargs.get("n_layers", 12),
        #     layers_per_worker=kwargs.get("layers_per_worker", 6),
        # )
        self.model = ModelInfo(
            name="stories-110m",
            n_layers=kwargs.get("n_layers", 12),
            layers_per_worker=kwargs.get("layers_per_worker", 6),
        )

        self.aggregate_stats = WorkerStats()
        self.max_rates = WorkerStats()

        self.concurrency_size_pre = kwargs.get("concurrency_size_pre", 16)
        self.concurrency_size_att = kwargs.get("concurrency_size_att", 16)
        self.concurrency_size_post = kwargs.get("concurrency_size_post", 16)
        self.dummy_prompt_count = kwargs.get("dummy_count", 0)

        self.logger = logging.getLogger("coordinator")
        self.logger.setLevel(logging.INFO)

        if kwargs.get("ui", False):
            self.ui = CoordinatorUI(self, self.logger)
        else:
            self.ui = None
            self.logger.addHandler(logging.StreamHandler(sys.stderr))

    def create_routing_message(self):
        message = glinthawk_pb.SetRoute()
        for newest_layer in range(self.model.n_layers):
            layer = max([i for i in self.layer_cpu_workers.keys() if i <= newest_layer])
            # Pre Attention
            sub_msg = glinthawk_pb.SetRoute.LayerToAddress()
            sub_msg.layer_num = newest_layer
            sub_msg.stage = glinthawk_pb.SetRoute.LayerToAddress.ProtoStage.PreAttention
            sub_msg.ip = socket.inet_ntoa(self.layer_gpu_workers[layer].ip)
            sub_msg.port = self.layer_gpu_workers[layer].port
            message.layer_to_address.append(sub_msg)
            # Attention
            sub_msg = glinthawk_pb.SetRoute.LayerToAddress()
            sub_msg.layer_num = newest_layer
            sub_msg.stage = glinthawk_pb.SetRoute.LayerToAddress.ProtoStage.Attention
            sub_msg.ip = socket.inet_ntoa(self.layer_cpu_workers[layer].ip)
            sub_msg.port = self.layer_cpu_workers[layer].port
            message.layer_to_address.append(sub_msg)
            # Post Attention
            sub_msg = glinthawk_pb.SetRoute.LayerToAddress()
            sub_msg.layer_num = newest_layer
            sub_msg.stage = glinthawk_pb.SetRoute.LayerToAddress.ProtoStage.PostAttention
            sub_msg.ip = socket.inet_ntoa(self.layer_gpu_workers[layer].ip)
            sub_msg.port = self.layer_gpu_workers[layer].port
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
                self.logger.info(f"Worker {worker.id} is at {ip}:{port}, and sent {message.opcode}.")

                # assigning layers to this worker
                if message.opcode == Message.OpCode.HeyCPU:
                    worker.start_layer = len(self.layer_cpu_workers) * self.model.layers_per_worker
                    worker.end_layer = (len(self.layer_cpu_workers) + 1) * self.model.layers_per_worker - 1
                    worker.max_concurrency_size_pre = 0
                    worker.max_concurrency_size_att = self.concurrency_size_att
                    worker.max_concurrency_size_post = 0
                else:
                    worker.start_layer = len(self.layer_gpu_workers) * self.model.layers_per_worker
                    worker.end_layer = (len(self.layer_gpu_workers) + 1) * self.model.layers_per_worker - 1
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
                    # randomize=message.opcode == Message.OpCode.HeyCPU,
                    max_context_count=1024 if message.opcode == Message.OpCode.HeyCPU else 384,
                    randomize=False,
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

                if message.opcode == Message.OpCode.HeyCPU:
                    self.layer_cpu_workers[worker.start_layer] = worker
                else:
                    self.layer_gpu_workers[worker.start_layer] = worker

                if (len(self.layer_cpu_workers) == self.model.n_layers / self.model.layers_per_worker and len(
                        self.layer_gpu_workers) == len(self.layer_cpu_workers)):
                    # all layers have been assigned
                    # setting the route for the first worker
                    self.outgoing_messages.put_nowait(
                        [
                            self.layer_gpu_workers[0],
                            Message(
                                Message.OpCode.SetRoute,
                                self.create_routing_message().SerializeToString(),
                            ),
                        ]
                    )

                    # telling the first worker to generate dummy prompts
                    if self.dummy_prompt_count:
                        self.outgoing_messages.put_nowait(
                            [
                                self.layer_gpu_workers[0],
                                Message(
                                    Message.OpCode.PushDummyPrompts,
                                    str(self.dummy_prompt_count).encode(),
                                ),
                            ]
                        )

                        # For writing prompts to storage
                        self.outgoing_messages.put_nowait(
                            [
                                self.layer_gpu_workers[0],
                                Message(
                                    Message.OpCode.ProcessPrompts,
                                    "".encode(),
                                ),
                            ]
                        )

            elif message.opcode == Message.OpCode.InferenceState:
                self.logger.error("Received InferenceState message from a worker.")

            elif message.opcode == Message.OpCode.PromptCompleted:
                proto = glinthawk_pb.PromptCompleted()
                proto.ParseFromString(message.payload)
                for id in proto.prompt_ids:
                    self.logger.info(f"Prompt {id[:8]} completed.")

            elif message.opcode == Message.OpCode.WorkerStats:
                proto = glinthawk_pb.WorkerStats()
                proto.ParseFromString(message.payload)

                stats = WorkerStats()
                stats.states_sent = proto.states_sent
                stats.states_received = proto.states_received
                stats.states_processed = proto.states_processed
                stats.tokens_processed = proto.tokens_processed
                stats.tokens_generated = proto.tokens_generated
                stats.prompts_completed = proto.prompts_completed
                stats.bytes_sent = proto.bytes_sent
                stats.bytes_received = proto.bytes_received

                worker.current_stats = stats
                worker.last_stats_time = worker.current_stats_time
                worker.current_stats_time = datetime.datetime.now()

                self.aggregate_stats += stats

    def aggregate_rates(self):
        agg_rate = WorkerStats()

        for worker in self.workers:
            agg_rate += worker.work_rate()

        self.max_rates = agg_rate.combine_max(self.max_rates)
        return agg_rate

    async def main(self, listen_address, listen_port):
        server = await asyncio.start_server(
            self.handle_worker, listen_address, listen_port
        )

        async with server:
            asyncio.create_task(self.message_processor())
            asyncio.create_task(self.handle_outgoing_messages())

            if self.ui:
                asyncio.create_task(self.ui.render_ui())

            await server.serve_forever()


@click.command()
@click.option(
    "--n-layers", "-N", help="Total layers of the model", required=True, type=click.INT
)
@click.option("--layers-per-worker", "-L", required=True, type=click.INT)
@click.option("--listen-address", required=True)
@click.option("--listen-port", required=True)
@click.option(
    "--dummy-count",
    help="Number of dummy prompts to generate",
    type=click.INT,
    default=0,
)
@click.option("--concurrency-size-pre", "-C1", type=click.INT, default=1)
@click.option("--concurrency-size-att", "-C2", type=click.INT, default=1)
@click.option("--concurrency-size-post", "-C3", type=click.INT, default=1)
@click.option("--ui/--no-ui", default=False)
def main(listen_address, listen_port, **kwargs):
    coordinator = Coordinator(**kwargs)
    asyncio.run(coordinator.main(listen_address, listen_port))


if __name__ == "__main__":
    main()
