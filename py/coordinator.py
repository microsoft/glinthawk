#!/usr/bin/env python3

import settings

import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required to run this program.")

import io
import enum
import json
import click
import socket
import signal
import asyncio
import logging
import datetime
import itertools
import functools

from dataclasses import dataclass, field

from common.message import Message
from common.inference import InferenceState
from protobuf import glinthawk_pb2 as glinthawk_pb

from coordinator.ui import CoordinatorUI


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

    def __add__(self, other):
        return WorkerStats(
            states_sent=self.states_sent + other.states_sent,
            states_received=self.states_received + other.states_received,
            states_processed=self.states_processed + other.states_processed,
            tokens_processed=self.tokens_processed + other.tokens_processed,
            tokens_generated=self.tokens_generated + other.tokens_generated,
            prompts_completed=self.prompts_completed + other.prompts_completed,
        )

    def combine_max(self, other):
        return WorkerStats(
            states_sent=max(self.states_sent, other.states_sent),
            states_received=max(self.states_received, other.states_received),
            states_processed=max(self.states_processed, other.states_processed),
            tokens_processed=max(self.tokens_processed, other.tokens_processed),
            tokens_generated=max(self.tokens_generated, other.tokens_generated),
            prompts_completed=max(self.prompts_completed, other.prompts_completed),
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
    end_layer: int = 0
    max_concurrency_size: int = 1

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
        )


class Coordinator:
    def __init__(self, **kwargs):
        self.workers = []
        self.layer_workers = {}
        self.incoming_messages = asyncio.Queue()
        self.outgoing_messages = asyncio.Queue()
        self.model = ModelInfo(
            name="stories-110M-glint",
            n_layers=kwargs.get("n_layers", 12),
            layers_per_worker=kwargs.get("layers_per_worker", 6),
        )

        self.aggregate_stats = WorkerStats()
        self.max_rates = WorkerStats()

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
        for layer, worker in self.layer_workers.items():
            sub_msg = glinthawk_pb.SetRoute.LayerToAddress()
            sub_msg.layer_num = layer
            sub_msg.ip = socket.inet_ntoa(worker.ip)
            sub_msg.port = worker.port
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

            if message.opcode == Message.OpCode.Hey:
                address = message.payload.decode()
                ip, port = address.split(":")
                worker.ip = socket.inet_aton(ip)
                worker.port = int(port)
                self.logger.info(f"Worker {worker.id} is at {ip}:{port}.")

                # assinging layers to this worker
                worker.start_layer = worker.id * self.model.layers_per_worker
                worker.end_layer = (worker.id + 1) * self.model.layers_per_worker - 1

                initialization_message = glinthawk_pb.InitializeWorker(
                    model_name=self.model.name,
                    start_layer=worker.start_layer,
                    end_layer=worker.end_layer,
                    concurrency_size=worker.max_concurrency_size,
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

                self.layer_workers[worker.start_layer] = worker

                if (
                    len(self.layer_workers)
                    == self.model.n_layers / self.model.layers_per_worker
                ):
                    # all layers have been assigned
                    # setting the route for the first worker
                    self.outgoing_messages.put_nowait(
                        [
                            self.layer_workers[0],
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
                                self.layer_workers[0],
                                Message(
                                    Message.OpCode.PushDummyPrompts,
                                    str(self.dummy_prompt_count).encode(),
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
    "--dummy-count", help="Number of dummy prompts to generate", type=click.INT
)
@click.option("--ui/--no-ui", default=False)
def main(listen_address, listen_port, **kwargs):
    coordinator = Coordinator(**kwargs)
    asyncio.run(coordinator.main(listen_address, listen_port))


if __name__ == "__main__":
    main()
