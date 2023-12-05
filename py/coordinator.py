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
from protobuf import glinthawk_pb2 as glinthawk_pb

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.NOTSET, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)


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
    end_layer: int = 0
    max_concurrency_size: int = 16

    current_stats: WorkerStats = field(default_factory=WorkerStats)
    current_stats_time: datetime.datetime = None
    last_stats_time: datetime.datetime = None

    def work_rate(self):
        pass


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

        self.concurrency_size = kwargs.get("concurrency_size", 16)
        self.initial_dummy_count = kwargs.get("dummy_count", 0)

        self.generated_dummies = 0
        self.completed_dummies = 0

        self.logger = logging.getLogger("coordinator")
        self.logger.setLevel(logging.INFO)

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
                worker.max_concurrency_size = self.concurrency_size

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

                    self.logger.info(
                        f"Layer 0 is being served at {socket.inet_ntoa(self.layer_workers[0].ip)}:{self.layer_workers[0].port}; completions can be found there."
                    )

                    # telling the first worker to generate dummy prompts
                    if self.initial_dummy_count:
                        self.outgoing_messages.put_nowait(
                            [
                                self.layer_workers[0],
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
            if (
                self.generated_dummies - self.completed_dummies
                < self.initial_dummy_count // 2
            ):
                gen_count = (self.initial_dummy_count // 2) - (
                    self.generated_dummies - self.completed_dummies
                )
                if gen_count <= 0:
                    continue

                self.logger.warning(f"Generating {gen_count} dummy prompts.")
                self.outgoing_messages.put_nowait(
                    [
                        self.layer_workers[0],
                        Message(
                            Message.OpCode.PushDummyPrompts,
                            str(gen_count).encode(),
                        ),
                    ]
                )

                self.generated_dummies += gen_count

            await asyncio.sleep(30)

    async def main(self, listen_address, listen_port):
        server = await asyncio.start_server(
            self.handle_worker, listen_address, listen_port
        )

        async with server:
            asyncio.create_task(self.message_processor())
            asyncio.create_task(self.handle_outgoing_messages())
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
@click.option("--concurrency-size", "-C", type=click.INT, default=1)
def main(listen_address, listen_port, **kwargs):
    coordinator = Coordinator(**kwargs)
    asyncio.run(coordinator.main(listen_address, listen_port))


if __name__ == "__main__":
    main()
