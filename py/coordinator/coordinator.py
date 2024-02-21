import settings

import sys
import enum
import click
import socket
import asyncio
import logging
import datetime
import itertools

from enum import Enum
from rich.logging import RichHandler
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from google.protobuf.message import Message as ProtoMessage

from common.message import Message
from protobuf import glinthawk_pb2 as protobuf

from .worker import Worker
from .model import Model

logging.basicConfig(level=logging.NOTSET, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])


Platform = protobuf.Hey.Platform
Stage = protobuf.SetRoute.LayerToAddress.Stage


class Coordinator:
    def __init__(self, **kwargs):
        self.workers: List[Worker] = []

        self.first_worker = None
        self.classification_worker = None

        self.model = Model(
            name=kwargs.get("model", "stories-110m"),
            n_layers=kwargs.get("n_layers", 12),
            layers_per_worker=kwargs.get("layers_per_worker", 6),
        )

        # Message queues
        self.incoming_messages = asyncio.Queue()
        self.outgoing_messages = asyncio.Queue()

        # Dummy prompt generation
        self.initial_dummy_count = kwargs.get("dummy_count", 0)
        self.generated_dummies = 0
        self.completed_dummies = 0

        # Concurrency sizes
        self.concurrency_size_pre = kwargs.get("concurrency_size_pre", 16)
        self.concurrency_size_att = kwargs.get("concurrency_size_att", 16)
        self.concurrency_size_post = kwargs.get("concurrency_size_post", 16)
        self.concurrency_size_cls = kwargs.get("concurrency_size_cls", 16)

        self.cpu_context_count = kwargs.get("cpu_context_count", 36 * 81 * 2)
        self.gpu_context_count = kwargs.get("gpu_context_count", 18 * 81)

        # Logging
        self.logger = logging.getLogger("coordinator")
        self.logger.setLevel(logging.INFO)

    def create_routing_message(self):
        message = self.model.route_message(self.workers)
        message.route_id = 0
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

    def push_message(self, worker, opcode, payload):
        if isinstance(payload, ProtoMessage):
            payload = payload.SerializeToString()
        elif isinstance(payload, str):
            payload = payload.encode()
        elif not isinstance(payload, bytes):
            payload = str(payload).encode()

        self.outgoing_messages.put_nowait([worker, Message(opcode, payload)])

    async def handle_outgoing_messages(self):
        async def send_message(worker, message):
            worker.writer.write(message.serialize())
            await worker.writer.drain()

        while True:
            worker, message = await self.outgoing_messages.get()
            self.logger.info(f'Sending "{message!r}" to {worker.id}.')
            await send_message(worker, message)

    async def message_processor(self):
        while True:
            worker, message = await self.incoming_messages.get()
            # self.logger.info(f'Received "{message!r}" from {worker.id}.')

            if message.opcode == Message.OpCode.Hey:
                proto = protobuf.Hey()
                proto.ParseFromString(message.payload)
                worker.platform = proto.platform
                worker.ip = socket.inet_aton(proto.ip)
                worker.port = int(proto.port)

                if worker.platform not in [Platform.AMD64, Platform.CUDA] or not self.model.assign_slices(worker):
                    worker.state = Worker.State.Disconnected
                    self.push_message(worker, Message.OpCode.Bye, b"")
                    self.logger.warning(f"Dropped the connection to {worker.id}.")
                    continue

                if worker.model_slice_start[0] == 0 and worker.model_slice_start[1] == Stage.PreAttention:
                    self.first_worker = worker
                elif worker.model_slice_start[1] == Stage.Classification:
                    self.classification_worker = worker


                self.logger.info(
                    f"Worker {worker.id} is at {proto.ip}:{worker.port} (platform={Platform.Name(worker.platform)})."
                )

                # Set worker concurrency params
                max_concurrency_size_pre = 0
                max_concurrency_size_att = 0
                max_concurrency_size_post = 0
                max_concurrency_size_cls = 0
                context_count = 0

                if worker.model_slice_start[1] == Stage.Classification:
                    if worker.platform == Platform.AMD64:
                        max_concurrency_size_cls = 0
                    elif worker.platform == Platform.CUDA:
                        max_concurrency_size_cls = self.concurrency_size_cls
                else:
                    max_concurrency_size_att = self.concurrency_size_att
                    if worker.platform == Platform.AMD64:
                        context_count = self.cpu_context_count
                    elif worker.platform == Platform.CUDA:
                        context_count = self.gpu_context_count
                        max_concurrency_size_pre = self.concurrency_size_pre
                        max_concurrency_size_post = self.concurrency_size_post

                self.push_message(
                    worker,
                    Message.OpCode.InitializeWorker,
                    protobuf.InitializeWorker(
                        model_name=self.model.name,
                        start_layer=worker.model_slice_start[0],
                        end_layer=worker.model_slice_end[0],
                        concurrency_pre_att_size=max_concurrency_size_pre,
                        concurrency_att_size=max_concurrency_size_att,
                        concurrency_post_att_size=max_concurrency_size_post,
                        concurrency_cls_size=max_concurrency_size_cls,
                        max_context_count=context_count,
                        randomize=False,
                        blobstore_uri=settings.GLINTHAWK_PROMPT_BLOBSTORE,
                    ),
                )

                if self.model.all_assigned() and self.first_worker and self.classification_worker:
                    self.logger.info("All workers have been assigned layers; setting routes.")
                    routing_message = self.create_routing_message().SerializeToString()

                    for w in self.workers:
                        if w.state == Worker.State.Connected:
                            self.push_message(w, Message.OpCode.SetRoute, routing_message)

                    self.logger.info(
                        f"Layer 0 is at {socket.inet_ntoa(self.first_worker.ip)}:{self.first_worker.port};"
                        "completions can be found there."
                    )

                    # Telling the first worker to generate dummy prompts
                    if self.initial_dummy_count:
                        self.push_message(
                            self.first_worker,
                            Message.OpCode.PushDummyPrompts,
                            protobuf.PushDummyPrompts(
                                count=self.initial_dummy_count,
                                batch_size=self.concurrency_size_pre,
                            ),
                        )

                        self.generated_dummies += self.initial_dummy_count

                        # Periodically check if we need to generate more dummy prompts to keep the workers busy
                        asyncio.create_task(self.maybe_generate_dummies())

            elif message.opcode == Message.OpCode.PromptCompleted:
                count = int(message.payload.decode())
                self.logger.info(f"Worker {worker.id} completed {count} prompts.")
                self.completed_dummies += count

            else:
                self.logger.error(f"Unexpected message {message.opcode} from {worker.id}.")

    async def maybe_generate_dummies(self):
        while True:
            await asyncio.sleep(30)

            count = (self.initial_dummy_count // 2) - (self.generated_dummies - self.completed_dummies)
            if count <= 0:
                continue

            self.logger.warning(f"Generating {count} dummy prompts.")
            self.push_message(self.first_worker, Message.OpCode.PushDummyPrompts, count)
            self.generated_dummies += count

    async def main(self, listen_address, listen_port):
        server = await asyncio.start_server(self.handle_worker, listen_address, listen_port)

        async with server:
            asyncio.create_task(self.message_processor())
            asyncio.create_task(self.handle_outgoing_messages())
            await server.serve_forever()
