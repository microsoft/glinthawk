import settings

import os
import sys
import json
import socket
import asyncio
import logging

from enum import Enum
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from signal import SIGINT, SIGTERM

from rich.logging import RichHandler
from google.protobuf.message import Message as ProtoMessage
from google.protobuf.json_format import MessageToJson, MessageToDict

from common.message import Message
from protobuf import glinthawk_pb2 as protobuf

from .worker import Worker
from .model import Model

Platform = protobuf.Hey.Platform
Stage = protobuf.SetRoute.LayerToAddress.Stage


class Coordinator:
    def __init__(self, **kwargs):
        # Logging
        self.logger = logging.getLogger("coordinator")
        self.logger.setLevel(logging.INFO)

        # Workers
        self.workers: List[Worker] = []
        self.first_worker = None  # Handle to the first worker in the chain
        self.classification_worker = None  # Handle to the classification worker
        self.separate_classification_worker = kwargs.get("separate_cls", False)

        # Model
        self.model = Model(
            n_layers=kwargs["n_layers"],
            layers_per_worker=kwargs["layers_per_worker"],
            separate_cls=self.separate_classification_worker,
        )

        # Job info
        self.prompt_batch_size = 128
        self.assigned_prompts = 0
        self.completed_prompts = 0

        # Message queues
        self.incoming_messages = asyncio.Queue()
        self.outgoing_messages = asyncio.Queue()

        # Dummy prompt generation
        self.initial_dummy_count = kwargs.get("dummy_count", 0)
        self.generated_dummies = 0
        self.completed_dummies = 0

        # Concurrency sizes and context counts
        self.concurrency_size_pre = kwargs.get("concurrency_size_pre", 16)
        self.concurrency_size_att = kwargs.get("concurrency_size_att", 16)
        self.concurrency_size_post = kwargs.get("concurrency_size_post", 16)
        self.concurrency_size_cls = kwargs.get("concurrency_size_cls", 16)

        self.cpu_context_count = kwargs.get("cpu_context_count", 36 * 81 * 2)
        self.gpu_context_count = kwargs.get("gpu_context_count", 18 * 81)

        # Prompts and completions
        self.prompt_queue = []
        self.completion_queue = asyncio.Queue()
        self.load_prompts(kwargs.get("prompt_dir"))

        self.output_dir = kwargs.get("output_dir")
        os.makedirs(self.output_dir, exist_ok=True)

    def create_routing_message(self):
        message = self.model.route_message(self.workers)
        message.route_id = 0
        return message

    def load_prompts(self, prompt_dir: str):
        if not prompt_dir:
            return

        size_bytes = 0

        for filename in os.listdir(prompt_dir):
            path = os.path.join(prompt_dir, filename)
            if filename.endswith(".jsonl") and os.path.isfile(path):
                with open(path, "r") as f:
                    for line in f:
                        size_bytes += len(line)
                        p = self.make_prompt_from_json(line)
                        self.prompt_queue.append(p)

        self.logger.info(f"Loaded {len(self.prompt_queue)} prompts ({size_bytes} bytes).")

    def make_prompt_from_json(self, jsondata: str) -> protobuf.Prompt:
        data = json.loads(jsondata)
        prompt = protobuf.Prompt()

        if "id" not in data:
            raise ValueError("Prompt does not have an ID")

        prompt.id = data["id"]
        prompt.temperature = data.get("temperature", 0)
        prompt.max_tokens = data.get("max_tokens", 2048)
        prompt.prompt[:] = data.get("prompt", [1])
        prompt.completion[:] = []
        prompt.user_data = data.get("user_data", "")

        return prompt

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
                    # TODO(sadjad): remove the worker from the list
                    continue

                if worker.model_slice_start[0] == 0 and worker.model_slice_start[1] == Stage.PreAttention:
                    self.first_worker = worker
                elif worker.model_slice_start[1] == Stage.Classification:
                    self.classification_worker = worker

                context_count = self.gpu_context_count if worker.platform == Platform.CUDA else self.cpu_context_count

                worker.concurrency_size_pre = self.concurrency_size_pre
                worker.concurrency_size_att = self.concurrency_size_att
                worker.concurrency_size_post = self.concurrency_size_post
                worker.concurrency_size_cls = self.concurrency_size_cls

                self.push_message(
                    worker,
                    Message.OpCode.InitializeWorker,
                    protobuf.InitializeWorker(
                        model_name="model",
                        start_layer=worker.model_slice_start[0],
                        end_layer=worker.model_slice_end[0],
                        concurrency_pre_att_size=worker.concurrency_size_pre,
                        concurrency_att_size=worker.concurrency_size_att,
                        concurrency_post_att_size=worker.concurrency_size_post,
                        concurrency_cls_size=worker.concurrency_size_cls,
                        max_context_count=context_count,
                        randomize=False,
                    ),
                )

                self.logger.info(f"Worker {worker.id} is at {proto.ip}:{worker.port} [{worker}].")

                if (
                    self.model.all_assigned()
                    and self.first_worker
                    and (not self.separate_classification_worker or self.classification_worker)
                ):
                    self.logger.info("All workers have been assigned layers; setting routes.")
                    routing_message = self.create_routing_message().SerializeToString()

                    for w in self.workers:
                        if w.state == Worker.State.Connected:
                            self.push_message(w, Message.OpCode.SetRoute, routing_message)

                    self.logger.info(
                        f"The first layer is at {socket.inet_ntoa(self.first_worker.ip)}:{self.first_worker.port}."
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
                    else:
                        asyncio.create_task(self.maybe_send_prompts())

            elif message.opcode == Message.OpCode.PushCompletions:
                proto = protobuf.PushCompletions()
                proto.ParseFromString(message.payload)
                self.logger.info(f"Worker {worker.id} completed {len(proto.completions)} prompts.")

                if self.initial_dummy_count:
                    self.completed_dummies += len(proto.completions)
                else:
                    self.completed_prompts += len(proto.completions)

                for completion in proto.completions:
                    self.completion_queue.put_nowait(completion)

            else:
                self.logger.error(f"Unexpected message {message.opcode} from {worker.id}.")

    async def maybe_generate_dummies(self):
        while True:
            await asyncio.sleep(30)

            count = (self.initial_dummy_count // 2) - (self.generated_dummies - self.completed_dummies)
            if count <= 0:
                continue

            self.logger.warning(f"Generating {count} dummy prompts.")

            self.push_message(
                self.first_worker,
                Message.OpCode.PushDummyPrompts,
                protobuf.PushDummyPrompts(
                    count=count,
                    batch_size=self.concurrency_size_pre,
                ),
            )

            self.generated_dummies += count

    async def maybe_send_prompts(self):
        while True:
            if not self.prompt_queue:
                self.logger.info("All prompts have been submitted for processing.")
                return

            proto = protobuf.PushPrompts()

            while self.assigned_prompts - self.completed_prompts < self.prompt_batch_size and self.prompt_queue:
                proto.prompts.append(self.prompt_queue.pop(0))
                self.assigned_prompts += 1

            if not proto.prompts:
                continue

            self.logger.info(f"Sending {len(proto.prompts)} prompts to the first worker.")
            self.push_message(self.first_worker, Message.OpCode.PushPrompts, proto)

            await asyncio.sleep(30)

    async def dump_completions(self):
        with open(os.path.join(self.output_dir, f"completions.json"), "w") as f:
            while True:
                completion = await self.completion_queue.get()
                f.write(json.dumps(MessageToDict(completion), indent=None, separators=(",", ":")))
                f.write("\n")
                f.flush()

    async def main(self, listen_address, listen_port):
        server = await asyncio.start_server(self.handle_worker, listen_address, listen_port)

        async with server:
            asyncio.create_task(self.message_processor())
            asyncio.create_task(self.handle_outgoing_messages())
            asyncio.create_task(self.dump_completions())
            await server.serve_forever()
