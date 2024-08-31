import asyncio
import enum
import json
import logging
import os
import socket
import time
import datetime

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from signal import SIGINT, SIGTERM

import rich

from rich.text import Text
from rich.status import Status
from rich.panel import Panel
from rich.align import Align
from rich.console import Group
from rich.table import Table
from rich.live import Live

from google.protobuf.message import Message as ProtoMessage
from google.protobuf.json_format import MessageToJson, MessageToDict

from common.message import Message
from protobuf import glinthawk_pb2 as protobuf

from .model import Model
from .worker import Worker

Platform = protobuf.Hey.Platform
Stage = protobuf.SetRoute.LayerToAddress.Stage


class Coordinator:
    class State(enum.Enum):
        Starting = enum.auto()
        Running = enum.auto()
        Stopping = enum.auto()
        Stopped = enum.auto()

    def __init__(self, **kwargs):
        self.state = Coordinator.State.Starting

        # Logging
        self.logger = logging.getLogger("coordinator")
        self.logger.setLevel(logging.INFO)

        # Workers
        self.workers: List[Worker] = []
        self.first_worker = None  # Handle to the worker at slice=0, tier=0, rank=0

        # Model
        self.model = Model(
            model_name=kwargs.get("model_name"),
            n_layers=kwargs.get("n_layers"),
            n_slices=kwargs.get("n_slices"),
            tier_config=kwargs.get("tiers"),
            separate_cls_tiers=kwargs.get("separate_cls_tiers"),
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

        # Prompts and completions
        self.prompt_queue = []
        self.completion_queue = asyncio.Queue()

        self.prompt_dir = kwargs.get("prompt_dir")
        self.output_dir = kwargs.get("output_dir") + "/" + kwargs.get("config_name") + "/" + time.strftime(
            '%Y-%m-%d-%H-%M-%S', time.gmtime()) + "/"
        os.makedirs(self.output_dir, exist_ok=True)

        self.load_prompts(self.prompt_dir, self.output_dir)

        self.state = Coordinator.State.Running
        self.start_time = datetime.datetime.now()

    def is_running(self):
        return self.state == Coordinator.State.Running

    def create_routing_message(self):
        message = self.model.route_message(self.workers)
        message.route_id = 0
        return message

    def load_prompts(self, prompt_dir: str, output_dir: str) -> None:
        if not prompt_dir or not output_dir:
            return

        size_bytes = 0
        skipped_count = 0
        loaded_count = 0

        # (1) let's see what prompts have already been processed
        completed_prompts = set([])

        for filename in os.listdir(output_dir):
            path = os.path.join(output_dir, filename)
            if filename.endswith(".jsonl") and os.path.isfile(path):
                with open(path, "r") as f:
                    for line in f:
                        p = self.make_prompt_from_json(line)
                        completed_prompts.add(p.id)

        for filename in os.listdir(prompt_dir):
            path = os.path.join(prompt_dir, filename)
            if filename.endswith(".jsonl") and os.path.isfile(path):
                with open(path, "r") as f:
                    for line in f:
                        p = self.make_prompt_from_json(line)

                        if p.id in completed_prompts:
                            skipped_count += 1
                            continue
                        else:
                            loaded_count += 1
                            size_bytes += len(line)

                        self.prompt_queue.append(p)

        if skipped_count > 0:
            self.logger.info(
                f"Skipped {skipped_count} prompt{'s' if skipped_count != 1 else ''} that have already been processed."
            )

        self.logger.info(
            f"Loaded {loaded_count} prompt{'s' if loaded_count != 1 else ''} from {prompt_dir} ({size_bytes / 1024 / 1024:.2f} MiB)."
        )

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

    def push_message(self, worker, opcode, payload):
        if isinstance(payload, ProtoMessage):
            payload = payload.SerializeToString()
        elif isinstance(payload, str):
            payload = payload.encode()
        elif not isinstance(payload, bytes):
            payload = str(payload).encode()

        self.outgoing_messages.put_nowait([worker, Message(opcode, payload)])

    async def handle_worker(self, reader, writer):
        addr = writer.get_extra_info("peername")
        self.logger.info(f"New connection from {addr!r}.")

        worker = Worker(reader=reader, writer=writer)
        self.workers += [worker]

        while self.state != Coordinator.State.Stopped:
            try:
                message_header = await reader.readexactly(5)
                payload_length, opcode = Message.parse_header(message_header)
                message_payload = await reader.readexactly(payload_length)
                message = Message(opcode=opcode, payload=message_payload)
                await self.incoming_messages.put([worker, message])
            except:
                break

        worker.state = Worker.State.Disconnected

        if self.state not in [Coordinator.State.Stopping, Coordinator.State.Stopped]:
            asyncio.create_task(self.request_shutdown(None))

        self.logger.info(f"Connection from {addr!r} closed.")

    async def handle_outgoing_messages(self):
        async def send_message(worker, message):
            worker.writer.write(message.serialize())
            await worker.writer.drain()

        while self.state != Coordinator.State.Stopped:
            worker, message = await self.outgoing_messages.get()
            self.logger.info(f'Sending "{message!r}" to {worker.id}.')
            await send_message(worker, message)

    async def message_processor(self):
        while self.state != Coordinator.State.Stopped:
            worker, message = await self.incoming_messages.get()

            if message.opcode == Message.OpCode.Hey:
                proto = protobuf.Hey()
                proto.ParseFromString(message.payload)
                worker.platform = proto.platform
                worker.kernel = proto.kernel
                worker.ip = socket.inet_aton(proto.ip)
                worker.port = int(proto.port)

                if not self.model.assign_slices(worker):
                    worker.state = Worker.State.Disconnected
                    if worker in self.workers:
                        self.workers.remove(worker)
                    self.push_message(worker, Message.OpCode.Bye, b"")
                    self.logger.warning(f"Dropped the connection to {worker.id}.")
                    continue

                if worker.is_first_parent():
                    self.first_worker = worker

                self.push_message(
                    worker,
                    Message.OpCode.InitializeWorker,
                    protobuf.InitializeWorker(
                        model_name=self.model.model_name,
                        start_layer=worker.model_slice_start[0],
                        end_layer=worker.model_slice_end[0],
                        tier_concurrency_s=self.model.get_tier_concurrencies_message(),
                        tier=worker.tier,
                        rank=worker.rank,
                        randomize=False,
                    ),
                )

                self.logger.info(f"Worker {worker.id} is at {proto.ip}:{worker.port} [{worker}].")

            elif message.opcode == Message.OpCode.AckInitialize:
                worker.handshake_status = worker.Handshake.LayerAssigned

                if self.model.all_assigned() and all(w_.handshake_status == Worker.Handshake.LayerAssigned for w_ in
                                                     self.workers):
                    assert self.first_worker is not None
                    self.logger.info("All workers have been assigned layers; setting routes.")
                    routing_message = self.create_routing_message().SerializeToString()

                    for w in self.workers:
                        if w.state == Worker.State.Connected:
                            self.push_message(w, Message.OpCode.SetRoute, routing_message)

                    self.logger.info(
                        f"The first layer is at {socket.inet_ntoa(self.first_worker.ip)}:{self.first_worker.port}."
                    )

            elif message.opcode == Message.OpCode.AckRoute:
                worker.handshake_status = Worker.Handshake.RouteAssigned

                if all(w_.handshake_status == Worker.Handshake.RouteAssigned for w_ in
                       self.workers) and self.initial_dummy_count:
                    # Telling the first worker to generate dummy prompts
                    self.push_message(
                        self.first_worker,
                        Message.OpCode.PushDummyPrompts,
                        protobuf.PushDummyPrompts(
                            count=self.initial_dummy_count,
                        ),
                    )

                    self.generated_dummies += self.initial_dummy_count

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

            elif message.opcode == Message.OpCode.Bye:
                worker.state = Worker.State.Disconnected
                self.logger.info(f"Worker {worker.id} said goodbye.")

            else:
                self.logger.error(f"Unexpected message {message.opcode} from {worker.id}.")

    async def maybe_generate_dummies(self):
        while self.initial_dummy_count > 0:
            await asyncio.sleep(10)

            if not self.is_running():
                break

            if not (self.model.all_assigned() and all(w_.handshake_status == Worker.Handshake.RouteAssigned for w_ in
                                                      self.workers)):
                continue

            count = (self.initial_dummy_count // 2) - (self.generated_dummies - self.completed_dummies)
            if count <= 0:
                continue

            self.logger.warning(f"Generating {count} dummy prompts.")

            self.push_message(
                self.first_worker,
                Message.OpCode.PushDummyPrompts,
                protobuf.PushDummyPrompts(
                    count=count,
                ),
            )

            self.generated_dummies += count

    async def maybe_send_prompts(self):
        while True:
            await asyncio.sleep(10)

            if not self.is_running():
                break

            if not (self.model.all_assigned() and all(w_.handshake_status == Worker.Handshake.RouteAssigned for w_ in
                                                      self.workers)):
                continue

            if not self.prompt_queue:
                self.logger.info("All prompts have been submitted for processing.")
                break

            proto = protobuf.PushPrompts()

            while (
                    len(self.prompt_queue) > 0 and self.assigned_prompts - self.completed_prompts < self.prompt_batch_size
            ):
                proto.prompts.append(self.prompt_queue.pop(0))
                self.assigned_prompts += 1

            if not proto.prompts:
                continue

            self.logger.info(f"Sending {len(proto.prompts)} prompts to the first worker.")
            self.push_message(self.first_worker, Message.OpCode.PushPrompts, proto)

    async def dump_completions(self):
        with open(os.path.join(self.output_dir, f"completions.jsonl"), "a") as f:
            while self.state != Coordinator.State.Stopped:
                completion = await self.completion_queue.get()
                f.write(json.dumps(MessageToDict(completion), indent=None, separators=(",", ":")))
                f.write("\n")
                f.flush()

    async def show_status(self):
        with Live(transient=True, auto_refresh=False) as live:
            while self.state != Coordinator.State.Stopped:
                elapsed_time = datetime.datetime.now() - self.start_time
                elapsed_time = "{:02d}:{:02d}:{:02d}".format(
                    int(elapsed_time.total_seconds()) // 3600,
                    int(elapsed_time.total_seconds()) // 60 % 60,
                    int(elapsed_time.total_seconds()) % 60,
                )

                grid = Table(
                    title=f"\n{elapsed_time}",
                    expand=False,
                    show_header=False,
                    show_lines=True,
                    box=rich.box.ROUNDED,
                    title_justify="right",
                )
                grid.add_column(justify="left")
                grid.add_column(justify="left")

                grid.add_row(
                    "Model",
                    Text(f"{self.model.n_layers} layers, {self.model.layers_per_worker} per worker"),
                )

                grid.add_row(
                    "Prompts",
                    Text(
                        f"{self.assigned_prompts} assigned, {self.completed_prompts} completed, {len(self.prompt_queue)} queued",
                    ),
                )

                grid.add_row("Workers", Text(f"{len(self.workers)} connected"))

                live.update(Align.right(grid), refresh=True)
                await asyncio.sleep(1)

    async def request_shutdown(self, sig):
        if self.state == Coordinator.State.Stopping:
            self.logger.warning(f"Shutdown was already in progress; force quitting.")
            return
        else:
            if sig is not None:
                self.logger.warning(f"Received signal {sig!r}; shutting down gracefully...")
            else:
                self.logger.warning(f"Shutting down gracefully...")

            self.state = Coordinator.State.Stopping

        loop = asyncio.get_running_loop()
        loop.remove_signal_handler(SIGINT)
        loop.remove_signal_handler(SIGTERM)

        # Send a bye message to all workers, asking them to finish up
        for worker in self.workers:
            if worker.state == Worker.State.Connected:
                self.push_message(worker, Message.OpCode.Bye, b"")

        # Wait for all workers to finish up
        while any(worker.state != Worker.State.Disconnected for worker in self.workers):
            await asyncio.sleep(1)

        for worker in self.workers:
            worker.reader.feed_eof()
            worker.writer.close()
            await worker.writer.wait_closed()

        if len(self.workers) > 0:
            self.logger.info("All workers have disconnected; shutting down...")

        self.state = Coordinator.State.Stopped
        self.server.close()
        await self.server.wait_closed()

        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
                await task

    async def main(self, listen_address, listen_port):
        self.server = await asyncio.start_server(self.handle_worker, listen_address, listen_port)
        self.logger.info(f"Listening on {listen_address}:{listen_port}.")

        async with self.server:
            loop = asyncio.get_running_loop()
            for sig in (SIGINT, SIGTERM):
                loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(self.request_shutdown(sig)))

            asyncio.create_task(self.show_status())
            asyncio.create_task(self.message_processor())
            asyncio.create_task(self.handle_outgoing_messages())
            asyncio.create_task(self.dump_completions())
            asyncio.create_task(self.maybe_generate_dummies())
            asyncio.create_task(self.maybe_send_prompts())

            try:
                await self.server.serve_forever()
            except asyncio.CancelledError:
                pass
