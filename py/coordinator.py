#!/usr/bin/env python3

import settings

import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required to run this program.")

import enum
import json
import socket
import asyncio
import logging
import datetime
import itertools

from dataclasses import dataclass, field

from common.message import Message
from common.inference import InferenceState
from protobuf import glinthawk_pb2 as glinthawk_pb

import rich
from rich.live import Live as RichLive
from rich.table import Table as RichTable
from rich.layout import Layout as RichLayout


logging.basicConfig(level=logging.WARNING)


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


model = ModelInfo(name="stories-110M-glint", n_layers=12, layers_per_worker=6)
workers = []
layer_workers = {}
incoming_messages = asyncio.Queue()
outgoing_messages = asyncio.Queue()
aggregate_stats = WorkerStats()


def create_routing_message():
    global layer_workers
    message = glinthawk_pb.SetRoute()
    for layer, worker in layer_workers.items():
        sub_msg = glinthawk_pb.SetRoute.LayerToAddress()
        sub_msg.layer_num = layer
        sub_msg.ip = socket.inet_ntoa(worker.ip)
        sub_msg.port = worker.port
        message.layer_to_address.append(sub_msg)

    return message


async def handle_worker(reader, writer):
    global workers, incoming_messages

    addr = writer.get_extra_info("peername")
    logging.info(f"New connection from {addr!r}.")

    worker = Worker(reader=reader, writer=writer)
    workers += [worker]

    while True:
        try:
            message_header = await reader.readexactly(5)
            payload_length, opcode = Message.parse_header(message_header)
            message_payload = await reader.readexactly(payload_length)
            message = Message(opcode=opcode, payload=message_payload)
            await incoming_messages.put([worker, message])
        except:
            worker.state = Worker.State.Disconnected
            return


async def send_message(worker, message):
    worker.writer.write(message.serialize())
    await worker.writer.drain()


async def handle_outgoing_messages():
    while True:
        worker, message = await outgoing_messages.get()
        logging.info(f'Sending "{message!r}" to {worker.id}.')
        await send_message(worker, message)


async def message_processor():
    global initialized_workers, aggregate_stats

    while True:
        worker, message = await incoming_messages.get()
        logging.info(f'Received "{message!r}" from {worker.id}.')

        if message.opcode == Message.OpCode.Hey:
            address = message.payload.decode()
            ip, port = address.split(":")
            worker.ip = socket.inet_aton(ip)
            worker.port = int(port)
            logging.info(f"Worker {worker.id} is at {ip}:{port}.")

            # assinging layers to this worker
            worker.start_layer = worker.id * model.layers_per_worker
            worker.end_layer = (worker.id + 1) * model.layers_per_worker - 1

            initialization_message = glinthawk_pb.InitializeWorker(
                model_name=model.name,
                start_layer=worker.start_layer,
                end_layer=worker.end_layer,
                concurrency_size=worker.max_concurrency_size,
                blobstore_uri=settings.GLINTHAWK_PROMPT_BLOBSTORE,
            )

            asyncio.create_task(
                send_message(
                    worker,
                    Message(
                        Message.OpCode.InitializeWorker,
                        initialization_message.SerializeToString(),
                    ),
                )
            )

            layer_workers[worker.start_layer] = worker

            if len(layer_workers) == model.n_layers / model.layers_per_worker:
                # all layers have been assigned

                # setting the route for the first worker
                outgoing_messages.put_nowait(
                    [
                        layer_workers[0],
                        Message(
                            Message.OpCode.SetRoute,
                            create_routing_message().SerializeToString(),
                        ),
                    ]
                )

                # telling the first worker to generate dummy prompts
                outgoing_messages.put_nowait(
                    [
                        layer_workers[0],
                        Message(Message.OpCode.PushDummyPrompts, b""),
                    ]
                )

                # outgoing_messages.put_nowait(
                #     [
                #         layer_workers[0],
                #         Message(
                #             Message.OpCode.ProcessPrompts,
                #             glinthawk_pb.ProcessPrompts(
                #                 prompt_ids=prompts
                #             ).SerializeToString(),
                #         ),
                #     ]
                # )

        elif message.opcode == Message.OpCode.InferenceState:
            logging.error("Received InferenceState message from a worker.")

        elif message.opcode == Message.OpCode.PromptCompleted:
            proto = glinthawk_pb.PromptCompleted()
            proto.ParseFromString(message.payload)
            for id in proto.prompt_ids:
                logging.info(f"Prompt {id[:8]} completed.")

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

            aggregate_stats += stats


def aggregate_rates():
    global workers, aggregate_stats

    agg_rate = WorkerStats()

    for worker in workers:
        agg_rate += worker.work_rate()

    return agg_rate


async def render_ui():
    global aggregate_stats, workers

    layout = RichLayout()
    layout.split_row(
        RichLayout(name="left"),
        RichLayout(name="right"),
    )

    layout["left"].split_column(
        RichLayout(name="top"),
        RichLayout(name="bottom"),
    )

    with RichLive(layout, refresh_per_second=1, transient=True):
        while True:
            await asyncio.sleep(1)
            rates = aggregate_rates()

            stats_table = RichTable(title="Status")
            stats_table.add_column("Metric")
            stats_table.add_column("Value")
            stats_table.add_row("Active Workers", f"{len(workers)}")
            stats_table.add_row(
                "\u03a3 States Processed", f"{aggregate_stats.states_processed:.2f}"
            )
            stats_table.add_row(
                "\u03a3 Tokens Processed", f"{aggregate_stats.tokens_processed:.2f}"
            )
            stats_table.add_row(
                "\u03a3 Tokens Generated", f"{aggregate_stats.tokens_generated:.2f}"
            )
            stats_table.add_row("Active Prompts", "1")

            layout["left"]["top"].update(stats_table)

            rate_table = RichTable(title="Rates")
            rate_table.add_column("Metric")
            rate_table.add_column("Rate (Hz)")
            rate_table.add_row("States Processed", f"{rates.states_processed:.2f}")
            rate_table.add_row("Tokens Processed", f"{rates.tokens_processed:.2f}")
            rate_table.add_row("Tokens Generated", f"{rates.tokens_generated:.2f}")

            layout["left"]["bottom"].update(rate_table)


async def main(listen_address, listen_port):
    server = await asyncio.start_server(handle_worker, listen_address, listen_port)

    async with server:
        asyncio.create_task(message_processor())
        asyncio.create_task(handle_outgoing_messages())
        asyncio.create_task(render_ui())
        await server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <listen_address> <listen_port>")

    listen_address = sys.argv[1]
    listen_port = int(sys.argv[2])

    asyncio.run(main(listen_address, listen_port))
