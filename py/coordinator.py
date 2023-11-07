#!/usr/bin/env python3

import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required to run this program.")

import enum
import json
import socket
import asyncio
import logging
import itertools

from dataclasses import dataclass, field

from common.message import Message
from common.inference import InferenceState

from protobuf import glinthawk_pb2 as glinthawk_pb

import settings

logging.basicConfig(level=logging.INFO)


@dataclass
class ModelInfo:
    name: str
    n_layers: int
    layers_per_worker: int


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


prompts = [
    "2EUiyvY2VxVjgRTrCYgRQYE94V2D2BKRLx2CmWgieXpv",
    "5LXEQ2fcpjDShQtvbEsingGYum5g4Po3bcVzEVg2Frtz",
    "7LiXa2RktTe9VYms5jVge8zWJgvQv1tMG3oxafFaLoRX",
    "AuRqaXzRYmXUdq8xvUaiVjiyDGnJon6zcGuF56817Mum",
    "BN4L7ACfBqTJf4Vs9vTQAjnHRSb6KA14Sd2rNp7DidTR",
    "Dqw7wrwE8R3N19PcdKQCNcGmjxJeLhk8EMT7xEaG5J3n",
    "DsJ1mCF1td68kioyVMtPaUYGUgKvcmaXYhVMYfzq3nUW",
    "Ea9GDb1dQzZN7vMY2dZsvzuWSqBkdrcoAt5sJnrb9cdn",
    "H6Lt2aQYLJYCNdSjCAa7cFL9m6Rmvh3MtzehxktdHzh5",
    "Q9gt4RUKdvM27u4oMPHkjpWxQukUJyQrwYJEV7AG2s3",
]

prompts_blobstore_uri = "file:///home/sadjad/prompts/"

model = ModelInfo(name="stories-110M-glint", n_layers=12, layers_per_worker=4)
workers = []
layer_workers = {}
incoming_messages = asyncio.Queue()


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
    global workers

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


async def message_processor():
    global initialized_workers

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
                asyncio.create_task(
                    send_message(
                        layer_workers[0],
                        Message(
                            Message.OpCode.SetRoute,
                            create_routing_message().SerializeToString(),
                        ),
                    )
                )

                asyncio.create_task(
                    send_message(
                        layer_workers[0],
                        Message(
                            Message.OpCode.ProcessPrompts,
                            glinthawk_pb.ProcessPrompts(
                                prompt_ids=prompts
                            ).SerializeToString(),
                        ),
                    )
                )

        elif message.opcode == Message.OpCode.InferenceState:
            logging.error("Received InferenceState message from a worker.")

        elif message.opcode == Message.OpCode.PromptCompleted:
            proto = glinthawk_pb.PromptCompleted()
            proto.ParseFromString(message.payload)
            for id in proto.prompt_ids:
                logging.info(f"Prompt {id[:8]} completed.")


async def main(listen_address, listen_port):
    server = await asyncio.start_server(handle_worker, listen_address, listen_port)

    async with server:
        asyncio.create_task(message_processor())
        await server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <listen_address> <listen_port>")

    listen_address = sys.argv[1]
    listen_port = int(sys.argv[2])

    asyncio.run(main(listen_address, listen_port))
