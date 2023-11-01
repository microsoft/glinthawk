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
from sentencepiece import SentencePieceProcessor

from dataclasses import dataclass, field

from common.message import Message
from common.inference import InferenceState

from protobuf import glinthawk_pb2 as proto

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


model = ModelInfo(name="stories-110M-glint", n_layers=12, layers_per_worker=4)
# model = ModelInfo(name="llama2-7b-chat", n_layers=20, layers_per_worker=10)
workers = []
layer_workers = {}
incoming_messages = asyncio.Queue()

prompts_blobstore_uri = "file:///tmp/prompts/"


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
            # A little trick to test GPU setup without running out of memory
            # if worker.id == 0:
            #     worker.start_layer = worker.id * model.layers_per_worker
            #     worker.end_layer = (worker.id + 1) * model.layers_per_worker - 1
            # else:
            #     worker.start_layer = 22
            #     worker.end_layer = 31

            initialization_message = proto.InitializeWorker(
                model_name=model.name,
                start_layer=worker.start_layer,
                end_layer=worker.end_layer,
                concurrency_size=worker.max_concurrency_size,
                blobstore_uri=prompts_blobstore_uri,
            )

            response = Message(
                Message.OpCode.InitializeWorker,
                initialization_message.SerializeToString(),
            )

            asyncio.create_task(send_message(worker, response))

            layer_workers[worker.start_layer] = [worker.ip, worker.port]

            if len(layer_workers) == model.n_layers / model.layers_per_worker:
                for context_test in range(10):
                    for conc_i in range(
                        worker.max_concurrency_size * len(layer_workers)
                    ):
                        # we're ready for lift-off
                        state = InferenceState(layer_workers=layer_workers)
                        message = Message(
                            Message.OpCode.InferenceState, state.serialize()
                        )

                        for w in workers:
                            if w.start_layer == 0:
                                logging.info(f"Sending InferenceState to {w.id}.")
                                asyncio.create_task(send_message(w, message))
                                break
        elif message.opcode == Message.OpCode.InferenceState:
            state = InferenceState()
            state.load_from_payload(message.payload)
            logging.info(
                f"Worker {worker.id} returned token {tokenizer.decode([state.token])}(pos={state.token_pos}) for prompt {state.prompt_id.hex()[:8]}."
            )


async def main(listen_address, listen_port):
    server = await asyncio.start_server(handle_worker, listen_address, listen_port)

    async with server:
        asyncio.create_task(message_processor())
        await server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} tokenizer_path <listen_address> <listen_port>")

    listen_address = sys.argv[2]
    listen_port = int(sys.argv[3])
    tokenizer = SentencePieceProcessor(model_file=sys.argv[1])

    asyncio.run(main(listen_address, listen_port))
