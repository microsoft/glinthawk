#!/usr/bin/env python3

import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required to run this program.")

import enum
import json
import asyncio
import logging
import itertools

from dataclasses import dataclass, field

from common.message import Message

logging.basicConfig(level=logging.INFO)


@dataclass
class Worker:
    class State(enum.Enum):
        Connected = enum.auto()
        Disconnected = enum.auto()

    id: int = field(default_factory=itertools.count().__next__)
    state: State = State.Connected
    address: str = None
    reader: asyncio.StreamReader = None
    writer: asyncio.StreamWriter = None


workers = []
incoming_messages = asyncio.Queue()


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
    while True:
        worker, message = await incoming_messages.get()
        logging.info(f'Received "{message!r}" from {worker.id}.')

        if message.opcode == Message.OpCode.Hey:
            worker.address = message.payload.decode()
            initialization_message = {
                "model_name": "XXX",
                "start_layer": worker.id,
                "end_layer": worker.id,
            }

            message = Message(
                Message.OpCode.InitializeWorker,
                json.dumps(initialization_message).encode(),
            )

            asyncio.create_task(send_message(worker, message))


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
