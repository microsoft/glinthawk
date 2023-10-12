import os
import sys
import asyncio

from itertools import count
from dataclasses import dataclass, field

from common.message import Message

@dataclass
class Worker:
    id: int = field(default_factory=count().__next__)
    reader: asyncio.StreamReader = None
    writer: asyncio.StreamWriter = None

workers = []
incoming_messages = asyncio.Queue()

async def handle_worker(reader, writer):
    global workers

    addr = writer.get_extra_info('peername')
    print(f'New connection from {addr!r}.')

    client = Worker(reader=reader, writer=writer)
    workers += [client]

    while True:
        try:
            message_header = await reader.readexactly(5)
        except asyncio.IncompleteReadError:
            message_header = None
            pass

        if not message_header:
            print(f'Connection from {addr!r} closed.')
            break

        payload_length, opcode = Message.parse_header(message_header)

        try:
            message_payload = await reader.readexactly(payload_length)
        except asyncio.IncompleteReadError:
            print(f'Connection from {addr!r} closed.')
            break

        message = Message(opcode=opcode, payload=message_payload)

        print(f'Received {message!r} from {addr!r}.')
        await incoming_messages.put(message)

async def main():
    server = await asyncio.start_server(handle_worker, '127.0.0.1', 8888)

    async with server:
        await server.serve_forever()

asyncio.run(main())
