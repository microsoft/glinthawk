import os
import sys
import asyncio

from common.message import Message

async def main(coordinator_address, coordinator_port):
    message = Message(opcode=Message.OpCode.Hey, payload=b'Hello, world!')
    reader, writer = await asyncio.open_connection(coordinator_address, coordinator_port)
    writer.write(message.serialize())
    await writer.drain()

    writer.close()
    await writer.wait_closed()

def usage():
    print(f'Usage: {sys.argv[0]} <coordinator-address> <coordinator-port>')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        sys.exit(1)

    coordinator_address = sys.argv[1]
    coordinator_port = sys.argv[2]

    asyncio.run(main(coordinator_address=coordinator_address, coordinator_port=coordinator_port))
