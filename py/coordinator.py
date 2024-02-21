#!/usr/bin/env python3
import sys

import settings
import asyncio

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required to run this program.")

import click
import logging

from rich.logging import RichHandler
from coordinator.coordinator import Coordinator

logging.basicConfig(level=logging.NOTSET, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])


@click.command()
@click.option("--n-layers", "-N", help="Total layers of the model", required=True, type=click.INT)
@click.option("--layers-per-worker", "-L", required=True, type=click.INT)
@click.option("--listen-address", required=True)
@click.option("--listen-port", required=True)
@click.option("--dummy-count", help="Number of dummy prompt batches", type=click.INT, default=0)
@click.option("--cpu_context_count", "-NCPU", type=click.INT, default=0)
@click.option("--gpu_context_count", "-NGPU", type=click.INT, default=0)
@click.option("--concurrency-size-pre", "-C1", type=click.INT, default=1)
@click.option("--concurrency-size-att", "-C2", type=click.INT, default=1)
@click.option("--concurrency-size-post", "-C3", type=click.INT, default=1)
@click.option("--concurrency-size-cls", "-C4", type=click.INT, default=1)
def main(listen_address, listen_port, **kwargs):
    coordinator = Coordinator(**kwargs)
    asyncio.run(coordinator.main(listen_address, listen_port))


if __name__ == "__main__":
    main()
