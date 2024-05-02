#!/usr/bin/env python3
import sys

import rich.highlighter

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required to run this program.")

import click
import logging
import settings
import asyncio
import rich

from rich.logging import RichHandler
from coordinator.coordinator import Coordinator

logging.basicConfig(
    level=logging.NOTSET,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, highlighter=rich.highlighter.NullHighlighter())],
)

logo = rich.align.Align.center(
    rich.panel.Panel(
        rich.text.Text(
            """
░█▀▀░█░░░▀█▀░█▀█░▀█▀░█░█░█▀█░█░█░█░█
░█░█░█░░░░█░░█░█░░█░░█▀█░█▀█░█▄█░█▀▄
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀░▀░▀░▀░▀░▀
""",
            justify="center",
        ),
        expand=False,
        padding=(0, 2),
        style="",
    )
)

# TODO(sadjad): Add option to specify worker count instead of layers per worker
# TODO(sadjad): Streamline the concurrency sizes and context counts


@click.command()
@click.option("--n-layers", "-N", help="Total layers of the model", required=True, type=click.INT)
@click.option("--layers-per-worker", "-L", required=True, type=click.INT)
@click.option("--listen-address", required=True)
@click.option("--listen-port", required=True)
@click.option("--dummy-count", help="Number of dummy prompt batches", type=click.INT, default=0)
@click.option("--separate-cls", "-S", help="Run classification on a separate worker", is_flag=True, default=False)
@click.option("--cpu-context-count", "-NCPU", type=click.INT, default=0)
@click.option("--gpu-context-count", "-NGPU", type=click.INT, default=0)
@click.option("--concurrency-size-pre", "-C1", type=click.INT, default=1)
@click.option("--concurrency-size-att", "-C2", type=click.INT, default=1)
@click.option("--concurrency-size-post", "-C3", type=click.INT, default=1)
@click.option("--concurrency-size-cls", "-C4", type=click.INT, default=1)
@click.option("--prompt-dir", "-P", help="Directory for input files.")
@click.option("--output-dir", "-O", help="Directory for output files.")
def main(listen_address, listen_port, **kwargs):
    rich.print("\n", logo, "\n")
    logging.info(f"Starting coordinator with {kwargs}...", extra={"highlighter": rich.highlighter.JSONHighlighter()})

    coordinator = Coordinator(**kwargs)
    asyncio.run(coordinator.main(listen_address, listen_port))


if __name__ == "__main__":
    main()
