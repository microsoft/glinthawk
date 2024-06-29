#!/usr/bin/env python3
import sys

import rich.highlighter

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required to run this program.")

import click
import logging
import asyncio
import rich
import json

from rich.logging import RichHandler
from coordinator.coordinator import Coordinator

logging.basicConfig(
    level=logging.NOTSET,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, highlighter=rich.highlighter.NullHighlighter(), show_path=False)],
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


@click.command()
@click.option("--config-file", "-C", help="Config file for setting up nodes", required=True, type=click.STRING)
@click.option("--dummy-count", "-N", help="Number of dummy prompts", type=click.INT)
@click.option("--prompt-dir", "-P", help="Directory for input files.", type=click.STRING)
@click.option("--output-dir", "-O", help="Directory for output files.", required=True, type=click.STRING)
def main(config_file, **kwargs):
    with open(config_file, 'rb') as f:
        config = json.load(f)
    config.update(kwargs)
    assert 'dummy_count' in config or (
                'prompt_dir' in config and 'output_dir' in config), ("Neither dummy_count given nor prompt-dir/output-"
                                                                     "dir. Need at least one of these to start.")
    rich.print("\n", logo, "\n")
    logging.info(f"Starting coordinator with {config}...", extra={"highlighter": rich.highlighter.JSONHighlighter()})

    coordinator = Coordinator(**config)
    asyncio.run(coordinator.main(config['listen_address'], config['listen_port']))


if __name__ == "__main__":
    main()
