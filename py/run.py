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
    rich.text.Text(
        """
░█▀▀░█░░░▀█▀░█▀█░▀█▀░█░█░█▀█░█░█░█░█
░█░█░█░░░░█░░█░█░░█░░█▀█░█▀█░█▄█░█▀▄
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀░▀░▀░▀░▀░▀
""",
    ),
)


@click.command()
@click.option("--config-file", "-C", help="Config file for setting up nodes", required=True, type=click.STRING)
@click.option("--dummy-count", "-N", help="Number of dummy prompts", type=click.INT, default=0)
@click.option("--faux", is_flag=True, help="Do a microbenchmark with one slice.")
@click.option("--prompt-dir", "-P", help="Directory for input files.", type=click.STRING)
@click.option("--dataset", "-P", help="Path to a prompt input/output length dataset.",
              type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-dir", "-O", help="Directory for output files.", required=True, type=click.STRING)
def main(config_file, **kwargs):
    with open(config_file, 'rb') as f:
        config = json.load(f)
    config.update(kwargs)

    prompt_sources = 0
    prompt_sources += 1 if config['dummy_count'] > 0 else 0
    prompt_sources += 1 if config.get('dataset', None) is not None else 0
    prompt_sources += 1 if (
                config.get('prompt_dir', None) is not None and config.get('output_dir', None) is not None) else 0

    assert prompt_sources == 1, (
        f"Must be given exactly one of dummy_count, dataset or prompt-dir/output-dir, but given {prompt_sources}.")

    rich.print("\n", logo, "\n")
    logging.info(f"Starting coordinator with {config}...", extra={"highlighter": rich.highlighter.JSONHighlighter()})

    coordinator = Coordinator(**config)
    asyncio.run(coordinator.main(config['listen_address'], config['listen_port']))


if __name__ == "__main__":
    main()
