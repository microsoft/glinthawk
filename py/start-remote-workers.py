#!/usr/bin/env python3

import re
import asyncio
import logging
import signal
import hashlib

import click
import rich
import rich.logging

logging.basicConfig(
    level=logging.NOTSET,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        rich.logging.RichHandler(rich_tracebacks=True, highlighter=rich.highlighter.NullHighlighter(), show_path=False)
    ],
)


def get_worker_command(
    worker_address: str,
    worker_port: int,
    coordinator_address: str,
    coordinator_port: int,
    model_path: str,
    model_name: str,
    ssh_user: str,
    ssh_port: int = None,
    ssh_key: str = None,
):
    docker_command = [
        "docker",
        "run",
        "-t",
        "--rm",
        f"--name=glinthawk-worker-{hashlib.md5((worker_address + str(worker_port)).encode()).hexdigest()[:8]}",
        "--runtime=nvidia",
        "--gpus=all",
        "--network=host",
        "--no-healthcheck",
        "--read-only",
        "--ulimit=nofile=65535:65535",
        f"--mount=type=bind,src={model_path},dst=/app/model/,readonly",
        "$(test -S /tmp/telegraf.sock && echo '--mount=type=bind,src=/tmp/telegraf.sock,dst=/tmp/telegraf.sock' || echo '')",
        "glinthawk.azurecr.io/glinthawk-worker-cuda:latest",
        "/app/model/",
        f"{model_name}",
        f"{worker_address}",
        f"{worker_port}",
        f"{coordinator_address}",
        f"{coordinator_port}",
    ]

    ssh_command = [
        "ssh",
        "-t",
        "-t",
    ]

    if ssh_key:
        ssh_command += ["-i", ssh_key]

    if ssh_port:
        ssh_command += ["-p", f"{ssh_port}"]

    ssh_command += [
        f"{ssh_user}@{worker_address}",
        "/bin/bash",
        "-O",
        "huponexit",
        "-c",
        f'"{" ".join(docker_command)}"',
    ]

    return ssh_command


ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


async def run_command(command, addr, port):
    try:
        process = await asyncio.create_subprocess_exec(*command, stdout=None, stderr=None)
        await process.communicate()
        logging.warning(f"Process {addr}:{port} exited with code {process.returncode}.")

    except asyncio.CancelledError:
        if process and process.returncode is None:
            process.terminate()
            await process.wait()

        logging.warning(f"Process {addr}:{port} was cancelled.")


def shutdown():
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()


async def main(**kwargs):
    coord_addr = kwargs["coord_addr"]
    coord_port = kwargs["coord_port"]
    model_path = kwargs["model_path"]
    model_name = kwargs["model_name"]
    ssh_user = kwargs.get("ssh_user")
    ssh_key = kwargs.get("ssh_key")
    ssh_port = kwargs.get("ssh_port")

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, lambda: shutdown())

    logging.info("Starting workers...")

    tasks = []
    with open(kwargs["workers_file"], "r") as f:
        for line in f:
            line = line.strip().split(" ")
            assert len(line) >= 2

            worker_address = line[0]
            worker_port = int(line[1])
            tasks += [
                run_command(
                    get_worker_command(
                        worker_address,
                        worker_port,
                        coord_addr,
                        coord_port,
                        model_path,
                        model_name,
                        ssh_user,
                        ssh_port,
                        ssh_key,
                    ),
                    worker_address,
                    worker_port,
                )
            ]

    logging.info(f"{len(tasks)} worker(s) started.")
    logging.info("Press Ctrl+C to stop all workers.")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logging.warning("Cancelled all workers.")


@click.command()
@click.option("--coord-addr", "-C", help="Address of the coordinator", required=True)
@click.option("--coord-port", "-P", help="Port of the coordinator", required=True)
@click.option("--model-path", "-M", help="Path to the model directory on the remote", required=True)
@click.option("--model-name", "-N", help="Name of the model", required=True)
@click.option("--workers-file", "-W", help="File containing worker addresses", required=True)
@click.option("--ssh-user", "-u", help="SSH username to connect to hosts", required=False)
@click.option("--ssh-key", "-k", help="SSH private key file", required=False)
@click.option("--ssh-port", "-p", help="SSH port", default=22, required=False)
def start(**kwargs):
    asyncio.run(main(**kwargs))


if __name__ == "__main__":
    start()
