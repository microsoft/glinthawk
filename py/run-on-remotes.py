#!/usr/bin/env python3

import os
import asyncio
import logging
import signal
import shlex
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


def get_ssh_command(
    command: str,
    worker_address: str,
    ssh_user: str,
    ssh_port: int = None,
    ssh_key: str = None,
):
    ssh_command = [
        "ssh",
    ]

    if ssh_key:
        ssh_command += ["-i", shlex.quote(ssh_key)]

    if ssh_port:
        ssh_command += ["-p", f"{ssh_port}"]

    ssh_command += [
        f"{ssh_user}@{worker_address}",
        "/bin/bash",
        "-O",
        "huponexit",
        "-c",
        f"{shlex.quote(command)}",
    ]

    return ssh_command


def get_worker_command(
    worker_address: str,
    worker_port: int,
    image_name: str,
    image_args: list,
    ssh_user: str,
    ssh_port: int = None,
    ssh_key: str = None,
    **kwargs,
):
    container_name = (
        "glinthawk-"
        + hashlib.sha256((image_name + " ".join(image_args) + worker_address + str(worker_port)).encode()).hexdigest()[
            :12
        ]
    )

    docker_command = [
        "docker",
        "run",
        "-t",
        "--rm",
        f"--name={container_name}",
        "--network=host",
        "--no-healthcheck",
        "--read-only",
        "--ulimit=nofile=65535:65535",
    ]

    for option in kwargs.get("docker_options", []):
        docker_command += shlex.split(option)

    for src, dst in kwargs.get("mount_ro", []):
        docker_command += [f"--mount=type=bind,src={shlex.quote(src)},dst={shlex.quote(dst)},readonly"]

    for src, dst in kwargs.get("mount_rw", []):
        docker_command += [f"--mount=type=bind,src={shlex.quote(src)},dst={shlex.quote(dst)}"]

    image_instance_args = list(image_args[:])

    for i, arg in enumerate(image_instance_args):
        if arg == "__addr__":
            image_instance_args[i] = worker_address
        elif arg == "__port__":
            image_instance_args[i] = str(worker_port)

    docker_command += [
        image_name,
        *image_instance_args,
    ]

    return (
        get_ssh_command(
            shlex.join(docker_command),
            worker_address,
            ssh_user,
            ssh_port,
            ssh_key,
        ),
        container_name,
    )


async def run_command(command, container_name, addr, port, log_stdout_dir=None, log_stderr_dir=None, **kwargs):
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL if log_stdout_dir is None else asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL if log_stderr_dir is None else asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        stdout, stderr = await process.communicate()
        logging.warning(f"Process {addr}:{port} exited with code {process.returncode}.")

        if log_stdout_dir and stdout:
            with open(os.path.join(log_stdout_dir, f"{addr}-{port}.stdout.log"), "wb") as f:
                f.write(stdout)

        if log_stderr_dir and stderr:
            with open(os.path.join(log_stderr_dir, f"{addr}-{port}.stderr.log"), "wb") as f:
                f.write(stderr)

    except asyncio.CancelledError:
        logging.warning(f"Process {addr}:{port} was cancelled.")
    finally:
        if process and process.returncode is None:
            os.killpg(os.getpgid(process.pid), signal.SIGHUP)
            await process.wait()

        logging.info(f"Stopping container {container_name} on {addr}:{port}...")
        process = await asyncio.create_subprocess_exec(
            *get_ssh_command(
                f"docker container kill --signal=SIGINT {container_name}",
                addr,
                kwargs["ssh_user"],
                kwargs.get("ssh_port"),
                kwargs.get("ssh_key"),
            ),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )

        await process.communicate()

        logging.info(f"Cleaned up {addr}:{port}.")


def shutdown():
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()


async def main(**kwargs):
    workers_file = kwargs["workers_file"]

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, lambda: shutdown())

    logging.info("Starting workers...")

    tasks = []
    with open(workers_file, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            assert len(line) >= 2

            worker_address = line[0]
            worker_port = int(line[1])

            command, container_name = get_worker_command(
                worker_address,
                worker_port,
                **kwargs,
            )

            tasks += [
                run_command(
                    command,
                    container_name,
                    worker_address,
                    worker_port,
                    log_stdout_dir=kwargs.get("log_stdout"),
                    log_stderr_dir=kwargs.get("log_stderr"),
                    **kwargs,
                )
            ]

    logging.info(f"{len(tasks)} worker(s) started.")
    logging.info("Press Ctrl+C to stop all workers.")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logging.warning("Cancelled all workers.")


@click.command()
@click.option(
    "--workers-file",
    "-W",
    help="File containing worker addresses. Each line should contain an address and a port separated by a space.",
    required=True,
)
@click.option("--ssh-user", "-u", help="SSH username to connect to hosts.", required=False)
@click.option("--ssh-key", "-k", help="SSH private key file.", required=False)
@click.option("--ssh-port", "-p", help="SSH port.", default=22, required=False)
@click.option(
    "--docker-options", "-X", help="Extra arguments to pass to the Docker command.", multiple=True, required=False
)
@click.option("--mount-ro", nargs=2, multiple=True, help="Mount a read-only volume.", required=False)
@click.option("--mount-rw", nargs=2, multiple=True, help="Mount a read-write volume.", required=False)
@click.option(
    "--log-stdout", type=click.Path(dir_okay=True, file_okay=False, exists=True), help="Log stdouts.", required=False
)
@click.option(
    "--log-stderr", type=click.Path(dir_okay=True, file_okay=False, exists=True), help="Log stderrs.", required=False
)
@click.argument("image-name", required=True)
@click.argument("image-args", nargs=-1)
def start(**kwargs):
    """This program runs a Docker container on a list of remote workers using SSH.

    You can use `__addr__` and `__port__` in the command arguments to replace with the worker's address and port.
    """
    asyncio.run(main(**kwargs))


if __name__ == "__main__":
    start()
