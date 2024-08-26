#!/usr/bin/env python3

import asyncio
import logging
import os
import shlex
import signal
import time

import click
import rich.logging

logging.basicConfig(
    level=logging.NOTSET,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        rich.logging.RichHandler(rich_tracebacks=True, highlighter=rich.highlighter.NullHighlighter(), show_path=False)
    ],
)


def get_scp_command(
        src_path: str,
        dest_path: str,
        worker_address: str,
        ssh_user: str,
        ssh_port: int = None,
        ssh_key: str = None,
):
    scp_command = [
        "scp",
        "-r",
    ]

    if ssh_key:
        scp_command += ["-i", shlex.quote(ssh_key)]

    if ssh_port:
        scp_command += ["-P", f"{ssh_port}"]

    scp_command += [
        f"{src_path}",
        f"{ssh_user}@{worker_address}:{dest_path}",
    ]

    return scp_command


async def run_command(command, addr, port, log_stdout_dir=None, log_stderr_dir=None):
    try:
        f_out = open(os.path.join(log_stdout_dir, f"{addr}-{port}.stdout.log"),
                     "wb") if log_stdout_dir else asyncio.subprocess.DEVNULL
        f_err = open(os.path.join(log_stderr_dir, f"{addr}-{port}.stderr.log"),
                     "wb") if log_stderr_dir else asyncio.subprocess.DEVNULL

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=f_out,
            stderr=f_err,
            start_new_session=True,
        )

        await process.communicate()
        logging.warning(f"Process {addr}:{port} exited with code {process.returncode}.")

    except asyncio.CancelledError:
        logging.warning(f"Process {addr}:{port} was cancelled.")
    finally:
        if process and process.returncode is None:
            os.killpg(os.getpgid(process.pid), signal.SIGHUP)
            await process.wait()

        if log_stdout_dir:
            f_out.close()

        if log_stderr_dir:
            f_err.close()

        logging.info(f"Cleaned up {addr}:{port}.")


def shutdown():
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()


async def main(**kwargs):
    workers_file = kwargs["workers_file"]
    workers = []
    for path in workers_file:
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip().split(" ")
                assert len(line) >= 2

                worker_address = line[0]
                worker_port = int(line[1])

                workers.append((worker_address, worker_port))

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, lambda: shutdown())

    time_string = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())

    log_std_out = kwargs.get("log_stdout")
    if log_std_out:
        log_std_out = os.path.join(log_std_out, time_string)
        os.makedirs(log_std_out, exist_ok=True)

    log_std_err = kwargs.get("log_stderr")
    if log_std_err:
        log_std_err = os.path.join(log_std_err, time_string)
        os.makedirs(log_std_err, exist_ok=True)

    logging.info("Starting workers...")
    tasks = []
    for waddr, wip in workers:

        tasks += [
            run_command(
                get_scp_command(
                    kwargs.get("src_path"),
                    kwargs.get("dst_path"),
                    waddr,
                    kwargs.get("ssh_user"),
                    kwargs.get("ssh_port"),
                    kwargs.get("ssh_key"),
                ),
                waddr,
                wip,
                log_stdout_dir=log_std_out,
                log_stderr_dir=log_std_err,
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
    multiple=True,
    required=True,
)
@click.option("--ssh-user", "-u", help="SSH username to connect to hosts.", required=False)
@click.option("--ssh-key", "-k", help="SSH private key file.", required=False)
@click.option("--ssh-port", "-p", help="SSH port.", default=22, required=False)
@click.option("--src_path", "-X", help="Source path to send to remote.", required=True)
@click.option("--dst_path", "-X", help="Destination to put source data in.", required=True)
@click.option(
    "--log-stdout", type=click.Path(dir_okay=True, file_okay=False, exists=True), help="Log stdouts.", required=False
)
@click.option(
    "--log-stderr", type=click.Path(dir_okay=True, file_okay=False, exists=True), help="Log stderrs.", required=False
)
def start(**kwargs):
    """This program runs custom commands on a list of remote workers using SSH.

    You can use `__addr__` and `__port__` in the command arguments to replace with the worker's address and port.
    """
    asyncio.run(main(**kwargs))


if __name__ == "__main__":
    start()
