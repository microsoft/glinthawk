#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import shlex
import signal
import sys
import socket
from typing import List, Dict, Any

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

model_name_to_dir = {
    "llama2-7b-chat": "llama-2-7b-chat-glint",
    "llama2-70b-chat": "llama-2-70b-chat-glint",
}


def add_args(config: Dict[str, Any], kwargs: Dict[str, str], add_workers: bool = True, add_ssh: bool = True,
             add_logs: bool = True) -> List[str]:
    additions = []
    if add_logs:
        additions += [
            "--log-stdout", f"{kwargs['worker_log_path']}/{config['config_name']}/",
            "--log-stderr", f"{kwargs['worker_log_path']}/{config['config_name']}/",
        ]
    if add_workers:
        for i in range(len(config['tiers'])):
            additions += ["--workers-file", f"{kwargs['config_path']}/remote.tier{i}.conf"]
    if add_ssh:
        additions += ["--ssh-user", kwargs['ssh_user']]
        if kwargs['ssh_key']:
            additions += ["--ssh-key", kwargs['ssh_key']]
        if kwargs['ssh_port']:
            additions += ["--ssh-port", str(kwargs['ssh_port'])]
    return additions


async def reachable(
        worker_address_files: List[str],
        ssh_user: str,
        ssh_port: int = None,
        ssh_key: str = None) -> List[int]:
    tasks = []
    for path in worker_address_files:
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip().split(" ")
                assert len(line) >= 2

                worker_address = line[0]

                tasks += [
                    run_command(
                        get_ssh_command(
                            "hostname",
                            worker_address,
                            ssh_user,
                            ssh_port,
                            ssh_key,
                        ),
                    )
                ]

    try:
        reached = await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logging.warning("Cancelled all workers.")
        reached = [-1 for _ in range(len(tasks))]

    print(reached)
    return reached


def get_ssh_command(
        command: str,
        worker_address: str,
        ssh_user: str,
        ssh_port: int = None,
        ssh_key: str = None,
) -> List[str]:
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


async def run_command(command) -> int:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )

        await process.communicate()
        logging.warning(f"Command exited with code {process.returncode}.")
        output = process.returncode
    except asyncio.CancelledError:
        logging.warning(f"Command was cancelled.")
        output = -1
    except asyncio.TimeoutError:
        logging.warning(f"Command timed out.")
        output = -2
    finally:
        if process and process.returncode is None:
            # Use SIGINT here so the scripts running underneath finish gracefully (e.g., remove docker containers)
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            await process.wait()

    return output


def shutdown():
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()


async def main(**kwargs):
    with open(f"{kwargs['config_path']}/coord.json", 'rb') as f:
        config = json.load(f)
    for i in range(len(config['tiers'])):
        assert os.path.exists(f"{kwargs['config_path']}/remote.tier{i}.conf")
        with open(f"{kwargs['config_path']}/remote.tier{i}.conf", "r") as f:
            assert len(f.readlines()) == config['tiers'][i]['ranks'] * config['n_slices']

    os.makedirs(f"{kwargs['worker_log_path']}/{config['config_name']}/", exist_ok=True)
    os.makedirs(f"{kwargs['completion_log_path']}/{config['config_name']}/", exist_ok=True)

    if kwargs['reboot']:
        command = [
            "python3",
            "run-command-remotes.py",
            "--command", "sudo reboot",
        ]
        command += add_args(config, kwargs)
        await run_command(command)

        kwargs_reachable = {
            'worker_address_files': [f"{kwargs['config_path']}/remote.tier{i}.conf" for i in
                                     range(len(config['tiers']))],
            'ssh_user': kwargs['ssh_user'],
            'ssh_key': kwargs['ssh_key'],
            'ssh_port': kwargs['ssh_port'],
        }
        reached_all = False
        while not reached_all:
            await asyncio.sleep(5)
            reached_all = all(code == 0 for code in await reachable(**kwargs_reachable))

    if kwargs['pull_image']:
        command = [
            "python3",
            "run-command-remotes.py",
            "--command", "docker pull glinthawk.azurecr.io/glinthawk-worker-cuda:latest",
        ]
        command += add_args(config, kwargs)
        await run_command(command)

    if kwargs['send_model']:
        # First make the folder for the models
        command = [
            "python3",
            "run-command-remotes.py",
            "--command", f"sudo mkdir {kwargs['dst_model_path']}; sudo chown glinthawk {kwargs['dst_model_path']}",
        ]
        command += add_args(config, kwargs)
        await run_command(command)
        # Then copy the files
        assert os.path.exists(f"{kwargs['src_model_path']}/{model_name_to_dir[config['model_name']]}/")
        command = [
            "python3",
            "send-to-remotes.py",
            "--src_path", f"{kwargs['src_model_path']}/{model_name_to_dir[config['model_name']]}/",
            "--dst_path", kwargs['dst_model_path']

        ]
        command += add_args(config, kwargs)
        await run_command(command)

    tasks = []

    tasks.append([
        "python3",
        "run.py",
        "-C", f"{kwargs['config_path']}/coord.json",
        "-N", "10240",
        "-O", kwargs['completion_log_path']
    ])
    for i in range(len(config['tiers'])):
        if config['tiers'][i]['platform'] == 'cuda':
            command = [
                "python3",
                "run-docker-remotes.py",
                "--workers-file", f"{kwargs['config_path']}/remote.tier{i}.conf",
                "--docker-options", "--runtime=nvidia",
                "--docker-options", "--gpus all",
                "--mount-ro", f"{kwargs['dst_model_path']}/{model_name_to_dir[config['model_name']]}/", "/app/model",
                "--mount-rw", "/tmp/telegraf.sock", "/tmp/telegraf.sock",
            ]
            command += add_args(config, kwargs, add_workers=False)
            command += [
                "glinthawk.azurecr.io/glinthawk-worker-cuda:latest",
                "/app/model/",
                f"{config['model_name']}",
                f"{config['tiers'][i]['kernel']}",
                f"{config['tiers'][i]['context']}",
                "__addr__",
                "__port__",
                socket.gethostbyname(socket.gethostname()),
                "3020",
            ]
        elif config['tiers'][i]['platform'] == 'amd64':
            command = [
                "python3",
                "run-docker-remotes.py",
                "--workers-file", f"{kwargs['config_path']}/remote.tier{i}.conf",
                "--docker-options", "--runtime=nvidia",
                "--mount-ro", f"{kwargs['dst_model_path']}/{model_name_to_dir[config['model_name']]}/", "/app/model",
                "--mount-rw", "/tmp/telegraf.sock", "/tmp/telegraf.sock",
                "--docker-options", "--entrypoint /app/worker-amd64-fp32"
            ]
            command += add_args(config, kwargs, add_workers=False)
            command += [
                "glinthawk.azurecr.io/glinthawk-worker-cuda:latest",
                "/app/model/",
                f"{config['model_name']}",
                f"{config['tiers'][i]['kernel']}",
                f"{config['tiers'][i]['context']}",
                "__addr__",
                "__port__",
                socket.gethostbyname(socket.gethostname()),
                "3020",
            ]
        else:
            print(f"Platform {config['tiers'][i]['platform']} not supported right now")
            return
        tasks.append(command)

    tasks = [run_command(t) for t in tasks]

    logging.info(f"Coordinator and {len(tasks) - 1} tier(s) started.")
    logging.info("Press Ctrl+C to stop all processes.")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logging.warning("Cancelled all processes.")


@click.command()
@click.option("--config-path", "-C",
              help="Directory containing config. Directory should have coord.json and remote.tierX.conf", required=True)
@click.option("--ssh-user", "-u", help="SSH username to connect to hosts.", default='glinthawk',
              required=False)
@click.option("--ssh-key", "-k", help="SSH private key file.", required=False)
@click.option("--ssh-port", "-p", help="SSH port.", default=22, required=False)
@click.option("--worker-log-path", help="Where to log worker output.",
              default='/home/glinthawk/worker_logs/', required=False)
@click.option("--completion-log-path", help="Where to log worker output.",
              default='/home/glinthawk/completions/', required=False)
@click.option("--src_model_path", required=False, default="/mnt/models/",
              help="Directory to models in master.")
@click.option("--dst_model_path", required=False, default="/mnt/models/",
              help="Directory to models in remotes.")
@click.option("--send-model", is_flag=True, help="Send the model.")
@click.option("--pull-image", is_flag=True, help="Pull the docker image.")
@click.option("--reboot", is_flag=True, help="Reboot the machines.")
def start(**kwargs):
    """This program runs an inference session with a given config."""
    asyncio.run(main(**kwargs))


if __name__ == "__main__":
    start()
