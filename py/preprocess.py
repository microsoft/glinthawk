#!/usr/bin/env python3

import os
import sys
import json
import logging
import hashlib
import base58
import click

from common.tokenizer import Tokenizer

from google.protobuf.json_format import MessageToDict
from protobuf import glinthawk_pb2 as pb

logging.basicConfig(level=logging.INFO)

DEFAULTS_PROMPTS_PER_FILE = 2**16
B_INST, E_INST = ("[INST]", "[/INST]")
B_SYS, E_SYS = ("<<SYS>>\n", "\n<</SYS>>\n\n")


def create_chat_prompt(tokenizer, user_message, system_message, temperature):
    prompt_text = ""

    if system_message:
        prompt_text = f"{B_INST} {B_SYS} {system_message.strip()} {E_SYS} {user_message.strip()} {E_INST}"
    else:
        prompt_text = f"{B_INST} {user_message.strip()} {E_INST}"

    prompt_id = base58.b58encode(hashlib.sha256(prompt_text.encode()).digest()).decode()

    entry = pb.Prompt()
    entry.id = prompt_id
    entry.prompt.extend(tokenizer.encode(prompt_text, prepend_bos=True, append_eos=False))
    entry.temperature = int(255 * temperature)
    entry.prompt_text = prompt_text
    return entry


def preprocess_slice(tokenizer, files, output_name, system_message, temperature, max_seq_length):
    with open(output_name, "w") as fout:
        for f in files:
            with open(f, "r") as g:
                text = g.read()

            entry = create_chat_prompt(tokenizer, text, system_message, temperature)

            if len(entry.prompt) > max_seq_length:
                logging.warning(f"Skipping {f} due to excessive length ({len(entry.prompt)} > {max_seq_length}).")
                continue

            entry.user_data = f
            fout.write(json.dumps(MessageToDict(entry)) + "\n")


@click.command()
@click.option("--tokenizer-path", required=True, type=click.Path(exists=True))
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="A directory containing text files (.txt), one prompt per file.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for the preprocessed prompts.",
)
@click.option("--prompts-per-file", type=int, default=DEFAULTS_PROMPTS_PER_FILE)
@click.option("--temperature", type=float, default=0)
@click.option(
    "--system-message",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=None,
    help="A text file containing a system message to prepend to each prompt.",
)
@click.option("--max-seq-length", type=int, default=2048)
def main(
    tokenizer_path,
    input_dir,
    output_dir,
    prompts_per_file,
    temperature,
    system_message,
    max_seq_length,
):
    files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".txt")]
    if len(files) == 0:
        logging.error(f"No .txt files found in {input_dir}.")
        return

    system_message_str = None
    if system_message is not None:
        with open(system_message, "r") as f:
            system_message_str = f.read().strip()


    os.makedirs(output_dir, exist_ok=True)

    f_idx = 0
    tokenizer = Tokenizer(tokenizer_path)

    for i in range(0, len(files), prompts_per_file):
        files_slice = files[i : i + prompts_per_file]
        preprocess_slice(
            tokenizer,
            files_slice,
            os.path.join(output_dir, f"prompts_{f_idx}.jsonl"),
            system_message_str,
            temperature,
            max_seq_length,
        )

        f_idx += 1


if __name__ == "__main__":
    main()
