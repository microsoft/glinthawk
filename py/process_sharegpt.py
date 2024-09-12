#!/usr/bin/env python3

import json
import logging
import os
from typing import List, Tuple

import click
import numpy as np
from tqdm.auto import tqdm

from common.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO)

B_INST, E_INST = ("[INST]", "[/INST]")


def preprocess_json(tokenizer: Tokenizer, file: str) -> Tuple[List[int], List[int]]:
    with open(file, "r") as f:
        prompts = json.load(f)

    input_lengths = []
    output_lengths = []

    for chat in tqdm(prompts):
        if len(chat['conversations']) < 2:
            continue
        if chat['conversations'][0]['from'] != 'human':
            continue
        if chat['conversations'][1]['from'] != 'gpt':
            continue

        human_msg = chat['conversations'][0]['value']
        prompt_text = f"{B_INST} {human_msg.strip()} {E_INST}"
        new_in_len = len(tokenizer.encode(prompt_text, prepend_bos=True, append_eos=False))
        input_lengths.append(new_in_len)

        gpt_msg = chat['conversations'][1]['value']
        result_text = f"{gpt_msg.strip()}"
        new_out_len = len(tokenizer.encode(result_text, prepend_bos=False, append_eos=True))
        output_lengths.append(new_out_len)

    return input_lengths, output_lengths


@click.command()
@click.option("--tokenizer-path", required=True, type=click.Path(exists=True))
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="A directory containing the ShareGPT json files (.json).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for the preprocessed prompt distribution.",
)
def main(
        tokenizer_path,
        input_dir,
        output_dir,
):
    files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".json")]
    if len(files) == 0:
        logging.error(f"No .json files found in {input_dir}.")
        return

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    input_len = []
    output_len = []

    for i in range(len(files)):
        input_len_this_file, output_len_this_file = preprocess_json(
            tokenizer,
            files[i],
        )

        input_len.extend(input_len_this_file)
        output_len.extend(output_len_this_file)

    len_arr = np.c_[input_len, output_len]
    np.save(f"{output_dir}/full_len.npy", len_arr)


if __name__ == "__main__":
    main()
