#!/usr/bin/env python3

import os
import sys
import logging
import json
import click

from common.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO)


def postprocess_file(tokenizer, input_file, output_file, discard_tokens=False):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            entry = json.loads(line)
            prompt_tokens = entry["prompt"]
            completion_tokens = entry["completion"]
            entry["prompt_text"] = "".join(tokenizer.decode(prompt_tokens))
            entry["completion_text"] = "".join(tokenizer.decode(completion_tokens))

            if discard_tokens:
                del entry["prompt"]
                del entry["completion"]

            print(json.dumps(entry), file=f_out, end="\n")


@click.command()
@click.option("--tokenizer-path", required=True, type=click.Path(exists=True))
@click.option(
    "--input-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="A JSONL file containing completed prompts.",
)
@click.option(
    "--output-file",
    required=True,
    type=click.Path(),
    help="A JSONL file containing the postprocessed prompts.",
)
@click.option(
    "--discard-tokens",
    is_flag=True,
    help="Discard the list of tokens from the output.",
)
def main(
    tokenizer_path,
    input_file,
    output_file,
    **kwargs,
):
    tokenizer = Tokenizer(tokenizer_path)
    postprocess_file(tokenizer, input_file, output_file, **kwargs)


if __name__ == "__main__":
    main()
