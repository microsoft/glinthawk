#!/usr/bin/env python3

import os
import sys
import json
import logging
import multiprocessing as mp

from common.tokenizer import Tokenizer
from common.serdes import serialize

from google.protobuf.json_format import MessageToDict
from protobuf import glinthawk_pb2 as pb

logging.basicConfig(level=logging.INFO)

PROMPTS_PER_FILE = 2 ** 16

def preprocess_slice(tokenizer, files, output_name):
    with open(output_name, "w") as fout:
        for f in files:
            with open(f, "r") as g:
                text = g.read()

            entry = pb.Prompt()
            entry.prompt.extend(tokenizer.encode(text, prepend_bos=True, append_eos=False))

            entry.user_data = f
            entry.temperature = 0
            entry.prompt_text = text

            fout.write(json.dumps(MessageToDict(entry)) + "\n")


def main(tokenizer_path, input_dir, output_dir):
    tokenizer = Tokenizer(tokenizer_path)
    files = [
        os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".txt")
    ]

    if len(files) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    f_idx = 0

    for i in range(0, len(files), PROMPTS_PER_FILE):
        files_slice = files[i:i + PROMPTS_PER_FILE]
        preprocess_slice(tokenizer, files_slice, os.path.join(output_dir, f"prompts_{f_idx}.jsonl"))
        f_idx += 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: {} <tokenizer_path> <input_dir> <output_dir>".format(sys.argv[0]))
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    main(tokenizer_path, input_dir, output_dir)
