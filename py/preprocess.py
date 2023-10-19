#!/usr/bin/env python3

import os
import sys
import logging
import multiprocessing as mp

from preprocess.tokenizer import Tokenizer
from preprocess.serdes import serialize

logging.basicConfig(level=logging.INFO)


def preprocess_file(tokenizer, input_file, output_dir):
    with open(input_file, "r") as f:
        text = f.read()

    tokens = tokenizer.encode(text, prepend_bos=True, append_eos=False)
    serialize(tokens, output_dir)


def main(tokenizer_path, input_dir, output_dir):
    tokenizer = Tokenizer(tokenizer_path)
    files = [
        os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".txt")
    ]

    if len(files) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    with mp.Pool() as pool:
        pool.starmap(
            preprocess_file,
            [(tokenizer, x, output_dir) for x in files],
            chunksize=1,
        )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: {} <tokenizer_path> <input_dir> <output_dir>".format(sys.argv[0]))
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    main(tokenizer_path, input_dir, output_dir)
