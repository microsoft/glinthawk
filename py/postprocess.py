#!/usr/bin/env python3

import os
import sys
import struct
import logging
import multiprocessing as mp

from common.tokenizer import Tokenizer
from common.serdes import serialize

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.NOTSET, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

def postprocess_file(tokenizer, input_file, output_dir):
    output_file = os.path.join(output_dir, os.path.basename(input_file) + ".txt")

    if os.path.exists(output_file):
        return

    tokens = []

    with open(input_file, "rb") as f:
        # read first 4 bytes as token count (little-endian)
        token_count = struct.unpack("<I", f.read(4))[0]
        for i in range(token_count):
            token = struct.unpack("<I", f.read(4))[0]
            tokens += [token]

    tokens = tokenizer.decode(tokens)

    with open(output_file, "w") as f:
        f.write("".join(tokens))


def main(tokenizer_path, input_dir, output_dir):
    tokenizer = Tokenizer(tokenizer_path)
    files = [
        os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".ghc")
    ]

    if len(files) == 0:
        return

    logging.info("Postprocessing {} files...".format(len(files)))

    os.makedirs(output_dir, exist_ok=True)

    with mp.Pool() as pool:
        pool.starmap(
            postprocess_file,
            [(tokenizer, x, output_dir) for x in files],
            chunksize=32,
        )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: {} <tokenizer_path> <input_dir> <output_dir>".format(sys.argv[0]))
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    main(tokenizer_path, input_dir, output_dir)
