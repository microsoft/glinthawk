#!/usr/bin/env python3

# Adapted from meta-llama/llama3/llama/tokenizer.py

import sys
import struct
import tiktoken

from pathlib import Path
from tiktoken.load import load_tiktoken_bpe

pat_str = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    + r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| "
    + r"?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

num_reserved_special_tokens = 256

special_tokens = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_4|>",
    "<|eot_id|>",  # end of turn
] + [f"<|reserved_special_token_{i}|>" for i in range(5, num_reserved_special_tokens - 5)]


def main():
    if len(sys.argv) != 3:
        print("Usage: tokenizer2glint.py <input_path> <output_path>")
        exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    mergeable_ranks = load_tiktoken_bpe(input_path)
    num_base_tokens = len(mergeable_ranks)

    with open(output_path, "wb") as f:
        i = 0

        # write base tokens
        for token, rank in sorted(list(mergeable_ranks.items()), key=lambda x: x[1]):
            assert rank == i
            f.write(struct.pack("<I", len(token)))
            f.write(token)
            i += 1

        # write special tokens
        for token in special_tokens:
            f.write(struct.pack("<I", len(token)))
            f.write(token.encode("ascii"))
            i += 1

        assert i == num_base_tokens + num_reserved_special_tokens

if __name__ == "__main__":
    main()
