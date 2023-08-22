#!/usr/bin/env python3

"""
This script exports the Llama 2 bin files to Glinthawk format.
Adopted from https://github.com/karpathy/llama2.c.
"""

import os
import sys
import struct
from pprint import pprint
from typing import Dict, Tuple

import numpy as np
import torch


# from model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def export(config: Dict[str, int], state_dict: Dict[str, bytes], dest_dir: str, dtype_text: str = "FP16"):
    dtype = np.float16 if dtype_text == "FP16" else np.float32

    os.makedirs(dest_dir, exist_ok=True)

    def serialize(f, key: str):
        print(f"writing {key}...")
        t = np.frombuffer(state_dict[key], dtype=np.float16).astype(dtype)
        f.write(memoryview(t))
        del state_dict[key]

    def serialize_from_tensor(f, key: str):
        print(f"writing {key}...")
        t = state_dict[key].contiguous().view(-1).numpy().astype(dtype)
        f.write(memoryview(t))
        del state_dict[key]

    header = struct.pack(
        "iiiiiii",
        config["dim"],
        config["hidden_dim"],
        config["n_layers"],
        config["n_heads"],
        config["n_kv_heads"],
        config["vocab_size"],
        config["max_seq_len"],
    )
    with open(os.path.join(dest_dir, "CONFIG"), "wb") as fout:
        fout.write(header)

    # next write out the embedding weights
    print("writing BASEWEIGHTS")

    with open(os.path.join(dest_dir, f"BASEWEIGHTS_{dtype_text}"), "wb") as fout:
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config["dim"] // config["n_heads"], config["max_seq_len"] * 2
        )
        state_dict["freqs_cos"] = freqs_cos[: config["max_seq_len"]]
        state_dict["freqs_sin"] = freqs_sin[: config["max_seq_len"]]

        serialize(fout, "tok_embeddings.weight")
        serialize(fout, "norm.weight")
        serialize_from_tensor(fout, "freqs_cos")
        serialize_from_tensor(fout, "freqs_sin")
        if config["vocab_size"] < 0:
            serialize(fout, "output.weight")

    for i in range(config["n_layers"]):
        with open(os.path.join(dest_dir, f"LAYER{i}_{dtype_text}"), "wb") as fout:
            print(f"writing LAYER{i}")
            serialize(fout, f"layers.{i}.attention_norm.weight")
            serialize(fout, f"layers.{i}.attention.wq.weight")
            serialize(fout, f"layers.{i}.attention.wk.weight")
            serialize(fout, f"layers.{i}.attention.wv.weight")
            serialize(fout, f"layers.{i}.attention.wo.weight")
            serialize(fout, f"layers.{i}.ffn_norm.weight")
            serialize(fout, f"layers.{i}.feed_forward.w1.weight")
            serialize(fout, f"layers.{i}.feed_forward.w2.weight")
            serialize(fout, f"layers.{i}.feed_forward.w3.weight")


def load_bin(model_path: str) -> Tuple[Dict[str, int], Dict[str, bytes]]:
    with open(model_path, "rb") as fin:
        model = fin.read()
    state_dict = {}
    config = struct.unpack("iiiiiii", model[:28])
    config = {
        "dim": config[0],
        "hidden_dim": config[1],
        "n_layers": config[2],
        "n_heads": config[3],
        "n_kv_heads": config[4],
        "vocab_size": config[5],
        "max_seq_len": config[6],
    }
    head_size = config["dim"] // config["n_heads"]
    ptr = 28
    wd = 2
    state_dict["tok_embeddings.weight"] = model[ptr: ptr + abs(config["vocab_size"]) * config["dim"] * wd]
    ptr += abs(config["vocab_size"]) * config["dim"] * wd

    for i in range(config["n_layers"]):
        state_dict[f"layers.{i}.attention_norm.weight"] = model[ptr: ptr + config["dim"] * wd]
        ptr += config["dim"] * wd

    for i in range(config["n_layers"]):
        state_dict[f"layers.{i}.attention.wq.weight"] = model[ptr: ptr + config["dim"] * config["n_heads"] * head_size * wd]
        ptr += config["dim"] * config["n_heads"] * head_size * wd

    for i in range(config["n_layers"]):
        state_dict[f"layers.{i}.attention.wk.weight"] = model[ptr: ptr + config["dim"] * config["n_kv_heads"] * head_size * wd]
        ptr += config["dim"] * config["n_kv_heads"] * head_size * wd

    for i in range(config["n_layers"]):
        state_dict[f"layers.{i}.attention.wv.weight"] = model[ptr: ptr + config["dim"] * config["n_kv_heads"] * head_size * wd]
        ptr += config["dim"] * config["n_kv_heads"] * head_size * wd

    for i in range(config["n_layers"]):
        state_dict[f"layers.{i}.attention.wo.weight"] = model[ptr: ptr + config["dim"] * config["n_heads"] * head_size * wd]
        ptr += config["dim"] * config["n_heads"] * head_size * wd

    for i in range(config["n_layers"]):
        state_dict[f"layers.{i}.ffn_norm.weight"] = model[ptr: ptr + config["dim"] * wd]
        ptr += config["dim"] * wd

    for i in range(config["n_layers"]):
        state_dict[f"layers.{i}.feed_forward.w1.weight"] = model[ptr: ptr + config["dim"] * config["hidden_dim"] * wd]
        ptr += config["dim"] * config["hidden_dim"] * wd

    for i in range(config["n_layers"]):
        state_dict[f"layers.{i}.feed_forward.w2.weight"] = model[ptr: ptr + config["dim"] * config["hidden_dim"] * wd]
        ptr += config["dim"] * config["hidden_dim"] * wd

    for i in range(config["n_layers"]):
        state_dict[f"layers.{i}.feed_forward.w3.weight"] = model[ptr: ptr + config["dim"] * config["hidden_dim"] * wd]
        ptr += config["dim"] * config["hidden_dim"] * wd

    state_dict[f"norm.weight"] = model[ptr: ptr + config["dim"] * wd]
    ptr += config["dim"] * wd

    ptr += config["max_seq_len"] * head_size * wd

    if config["vocab_size"] < 0:
        state_dict["output.weight"] = model[ptr: ptr + abs(config["vocab_size"]) * config["dim"] * wd]
        ptr += abs(config["vocab_size"]) * config["dim"] * wd

    assert ptr == len(model)
    return config, state_dict


def load_and_export(model_path: str, output_path: str, dtype_txt: str):
    config, state_dict = load_bin(model_path)
    export(config, state_dict, output_path, dtype_txt)


def main():
    if len(sys.argv) != 4:
        print("[Llama model bin path] [output folder path] [dtype=FP16|FP32]")
        exit()

    model_path = sys.argv[1]
    output_path = sys.argv[2]
    dtype_txt = sys.argv[3]
    load_and_export(model_path, output_path, dtype_txt)


if __name__ == "__main__":
    main()
