#!/usr/bin/env python3

"""
This script exports the Llama 2 weights in Glinthawk format.
Adopted from https://github.com/karpathy/llama2.c.
"""

import os
import sys
import struct
import json
import torch

from pathlib import Path


# from model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def export(p, state_dict, dest_dir, dtype_text="FP16"):
    dtype = torch.float16 if dtype_text == "FP16" else torch.float32

    os.makedirs(dest_dir, exist_ok=True)

    def serialize(f, key):
        print(f"writing {key}...")
        t = state_dict[key].contiguous().view(-1).type(dtype).numpy()
        f.write(memoryview(t))
        del state_dict[key]

    # first write out the header
    hidden_dim = state_dict["layers.0.feed_forward.w1.weight"].shape[0]
    p["vocab_size"] = 32000
    p["max_seq_len"] = 2048

    n_kv_heads = p.get("n_kv_heads") or p["n_heads"]
    header = struct.pack(
        "iiiiiii",
        p["dim"],
        hidden_dim,
        p["n_layers"],
        p["n_heads"],
        n_kv_heads,
        -p["vocab_size"],
        p["max_seq_len"],
    )
    # NOTE ABOVE: -ve vocab_size is indicating that the classifier weights are present
    # in the checkpoint and should be loaded.
    with open(os.path.join(dest_dir, "CONFIG"), "wb") as fout:
        fout.write(header)

    # next write out the embedding weights
    print("writing BASEWEIGHTS")

    with open(os.path.join(dest_dir, f"BASEWEIGHTS_{dtype_text}"), "wb") as fout:
        freqs_cos, freqs_sin = precompute_freqs_cis(
            p["dim"] // p["n_heads"], p["max_seq_len"] * 2
        )
        state_dict["freqs_cos"] = freqs_cos[: p["max_seq_len"]]
        state_dict["freqs_sin"] = freqs_sin[: p["max_seq_len"]]

        serialize(fout, "tok_embeddings.weight")
        serialize(fout, "norm.weight")
        serialize(fout, "freqs_cos")
        serialize(fout, "freqs_sin")
        serialize(fout, "output.weight")

    for i in range(p["n_layers"]):
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


def concat_weights(models):
    state_dict = {}
    for name in list(models[0]):
        tensors = [model[name] for model in models]
        if len(tensors) == 1 or len(tensors[0].shape) == 1:
            state_dict[name] = tensors[0]
            continue
        is_axis_1 = (
            name.startswith("tok_embeddings.")
            or name.endswith(".attention.wo.weight")
            or name.endswith(".feed_forward.w2.weight")
        )
        axis = 1 if is_axis_1 else 0
        state_dict[name] = torch.cat(tensors, dim=axis)
        for model in models:
            del model[name]
    return state_dict


def load_and_export(model_path, output_path, dtype_txt):
    params_path = os.path.join(model_path, "params.json")
    with open(params_path) as f:
        params = json.load(f)
        print(params)

    model_paths = sorted(list(Path(model_path).glob("consolidated.*.pth")))
    models = [torch.load(p, map_location="cpu") for p in model_paths]
    state_dict = concat_weights(models)
    del models
    export(params, state_dict, output_path, dtype_txt)


def main():
    if len(sys.argv) != 4:
        print("[Llama model folder path] [output folder path] [dtype=FP16|FP32]")
        exit()

    model_path = sys.argv[1]
    output_path = sys.argv[2]
    dtype_txt = sys.argv[3]
    load_and_export(model_path, output_path, dtype_txt)


if __name__ == "__main__":
    main()
