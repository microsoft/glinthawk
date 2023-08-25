#!/usr/bin/env python3

"""
This script exports the Llama 2 weights in Glinthawk format.
Adopted from https://github.com/karpathy/llama2.c.
"""
import gc
import os
import pickle
import sys
import struct
import json
from collections import defaultdict
from typing import List, Dict, Tuple, BinaryIO, Union

import torch

from pathlib import Path

from tqdm import tqdm


# from model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def export(p: Dict[str, int], state_dict_map: Dict[str, List[Path,]], dest_dir: str, dtype_text: str = "FP16"):
    dtype = torch.float16 if dtype_text == "FP16" else torch.float32

    os.makedirs(dest_dir, exist_ok=True)

    def serialize(f: BinaryIO, key: str):
        print(f"writing {key}...")
        if isinstance(state_dict_map[key], list):
            t = load_tensor(key, state_dict_map[key])
            gc.collect()
        elif isinstance(state_dict_map[key], torch.Tensor):
            t = state_dict_map[key]
            del state_dict_map[key]
        else:
            raise TypeError("The type of this key is not string or a torch Tensor")
        t = t.contiguous().view(-1).type(dtype).numpy()
        f.write(memoryview(t).tobytes())

    # first write out the header
    p["vocab_size"] = 32000
    p["max_seq_len"] = 2048

    n_kv_heads = p.get("n_kv_heads") or p["n_heads"]
    header = struct.pack(
        "iiiiiii",
        p["dim"],
        p["hidden_dim"],
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
        state_dict_map["freqs_cos"] = freqs_cos[: p["max_seq_len"]]
        state_dict_map["freqs_sin"] = freqs_sin[: p["max_seq_len"]]

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


def load_tensor(key: str, path_list: List[Path]) -> torch.Tensor:
    tensors = []
    for piece_path in tqdm(path_list):
        with open(piece_path, "rb") as f:
            tensors.append(pickle.load(f))
        os.remove(piece_path)
    if len(tensors) == 1:
        ret_tensor = tensors[0]
    else:
        is_axis_1 = (
                key.startswith("tok_embeddings.")
                or key.endswith(".attention.wo.weight")
                or key.endswith(".feed_forward.w2.weight")
        )
        axis = 1 if is_axis_1 else 0
        ret_tensor = torch.cat(tensors, dim=axis)
    if ret_tensor.shape[0] == 1:
        print(key, ret_tensor)
    return ret_tensor


def form_dict(model_paths: List[Path]) -> Tuple[Dict[str, List[Path]], int]:
    state_dict_map = defaultdict(list)
    hidden_dim = 0
    for p in tqdm(model_paths):
        state_dict_piece = torch.load(p, map_location="cpu")
        for name in state_dict_piece:
            n = len(state_dict_map[name])
            with open(f"{p}_{name}_{n}", "wb") as f:
                pickle.dump(state_dict_piece[name], f)
            state_dict_map[name].append(f"{p}_{name}_{n}")
        if 'layers.0.feed_forward.w1.weight' in state_dict_piece:
            hidden_dim = state_dict_piece["layers.0.feed_forward.w1.weight"].shape[0]
        del state_dict_piece
        gc.collect()
    return state_dict_map, hidden_dim


def load_and_export(model_path: str, output_path: str, dtype_txt: str):
    params_path = os.path.join(model_path, "params.json")
    with open(params_path) as f:
        params = json.load(f)
    model_paths = sorted(list(Path(model_path).glob("consolidated.*.pth")))
    state_dict_map, params['hidden_dim'] = form_dict(model_paths)
    print(params)
    # del models
    export(params, state_dict_map, output_path, dtype_txt)


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
