#!/usr/bin/env python3

"""
This script exports the Llama 2 weights in Glinthawk format.
Adapted from https://github.com/karpathy/llama2.c and https://github.com/meta-llama/llama3.
"""
import gc
import os
import pickle
import sys
import struct
import json
import tempfile
import torch

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, BinaryIO


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
        if isinstance(state_dict_map[key], list):
            t = load_tensor(key, state_dict_map[key])
        elif isinstance(state_dict_map[key], torch.Tensor):
            t = state_dict_map[key]
            del state_dict_map[key]
        else:
            raise TypeError("The type of this key is not string or a torch Tensor")

        t = t.contiguous().view(-1).type(dtype).numpy()
        f.write(memoryview(t).tobytes())

    header = struct.pack(
        "iiiiiii",
        p["dim"],
        p["hidden_dim"],
        p["n_layers"],
        p["n_heads"],
        p["n_kv_heads"],
        -p["vocab_size"],
        p["max_seq_len"],
    )

    # NOTE ABOVE: -ve vocab_size is indicating that the classifier weights are present
    # in the checkpoint and should be loaded.
    with open(os.path.join(dest_dir, "CONFIG"), "wb") as fout:
        fout.write(header)

    base_weight_names = ["tok_embeddings", "norm", "output", "freqs_cos", "freqs_sin"]

    weight_names = [
        "attention_norm",
        "attention.wq",
        "attention.wk",
        "attention.wv",
        "attention.wo",
        "ffn_norm",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    ]

    # Writing base weights
    with open(os.path.join(dest_dir, f"BASEWEIGHTS_{dtype_text}"), "wb") as fout:
        freqs_cos, freqs_sin = precompute_freqs_cis(p["dim"] // p["n_heads"], p["max_seq_len"] * 2, p["rope_theta"])
        state_dict_map["freqs_cos.weight"] = freqs_cos[: p["max_seq_len"]]
        state_dict_map["freqs_sin.weight"] = freqs_sin[: p["max_seq_len"]]

        for name in tqdm(base_weight_names, desc="Writing base weight matrices"):
            serialize(fout, f"{name}.weight")

    # Writing layers
    for i in tqdm(range(p["n_layers"]), desc="Writing layers"):
        with open(os.path.join(dest_dir, f"LAYER{i}_{dtype_text}"), "wb") as fout:
            for name in tqdm(weight_names, desc=f"Writing weights for layer {i}", leave=False):
                serialize(fout, f"layers.{i}.{name}.weight")


def load_tensor(key: str, path_list: List[Path]) -> torch.Tensor:
    tensors = []
    for piece_path in path_list:
        with open(piece_path, "rb") as f:
            tensors.append(pickle.load(f))
        os.remove(piece_path)

    if len(tensors) == 1 or len(tensors[0].shape) == 1:
        ret_tensor = tensors[0]
    else:
        is_axis_1 = (
            key.startswith("tok_embeddings.")
            or key.endswith(".attention.wo.weight")
            or key.endswith(".feed_forward.w2.weight")
        )
        axis = 1 if is_axis_1 else 0
        ret_tensor = torch.cat(tensors, dim=axis)

    return ret_tensor


def form_dict(params: Dict[str, int], model_paths: List[Path], temp_path) -> Dict[str, List[Path]]:
    state_dict_map = {}
    hidden_dim = 0

    for p in tqdm(model_paths, desc="Loading checkpoint pieces"):
        state_dict_piece = torch.load(p, map_location="cpu", mmap=True)

        if not hidden_dim and "layers.0.feed_forward.w1.weight" in state_dict_piece:
            hidden_dim = state_dict_piece["layers.0.feed_forward.w1.weight"].shape[0]

        for name in tqdm(state_dict_piece, desc="Dumping tensors", leave=False):
            out_name = os.path.join(temp_path, f"{os.path.basename(p)}_{name}_{len(state_dict_map.get(name, []))}")
            with open(out_name, "wb") as f:
                pickle.dump(state_dict_piece[name], f)

            state_dict_map[name] = state_dict_map.get(name, []) + [out_name]

    if hidden_dim:
        params["hidden_dim"] = hidden_dim

    return state_dict_map


def load_and_export(model_path: str, temp_path: str, output_path: str, dtype_txt: str):
    params_path = os.path.join(model_path, "params.json")

    with open(params_path) as f:
        params = json.load(f)
        params["n_kv_heads"] = params.get("n_kv_heads", params["n_heads"])
        params["max_seq_len"] = params.get("max_seq_len", 2048)

    model_paths = sorted(list(Path(model_path).glob("consolidated.*.pth")))

    print(f"Files to load: {model_paths}")
    state_dict_map = form_dict(params, model_paths, temp_path)

    print(f"{params=}")
    export(params, state_dict_map, output_path, dtype_txt)


def main():
    if len(sys.argv) != 4:
        print("[Llama model folder path] [output folder path] [dtype=FP16|FP32]")
        exit(1)

    model_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    dtype_txt = sys.argv[3]

    with tempfile.TemporaryDirectory(delete=True) as temp_dir:
        load_and_export(model_path, temp_dir, output_path, dtype_txt)


if __name__ == "__main__":
    main()
