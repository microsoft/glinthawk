#!/usr/bin/env python3
import os.path
import struct


class Model:
    vocab_size: int
    dim: int
    kv_dim: int
    hidden_dim: int
    n_layers: int
    head_size: int
    n_heads: int
    n_kv_heads: int
    gqa_size: int
    seq_len: int
    wcls_present: bool
    data_width: int

    def __init__(self, dwidth: int):
        self.data_width = dwidth
        assert self.dim == self.head_size * self.n_heads
        assert self.n_kv_heads * self.gqa_size == self.n_heads

    def base_size(self, last_layer_cls: bool = True, first_layer_pre: bool = True, att: bool = True) -> int:
        size = 0
        if first_layer_pre:
            size += self.vocab_size * self.dim                          # Embedding table
        if last_layer_cls:
            size += self.dim                                            # RMS
            if self.wcls_present or not first_layer_pre:
                size += self.vocab_size * self.dim                      # Classification table
        if att:
            size += self.seq_len * self.head_size                       # RoPE
        return size * self.data_width

    def layer_size(self, pre: bool = True, post: bool = True) -> int:
        size = 0
        if pre:
            size += self.dim                                            # RMS
            size += self.dim * self.dim                                 # Query
            size += self.dim * self.kv_dim * 2                          # Key/Value
        if post:
            size += self.dim * self.dim                                 # WO
            size += self.dim                                            # RMS
            size += self.dim * self.hidden_dim * 3                      # FFN
        return size * self.data_width

    def kv_size(self, num_layers: int):
        size = self.kv_dim * 2 * self.seq_len * num_layers
        return size * self.data_width

    def bis_size(self, batch_size: int, after_stage: str):
        if after_stage == "pre":
            return batch_size * self.data_width * (self.dim * 2 + self.kv_dim * 2)
        elif after_stage == "att":
            return batch_size * self.data_width * (self.dim * 2)
        elif after_stage == "post":
            return batch_size * self.data_width * self.dim
        elif after_stage == "cls":
            return 0
        else:
            raise ValueError(f"No known stage={after_stage}")

    def bis_transit_ms(self, batch_size: int, after_stage: str, link_rtt_ms: float, link_cap_bps: int):
        return link_rtt_ms + self.bis_size(batch_size, after_stage) * 8 / link_cap_bps * 1000

    def write_config(self, config_path: str):
        assert os.path.exists(os.path.dirname(config_path))
        assert not os.path.exists(config_path)

        header = struct.pack(
            "iiiiiii",
            self.dim,
            self.hidden_dim,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            -self.vocab_size,
            self.seq_len,
        )
        # NOTE ABOVE: -ve vocab_size is indicating that the classifier weights are present
        # in the checkpoint and should be loaded.
        with open(config_path, "wb") as fout:
            fout.write(header)


class LLama3_405B(Model):
    vocab_size: int = 128256
    dim: int = 16384
    kv_dim: int = 1024
    hidden_dim: int = 53248
    n_layers: int = 126
    head_size: int = 128
    n_heads: int = 128
    n_kv_heads: int = 8
    gqa_size: int = 16
    seq_len: int = 2048
    wcls_present: bool = True

    def __init__(self, dwidth: int):
        super(LLama3_405B, self).__init__(dwidth)


class LLama3_70B(Model):
    vocab_size: int = 128256
    dim: int = 8192
    kv_dim: int = 1024
    hidden_dim: int = 28672
    n_layers: int = 80
    head_size: int = 128
    n_heads: int = 64
    n_kv_heads: int = 8
    gqa_size: int = 8
    seq_len: int = 2048
    wcls_present: bool = True

    def __init__(self, dwidth: int):
        super(LLama3_70B, self).__init__(dwidth)


class LLama3_8B(Model):
    vocab_size: int = 128256
    dim: int = 4096
    kv_dim: int = 1024
    hidden_dim: int = 14336
    n_layers: int = 32
    head_size: int = 128
    n_heads: int = 32
    n_kv_heads: int = 8
    gqa_size: int = 4
    seq_len: int = 2048
    wcls_present: bool = True

    def __init__(self, dwidth: int):
        super(LLama3_8B, self).__init__(dwidth)


class LLama2_70B(Model):
    vocab_size: int = 32000
    dim: int = 8192
    kv_dim: int = 1024
    hidden_dim: int = 28672
    n_layers: int = 80
    head_size: int = 128
    n_heads: int = 64
    n_kv_heads: int = 8
    gqa_size: int = 8
    seq_len: int = 2048
    wcls_present: bool = True

    def __init__(self, dwidth: int):
        super(LLama2_70B, self).__init__(dwidth)


class LLama2_13B(Model):
    vocab_size: int = 32000
    dim: int = 5120
    kv_dim: int = 5120
    hidden_dim: int = 13824
    n_layers: int = 40
    head_size: int = 128
    n_heads: int = 40
    n_kv_heads: int = 40
    gqa_size: int = 1
    seq_len: int = 2048
    wcls_present: bool = True

    def __init__(self, dwidth: int):
        super(LLama2_13B, self).__init__(dwidth)


class LLama2_7B(Model):
    vocab_size: int = 32000
    dim: int = 4096
    kv_dim: int = 4096
    hidden_dim: int = 11008
    n_layers: int = 32
    head_size: int = 128
    n_heads: int = 32
    n_kv_heads: int = 32
    gqa_size: int = 1
    seq_len: int = 2048
    wcls_present: bool = True

    def __init__(self, dwidth: int):
        super(LLama2_7B, self).__init__(dwidth)


class Stories_110M(Model):
    vocab_size: int = 32000
    dim: int = 768
    kv_dim: int = 768
    hidden_dim: int = 2048
    n_layers: int = 12
    head_size: int = 64
    n_heads: int = 12
    n_kv_heads: int = 12
    gqa_size: int = 1
    seq_len: int = 1024
    wcls_present: bool = False

    def __init__(self, dwidth: int):
        super(Stories_110M, self).__init__(dwidth)


def get_model(model_name: str, dwidth: int) -> Model:
    if model_name == "llama2-70b-chat":
        return LLama2_70B(dwidth)
    elif model_name == "llama2-13b-chat":
        return LLama2_13B(dwidth)
    elif model_name == "llama2-7b-chat":
        return LLama2_7B(dwidth)
    elif model_name == "llama3-8b-chat":
        return LLama3_8B(dwidth)
    elif model_name == "stories-110m":
        return Stories_110M(dwidth)
    else:
        raise ValueError(f"No such model={model_name}")
