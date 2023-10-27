# Adapted from src/models/common/model.{hh,cc}

import struct
import hashlib
import datetime

from typing import Mapping, Tuple, Any
from dataclasses import dataclass, field

SIZE_HEADER = struct.calcsize("=32sIIIIf?BQI")
SIZE_MAP_ITEM = struct.calcsize("=I4sH")


@dataclass
class InferenceState:
    prompt_id: bytes = field(
        default_factory=lambda: hashlib.sha256(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f").encode()
        ).digest()
    )

    model_id: int = 0
    token: int = 1  # BOS
    token_pos: int = 0
    next_layer: int = 0
    prompt_length: int = 0
    temperature: float = 0.0
    finished: bool = False

    activations_dtype: int = 0
    activations_len: int = 0
    activations: bytes = b""

    layer_workers: Mapping[int, Any] = field(
        default_factory=dict
    )  # layer -> (worker_ip, worker_port)

    def load_from_payload(self, payload: bytes):
        self.prompt_id, self.model_id, self.token, self.token_pos, self.next_layer, self.temperature, self.finished, \
            self.activations_dtype, self.activations_len, len_layer_workers = struct.unpack("=32sIIIIf?BQI",
                                                                                            payload[:SIZE_HEADER])

        assert self.activations_len == 0, "Activations not yet supported"

        len_left = len(payload) - SIZE_HEADER

        while len_left > 0:
            k, rev_v0, v_1 = struct.unpack("=I4sH", payload[-len_left:-len_left + SIZE_MAP_ITEM])
            self.layer_workers[k] = [reversed(rev_v0), v_1]
            len_left -= SIZE_HEADER

    def serialize(self) -> bytes:
        if self.activations_len != 0:
            raise Exception("Activations not yet supported")

        if len(self.prompt_id) != 32:
            raise Exception("Invalid prompt id")

        message = struct.pack(
            "=32sIIIIIf?BQ",
            self.prompt_id,
            self.model_id,
            self.token,
            self.token_pos,
            self.next_layer,
            self.prompt_length,
            self.temperature,
            self.finished,
            self.activations_dtype,
            self.activations_len,
        )

        message += struct.pack("=I", len(self.layer_workers))

        for k, v in self.layer_workers.items():
            message += struct.pack("=I4sH", k, bytes(reversed(v[0])), v[1])

        return message
