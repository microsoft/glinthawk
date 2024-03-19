import enum
import socket
import itertools
import asyncio

from typing import Tuple
from dataclasses import dataclass, field

from .base import Stage, Platform


@dataclass
class Worker:
    class State(enum.Enum):
        Connected = enum.auto()
        Disconnected = enum.auto()

    id: int = field(default_factory=itertools.count().__next__)
    state: State = State.Connected
    platform: Platform = None

    ip: bytes = None
    port: int = None

    reader: asyncio.StreamReader = None
    writer: asyncio.StreamWriter = None

    model_slice_start: Tuple[int, Stage] = (0, Stage.PreAttention)
    model_slice_end: Tuple[int, Stage] = (0, Stage.Classification)

    concurrency_size_pre: int = 16
    concurrency_size_att: int = 16
    concurrency_size_post: int = 16
    concurrency_size_cls: int = 16

    def __repr__(self):
        return f"Worker(id={self.id}, state={self.state}, platform={Platform.Name(self.platform)}, ip={socket.inet_ntoa(self.ip)}, port={self.port}, start_layer={self.model_slice_start[0]}, start_stage={Stage.Name(self.model_slice_start[1])}, end_layer={self.model_slice_end[0]}, end_stage={Stage.Name(self.model_slice_end[1])}, C1={self.concurrency_size_pre}, C2={self.concurrency_size_att}, C3={self.concurrency_size_post}, C4={self.concurrency_size_cls})"
