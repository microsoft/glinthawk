import enum
import socket
import itertools
import asyncio

from typing import Tuple, Any
from dataclasses import dataclass, field

from .base import Stage, Platform, Kernel, Stage_Type, Platform_Type, Kernel_Type


@dataclass
class Worker:
    class State(enum.Enum):
        Connected = enum.auto()
        Disconnected = enum.auto()

    class Handshake(enum.Enum):
        Uninitiated = enum.auto()
        LayerAssigned = enum.auto()
        RouteAssigned = enum.auto()

    id: int = field(default_factory=itertools.count().__next__)
    state: State = State.Connected
    handshake_status: Handshake = Handshake.Uninitiated
    platform: Platform_Type = None
    kernel: Kernel_Type = None

    tier: int = -1
    rank: int = -1

    ip: bytes = None
    port: int = None

    reader: asyncio.StreamReader = None
    writer: asyncio.StreamWriter = None

    model_slice_start: Tuple[int, Stage_Type] = (0, Stage.PreAttention)
    model_slice_end: Tuple[int, Stage_Type] = (0, Stage.Classification)

    concurrency_size_pre: int = 16
    concurrency_size_att: int = 16
    concurrency_size_post: int = 16
    concurrency_size_cls: int = 16
    max_context_count: int = 0

    def is_first_parent(self) -> bool:
        return self.tier == 0 and self.rank == 0 and self.model_slice_start[0] == 0 and self.model_slice_start[1] == Stage.PreAttention

    def __repr__(self):
        return (f"Worker(id={self.id}, state={self.state}, platform={Platform.Name(self.platform)}, "
                f"kernel={Kernel.Name(self.kernel)}, "
                f"ip={socket.inet_ntoa(self.ip)}, port={self.port}, start_layer={self.model_slice_start[0]}, "
                f"start_stage={Stage.Name(self.model_slice_start[1])}, end_layer={self.model_slice_end[0]}, "
                f"end_stage={Stage.Name(self.model_slice_end[1])}, "
                f"tier={self.tier}, rank={self.rank}, "
                f"C1={self.concurrency_size_pre}, "
                f"C2={self.concurrency_size_att}, C3={self.concurrency_size_post}, C4={self.concurrency_size_cls})")
