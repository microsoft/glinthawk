import enum
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

    max_concurrency_size_pre: int = 16
    max_concurrency_size_att: int = 16
    max_concurrency_size_post: int = 16
    max_concurrency_size_cls: int = 16
