import asyncio
import enum
import itertools
import socket
from dataclasses import dataclass, field
from typing import Tuple, List

from .base import Stage, Platform, Kernel, Stage_Type, Platform_Type, Kernel_Type, vector_index, stages_in_order


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

    slice_index: int = -1
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
        return self.tier == 0 and self.rank == 0 and self.model_slice_start[0] == 0 and self.model_slice_start[
            1] == Stage.PreAttention

    def create_slice_hosting_table(self, n_layers: int) -> List[bool]:
        vi_start = vector_index(*self.model_slice_start)
        vi_end = vector_index(*self.model_slice_end)

        slice_hosting_table = []
        for layer in range(n_layers):
            for stage in stages_in_order:
                vi = vector_index(layer, stage)
                if stage == Stage.Classification and layer != n_layers - 1:
                    slice_hosting_table.append(False)
                elif vi_start <= vi <= vi_end:
                    slice_hosting_table.append(True)
                else:
                    slice_hosting_table.append(False)
        return slice_hosting_table

    def create_node_hosting_table(self, n_layers: int) -> List[bool]:
        vi_start = vector_index(*self.model_slice_start)
        vi_end = vector_index(*self.model_slice_end)

        node_hosting_table = []
        for layer in range(n_layers):
            for stage, concur in zip(stages_in_order,
                                     [self.concurrency_size_pre, self.concurrency_size_att,
                                      self.concurrency_size_post, self.concurrency_size_cls]):
                vi = vector_index(layer, stage)
                if stage == Stage.Classification and layer != n_layers - 1:
                    node_hosting_table.append(False)
                elif vi_start <= vi <= vi_end and concur > 0:
                    node_hosting_table.append(True)
                else:
                    node_hosting_table.append(False)
        return node_hosting_table

    def __repr__(self):
        return (f"Worker(id={self.id}, state={self.state}, platform={Platform.Name(self.platform)}, "
                f"kernel={Kernel.Name(self.kernel)}, "
                f"ip={socket.inet_ntoa(self.ip)}, port={self.port}, start_layer={self.model_slice_start[0]}, "
                f"start_stage={Stage.Name(self.model_slice_start[1])}, end_layer={self.model_slice_end[0]}, "
                f"end_stage={Stage.Name(self.model_slice_end[1])}, "
                f"tier={self.tier}, rank={self.rank}, "
                f"C1={self.concurrency_size_pre}, "
                f"C2={self.concurrency_size_att}, C3={self.concurrency_size_post}, C4={self.concurrency_size_cls})")
