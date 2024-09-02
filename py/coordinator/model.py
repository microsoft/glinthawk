import socket
from typing import List, Dict

from .base import Stage, Platform, Kernel
from .worker import Worker
from protobuf import glinthawk_pb2 as protobuf


class Model:
    def __init__(self, model_name: str, n_layers: int, n_slices: int, tier_config: List[Dict],
                 separate_cls_tiers: List[Dict]):
        assert n_layers % n_slices == 0, "Number of layers must be divisible by number of slices"

        self.model_name = model_name
        self.n_layers = n_layers
        self.n_slices = n_slices
        self.layers_per_worker = n_layers // n_slices
        self.tier_config = tier_config
        self.n_tiers = len(self.tier_config)

        self.tier_concurrency_s = [
            protobuf.InitializeWorker.TierConcurrency(
                ranks=self.tier_config[i]['ranks'],
                concurrency_pre_att_size=self.tier_config[i]['concurrency_size_pre'],
                concurrency_att_size=self.tier_config[i]['concurrency_size_att'],
                concurrency_post_att_size=self.tier_config[i]['concurrency_size_post'],
                concurrency_cls_size=self.tier_config[i]['concurrency_size_cls'],
                max_context_count=self.tier_config[i]['max_context_count'],
            )
            for i in range(self.n_tiers)
        ]

        for i_tier in range(self.n_tiers):
            platform_str = self.tier_config[i_tier]['platform']
            self.tier_config[i_tier]['platform_str'] = platform_str
            if platform_str == 'amd64':
                self.tier_config[i_tier]['platform'] = Platform.AMD64
            elif platform_str == 'cuda':
                self.tier_config[i_tier]['platform'] = Platform.CUDA
            else:
                raise ValueError(f'Unknown platform "{platform_str}" in config')

            kernel_str = self.tier_config[i_tier]['kernel']
            if kernel_str == 'batched':
                self.tier_config[i_tier]['kernel'] = Kernel.Batched
            elif kernel_str == 'hybrid':
                self.tier_config[i_tier]['kernel'] = Kernel.Hybrid
            elif kernel_str == 'simple_hybrid':
                self.tier_config[i_tier]['kernel'] = Kernel.SimpleHybrid
            elif kernel_str == 'simple_piped':
                self.tier_config[i_tier]['kernel'] = Kernel.SimplePiped
            else:
                raise ValueError(f'Unknown kernel "{kernel_str}" in config')

        self.separate_cls_tiers = separate_cls_tiers
        self.separate_cls = len(self.separate_cls_tiers) > 0

        assert self.separate_cls is False, "We don't support separate classification worker yet!"

        assert sum(self.tier_config[i]['concurrency_size_cls']*self.tier_config[i]['ranks'] for i in range(self.n_tiers)) == sum(
            self.tier_config[i]['concurrency_size_pre']*self.tier_config[i]['ranks'] for i in range(self.n_tiers))

        assert sum(self.tier_config[i]['concurrency_size_cls']*self.tier_config[i]['ranks'] for i in range(self.n_tiers)) == sum(
            self.tier_config[i]['concurrency_size_att']*self.tier_config[i]['ranks'] for i in range(self.n_tiers))

        assert sum(self.tier_config[i]['concurrency_size_cls']*self.tier_config[i]['ranks'] for i in range(self.n_tiers)) == sum(
            self.tier_config[i]['concurrency_size_post']*self.tier_config[i]['ranks'] for i in range(self.n_tiers))

        # For each tier, what is the next (slice, rank) to place the worker at.
        self._next_worker_loc = [{"slice": 0, "rank": 0} for _ in range(self.n_tiers)]

    def all_assigned(self) -> bool:
        for i_tier in range(self.n_tiers):
            if self._next_worker_loc[i_tier]["slice"] < self.n_slices:
                return False
        return True

    def _find_suitable_tier(self, worker: Worker) -> int:
        suitable_tier_index = -1
        for i in range(self.n_tiers):
            if (worker.platform == self.tier_config[i]['platform'] and
                    worker.kernel == self.tier_config[i]['kernel'] and
                    self._next_worker_loc[i]['slice'] < self.n_slices):
                suitable_tier_index = i
        return suitable_tier_index

    def get_tier_concurrencies_message(self):
        return self.tier_concurrency_s

    def assign_slices(self, worker) -> bool:
        # find applicable tiers
        i_tier = self._find_suitable_tier(worker)
        if i_tier == -1:
            return False

        first_layer = self._next_worker_loc[i_tier]['slice'] * self.layers_per_worker
        last_layer = (self._next_worker_loc[i_tier]['slice'] + 1) * self.layers_per_worker - 1
        # assign the worker to the correct slice, tier and rank
        worker.model_slice_start = (first_layer, Stage.PreAttention)
        worker.model_slice_end = (last_layer, Stage.PostAttention)
        if last_layer == self.n_layers - 1:
            # We're responsible for classification
            worker.model_slice_end = (last_layer, Stage.Classification)
        worker.slice_index = self._next_worker_loc[i_tier]['slice']
        worker.tier = i_tier
        worker.rank = self._next_worker_loc[i_tier]['rank']

        worker.concurrency_size_pre = self.tier_config[worker.tier]['concurrency_size_pre']
        worker.concurrency_size_att = self.tier_config[worker.tier]['concurrency_size_att']
        worker.concurrency_size_post = self.tier_config[worker.tier]['concurrency_size_post']
        worker.concurrency_size_cls = self.tier_config[worker.tier]['concurrency_size_cls']
        worker.max_context_count = self.tier_config[worker.tier]['max_context_count']

        # advance next_worker_loc
        self._next_worker_loc[i_tier]['rank'] += 1
        if self._next_worker_loc[i_tier]['rank'] == self.tier_config[i_tier]['ranks']:
            self._next_worker_loc[i_tier]['rank'] = 0
            self._next_worker_loc[i_tier]['slice'] += 1
        return True

    def route_message(self, workers):
        if not self.all_assigned():
            raise ValueError("Not all workers have been assigned layers")

        message = protobuf.SetRoute()

        for worker in workers:
            if worker.state != Worker.State.Connected:
                raise RuntimeError('A worker has disconnected. This possibly corrupts the path!')

            if worker.model_slice_start[1] == Stage.Classification:
                message.layer_to_address.append(
                    protobuf.SetRoute.LayerToAddress(
                        layer_num=self.n_layers - 1,
                        stage=Stage.Classification,
                        tier=worker.tier,
                        rank=worker.rank,
                        ip=socket.inet_ntoa(worker.ip),
                        port=worker.port,
                    )
                )
                continue

            start_layer, start_stage = worker.model_slice_start
            end_layer, _ = worker.model_slice_end

            for layer in range(start_layer, end_layer + 1):
                for stage in [Stage.PreAttention, Stage.Attention, Stage.PostAttention]:
                    message.layer_to_address.append(
                        protobuf.SetRoute.LayerToAddress(
                            layer_num=layer,
                            stage=stage,
                            tier=worker.tier,
                            rank=worker.rank,
                            ip=socket.inet_ntoa(worker.ip),
                            port=worker.port,
                        )
                    )

        return message
