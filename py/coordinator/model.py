import socket

from protobuf import glinthawk_pb2 as protobuf

from .base import Stage, Platform
from .worker import Worker


class Model:
    def __init__(self, n_layers, layers_per_worker, separate_cls):
        assert n_layers % layers_per_worker == 0, "Number of layers must be divisible by layers per worker"

        self.n_layers = n_layers
        self.layers_per_worker = layers_per_worker
        self.separate_cls = separate_cls

        self._assigned_workers = 0

    def all_assigned(self) -> bool:
        return self._assigned_workers == self.n_layers // self.layers_per_worker + (1 if self.separate_cls else 0)

    def assign_slices(self, worker):
        num_layer_workers = self.n_layers // self.layers_per_worker

        if self.separate_cls and self._assigned_workers == num_layer_workers:
            # all layers have been assigned, we need to assign the last worker to the classification stage
            worker.model_slice_start = (self.n_layers - 1, Stage.Classification)
            worker.model_slice_end = (self.n_layers - 1, Stage.Classification)
        elif self._assigned_workers < num_layer_workers:
            first_layer = self._assigned_workers * self.layers_per_worker
            last_layer = (self._assigned_workers + 1) * self.layers_per_worker - 1
            # assign the worker to the next set of layers
            worker.model_slice_start = (first_layer, Stage.PreAttention)
            worker.model_slice_end = (last_layer, Stage.PostAttention)

            if last_layer == self.n_layers - 1:
                # We're responsible for classification
                worker.model_slice_end = (last_layer, Stage.Classification)
        else:
            # No more workers should be assigned
            return False

        self._assigned_workers += 1
        return True

    def route_message(self, workers):
        if not self.all_assigned():
            raise ValueError("Not all workers have been assigned layers")

        message = protobuf.SetRoute()

        for worker in workers:
            if worker.state != Worker.State.Connected:
                continue

            if worker.model_slice_start[1] == Stage.Classification:
                message.layer_to_address.append(
                    protobuf.SetRoute.LayerToAddress(
                        layer_num=self.n_layers - 1,
                        stage=Stage.Classification,
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
                            ip=socket.inet_ntoa(worker.ip),
                            port=worker.port,
                        )
                    )

        return message
