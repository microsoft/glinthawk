import numpy as np

cimport
numpy as np

import bisect
from typing import Dict
from typing import List, Tuple

import numpy as np


def two_tier_pipeline(t1_layers: List[int], delay_dict: Dict[str, float], in_flight: int) -> List[float]:
    return cy_two_tier_pipeline(t1_layers, in_flight, delay_dict['mid_layer_t1'], delay_dict['t1_to_t2_comm'],
                                delay_dict['t1_to_t2_rtt'], delay_dict['t2_comp'], delay_dict['t2_to_t1_comm'],
                                delay_dict['last_layer_t1'])


def single_tier_pipeline(t1_layers: List[int], delay_dict: Dict[str, float], in_flight: int) -> List[float]:
    return cy_single_tier_pipeline(t1_layers, in_flight, delay_dict['mid_layer_comp'], delay_dict['mid_layer_comm'],
                                   delay_dict['rtt'], delay_dict['last_layer_comp'])

cdef single_tier_pipeline(int[:] t1_layers, int in_flight, ):
    workers: List[Pipe] = []

    # A mapping from the item step to where it should go. This mapping helps the DES manager find out where it needs to
    # send an item to in the next step.
    to_id_map: Dict[int, int] = {}
    layers_so_far = 0

    delay_arr: List[float] = []

    for i in range(sum(t1_layers) - 1):
        delay_arr.append(delay_dict["mid_layer_comp"])
        delay_arr.append(delay_dict["mid_layer_comm"])
        delay_arr.append(delay_dict["rtt"] / 2)
    delay_arr.append(delay_dict["last_layer_comp"])
    delay_arr.append(delay_dict["last_layer_comm"])
    delay_arr.append(delay_dict["rtt"] / 2)

    delay_arr: np.ndarray = np.array(delay_arr)

    for k, n in enumerate(t1_layers):
        workers.append(Pipe(delay_arr))
        workers.append(Pipe(delay_arr))
        workers.append(DelayPipe(delay_arr))

        for i in range(layers_so_far, layers_so_far + n):
            to_id_map[i * 3 + 0] = k * 3 + 0
            to_id_map[i * 3 + 1] = k * 3 + 1
            to_id_map[i * 3 + 2] = k * 3 + 2
        layers_so_far += n

    assert len(to_id_map) == delay_arr.shape[0]

    next_events = np.array([w.min_event() for w in workers])

    for i in range(in_flight):
        workers[0].push(0, 0, i)
    next_events[0] = workers[0].min_event()

    # We want to find throughput. So we only care about how long it takes for one particular batch to make a full pass.
    # That time is the "Time Per Token" value, and during that time, "in_flight" batches completed.
    batch_0_complete = []
    while len(batch_0_complete) < 5:
        wid = np.argmin(next_events)
        step, end_time, batch_id = workers[wid].pop(next_events[wid].item())
        next_events[wid] = workers[wid].min_event()

        if step == len(to_id_map):
            step = 0

        next_wid = to_id_map[step]
        workers[next_wid].push(step, end_time, batch_id)
        next_events[next_wid] = workers[next_wid].min_event()

        if batch_id == 0 and step == 0:
            batch_0_complete.append(end_time)
            assert end_time >= (delay_dict["mid_layer_comp"] + delay_dict["mid_layer_comm"] + delay_dict[
                "rtt"] / 2) * layers_so_far * 0.99

    return batch_0_complete


# A serial "pipe" box, part of a pipeline. Serial means it can only carry out one "item" at a time. The box has a
# fixed delay. It is appropriate for things like GPU kernels, network link delay (only the part due to bandwidth), etc.
class Pipe:
    delay: np.ndarray
    next_start_time_: float
    queue: List[Tuple[int, float, int]]

    def __init__(self, delay: np.ndarray):
        self.next_start_time_ = 0
        self.delay = delay
        self.queue = []

    def min_event(self) -> float:
        if len(self.queue) == 0:
            return np.inf
        else:
            return max(self.queue[0][1], self.next_start_time_)

    def queue_pop(self, current_time: float):
        best_i = 0
        for i in range(1, len(self.queue)):
            if self.queue[i][1] > current_time:
                break
            if self.queue[i][0] < self.queue[best_i][0]:
                best_i = i
        return self.queue.pop(best_i)

    # push adds a new item to the queue.
    def push(self, step: int, entry_time: float, batch_id: int):
        # The queue is indexed by (pipeline step, queue entry time, batch index).
        # The pipeline step represents (layer, stage). We save "-step" as the first key since heapq emits the smallest
        # key.
        bisect.insort(self.queue, (-step, entry_time, batch_id), key=lambda x: x[1])

    def pop(self, current_time: float) -> Tuple[int, float, int]:
        # Pop an item and pass it through the pipeline
        step, entry_time, batch_id = self.queue_pop(current_time)
        step = -step
        assert current_time >= entry_time, (step, entry_time, batch_id, current_time, self.next_start_time_, self.queue)
        assert current_time >= self.next_start_time_, (
            step, entry_time, batch_id, current_time, self.next_start_time_, self.queue)
        job_end_time = current_time + self.delay[step].item()
        self.next_start_time_ = job_end_time
        return step + 1, job_end_time, batch_id


# A simple variation of pipe that is "parallel", meaning it can pass multiple items through at the same time. It is
# appropriate for things like network link latency.
class DelayPipe(Pipe):
    def __init__(self, delay):
        super(DelayPipe, self).__init__(delay)

    def pop(self, current_time: float) -> Tuple[int, float, int]:
        step, entry_time, batch_id = self.queue_pop(current_time)
        step = -step
        return step + 1, entry_time + self.delay[step].item(), batch_id


def single_tier_pipeline(t1_layers: List[int], delay_dict: Dict[str, float], in_flight: int) -> List[float]:
    workers: List[Pipe] = []

    # A mapping from the item step to where it should go. This mapping helps the DES manager find out where it needs to
    # send an item to in the next step.
    to_id_map: Dict[int, int] = {}
    layers_so_far = 0

    delay_arr: List[float] = []

    for i in range(sum(t1_layers) - 1):
        delay_arr.append(delay_dict["mid_layer_comp"])
        delay_arr.append(delay_dict["mid_layer_comm"])
        delay_arr.append(delay_dict["rtt"] / 2)
    delay_arr.append(delay_dict["last_layer_comp"])
    delay_arr.append(delay_dict["last_layer_comm"])
    delay_arr.append(delay_dict["rtt"] / 2)

    delay_arr: np.ndarray = np.array(delay_arr)

    for k, n in enumerate(t1_layers):
        workers.append(Pipe(delay_arr))
        workers.append(Pipe(delay_arr))
        workers.append(DelayPipe(delay_arr))

        for i in range(layers_so_far, layers_so_far + n):
            to_id_map[i * 3 + 0] = k * 3 + 0
            to_id_map[i * 3 + 1] = k * 3 + 1
            to_id_map[i * 3 + 2] = k * 3 + 2
        layers_so_far += n

    assert len(to_id_map) == delay_arr.shape[0]

    next_events = np.array([w.min_event() for w in workers])

    for i in range(in_flight):
        workers[0].push(0, 0, i)
    next_events[0] = workers[0].min_event()

    # We want to find throughput. So we only care about how long it takes for one particular batch to make a full pass.
    # That time is the "Time Per Token" value, and during that time, "in_flight" batches completed.
    batch_0_complete = []
    while len(batch_0_complete) < 5:
        wid = np.argmin(next_events)
        step, end_time, batch_id = workers[wid].pop(next_events[wid].item())
        next_events[wid] = workers[wid].min_event()

        if step == len(to_id_map):
            step = 0

        next_wid = to_id_map[step]
        workers[next_wid].push(step, end_time, batch_id)
        next_events[next_wid] = workers[next_wid].min_event()

        if batch_id == 0 and step == 0:
            batch_0_complete.append(end_time)
            assert end_time >= (delay_dict["mid_layer_comp"] + delay_dict["mid_layer_comm"] + delay_dict[
                "rtt"] / 2) * layers_so_far * 0.99

    return batch_0_complete


def two_tier_pipeline(t1_layers: List[int], delay_dict: Dict[str, float], in_flight: int) -> List[float]:
    workers: List[Pipe] = []

    # A mapping from the item step to where it should go. This mapping helps the DES manager find out where it needs to
    # send an item to in the next step.
    to_id_map: Dict[int, int] = {}
    layers_so_far = 0

    delay_arr: List[float] = []

    for i in range(sum(t1_layers) - 1):
        delay_arr.append(delay_dict["mid_layer_t1"])
        delay_arr.append(delay_dict["t1_to_t2_comm"])
        delay_arr.append(delay_dict["t1_to_t2_rtt"] / 2)
        delay_arr.append(delay_dict["t2_comp"])
        delay_arr.append(delay_dict["t2_to_t1_comm"])
        delay_arr.append(delay_dict["t1_to_t2_rtt"] / 2)

    delay_arr.append(delay_dict["last_layer_t1"])
    delay_arr.append(delay_dict["t1_to_t2_comm"])
    delay_arr.append(delay_dict["t1_to_t2_rtt"] / 2)
    delay_arr.append(delay_dict["t2_comp"])
    delay_arr.append(delay_dict["t2_to_t1_comm"])
    delay_arr.append(delay_dict["t1_to_t2_rtt"] / 2)

    delay_arr: np.ndarray = np.array(delay_arr)

    for k, n in enumerate(t1_layers):
        workers.append(Pipe(delay_arr))
        workers.append(Pipe(delay_arr))
        workers.append(DelayPipe(delay_arr))
        workers.append(Pipe(delay_arr))
        workers.append(Pipe(delay_arr))
        workers.append(DelayPipe(delay_arr))

        for i in range(layers_so_far, layers_so_far + n):
            to_id_map[i * 6 + 0] = k * 6 + 0
            to_id_map[i * 6 + 1] = k * 6 + 1
            to_id_map[i * 6 + 2] = k * 6 + 2
            to_id_map[i * 6 + 3] = k * 6 + 3
            to_id_map[i * 6 + 4] = k * 6 + 4
            to_id_map[i * 6 + 5] = k * 6 + 5
        layers_so_far += n

    assert len(to_id_map) == delay_arr.shape[0]

    next_events = np.array([w.min_event() for w in workers])

    for i in range(in_flight):
        workers[0].push(0, 0, i)
    next_events[0] = workers[0].min_event()

    # We want to find throughput. So we only care about how long it takes for one particular batch to make a full pass.
    # That time is the "Time Per Token" value, and during that time, "in_flight" batches completed.
    batch_0_complete = []
    while len(batch_0_complete) < 2:
        wid = np.argmin(next_events)
        step, end_time, batch_id = workers[wid].pop(next_events[wid].item())
        next_events[wid] = workers[wid].min_event()

        if step == len(to_id_map):
            step = 0

        next_wid = to_id_map[step]
        workers[next_wid].push(step, end_time, batch_id)
        next_events[next_wid] = workers[next_wid].min_event()

        if batch_id == 0 and step == 0:
            batch_0_complete.append(end_time)
            assert end_time >= (delay_dict["mid_layer_t1"] + delay_dict["t1_to_t2_comm"] + delay_dict["t1_to_t2_rtt"] +
                                delay_dict["t2_comp"] + delay_dict["t2_to_t1_comm"]) * layers_so_far * 0.99
    return batch_0_complete
