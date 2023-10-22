from typing import List, Tuple
from heapq import heapify, heappop, heappush


class Query:
    index: int
    n_pipe: int
    asgn_idx_s: List[int]
    completed: bool
    next_step: int
    base_time: float
    next_step_time: float

    def __init__(self, index: int, n_pipe: int, base_time: float):
        self.index = index
        self.n_pipe = n_pipe
        self.asgn_idx_s = []
        self.completed = False
        self.next_step = 0
        self.base_time = base_time
        self.next_step_time = base_time

    def is_empty(self) -> bool:
        return self.next_step == 0 and self.completed is False and len(self.asgn_idx_s) == 0

    def is_handled(self) -> bool:
        return self.is_empty() or self.completed

    def reschedule_base(self, new_base_time: float):
        assert self.is_empty()
        self.base_time = new_base_time
        self.next_step_time = new_base_time

    def schedule_next_step(self, mach):
        if self.completed:
            # If already assigned to the end, ensure it is assigned again to the same machines
            assert mach.index == self.asgn_idx_s[self.next_step]
        assert mach.layer == self.next_step

        start_time = max(self.next_step_time, mach.next_free_time)
        end_time = start_time + mach.lat

        self.asgn_idx_s.append(mach.index)
        if self.next_step == 0:
            self.base_time = start_time
        self.next_step += 1
        if self.next_step == self.n_pipe:
            self.completed = True
            self.next_step = 0
        self.next_step_time = end_time


class Machine:
    lat: float
    index: int
    layer: int
    asgn_query_s: List[Query]
    next_free_time: float

    def __init__(self, lat: float, index: int, layer: int):
        assert lat > 0
        self.lat = lat
        self.index = index
        self.layer = layer
        self.asgn_query_s = []
        self.next_free_time = 0
        self.current_job_queue = []

    def add_scheduled_query(self, query: Query):
        assert query.asgn_idx_s[self.layer] == self.index
        if query not in self.asgn_query_s:
            self.asgn_query_s.append(query)
        self.next_free_time = query.next_step_time


class Timeline:
    heap: List[Tuple[float, Query, int]]

    def __init__(self):
        self.heap = []
        heapify(self.heap)

    def next_event(self) -> Tuple[float, Query, int]:
        return heappop(self.heap)

    def __len__(self):
        return len(self.heap)

    def add_event(self, query: Query, check: int):
        heappush(self.heap, (query.next_step_time, query, check))

    def all_handled(self) -> bool:
        assert all(q[2] >= 0 and q[1].is_empty() or q[2] == -1 for q in self.heap)
        return all(q[1].is_handled() for q in self.heap)


def rate_to_slots(thr: List[List[float]]):
    n_pipe = len(thr)
    idx = 0
    mach_list: List[List[Machine]] = []
    for i, layer in enumerate(thr):
        mach_list.append([])
        for t in layer:
            mach_list[i].append(Machine(1/t, idx, i))
            idx += 1

    idx = 0
    timeline = Timeline()
    for i in range(len(mach_list[0])):
        new_query = Query(idx, n_pipe, 0)
        timeline.add_event(new_query, i)

    all_handled = False
    while not all_handled:
        wall_time, query, check = timeline.next_event()
        
        # #############################
        # #### schedule/reschedule jobs

        # #############################
        all_handled = timeline.all_handled()


def test():
    pass


if __name__ == '__main__':
    test()
