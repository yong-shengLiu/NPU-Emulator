
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Dict, Any, Tuple
from collections import deque
import enum

@dataclass
class Request:
    id: int
    payload: Dict[str, Any] = field(default_factory=dict)
    preferred_lane: int = -1
    exec_cycles: int = 1

@dataclass
class Response:
    id: int
    lane: int
    cycles_spent: int
    payload: Dict[str, Any] = field(default_factory=dict)

class ArbPolicy(enum.Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOAD = "least_load"

class RoundRobinArbiter:
    def __init__(self, n: int):
        self.n = n
        self.ptr = 0
    def pick(self, ready: List[bool]) -> Optional[int]:
        for off in range(self.n):
            i = (self.ptr + off) % self.n
            if ready[i]:
                self.ptr = (i + 1) % self.n
                return i
        return None

from collections import deque
@dataclass
class Lane:
    id: int
    queue_depth: int
    credits: int
    executing: Optional[Tuple[Request, int]] = None
    q: Deque[Request] = field(default_factory=deque)
    def can_accept(self) -> bool:
        return len(self.q) < self.queue_depth
    def enqueue(self, req: Request) -> bool:
        if not self.can_accept():
            return False
        self.q.append(req)
        return True
    def tick(self) -> Optional[Response]:
        finished: Optional[Response] = None
        if self.executing is None and self.q and self.credits > 0:
            req = self.q.popleft()
            self.executing = (req, req.exec_cycles)
            self.credits -= 1
        if self.executing is not None:
            req, rem = self.executing
            rem -= 1
            if rem <= 0:
                finished = Response(id=req.id, lane=self.id, cycles_spent=req.exec_cycles, payload=req.payload)
                self.executing = None
                self.credits += 1
            else:
                self.executing = (req, rem)
        return finished

class Dispatcher:
    def __init__(self, num_lanes: int, queue_depth: int = 4, init_credits: int = 1, arb_policy: ArbPolicy = ArbPolicy.ROUND_ROBIN, trace: bool = True):
        self.num_lanes = num_lanes
        self.lanes = [Lane(i, queue_depth=queue_depth, credits=init_credits) for i in range(num_lanes)]
        self.incoming = deque()
        self.time = 0
        self.trace_enabled = trace
        self.rr = RoundRobinArbiter(num_lanes)
        self.arb_policy = arb_policy
        self.completed = []
        self.enq_drops = 0
    def push_request(self, req: Request) -> None:
        self.incoming.append(req)
        if self.trace_enabled:
            print(f"[t={self.time}] IN  req#{req.id} preferred={req.preferred_lane} cycles={req.exec_cycles}")
    def lane_set_credits(self, lane_id: int, credits: int) -> None:
        self.lanes[lane_id].credits = credits
    def lane_load(self, lane_id: int) -> int:
        l = self.lanes[lane_id]
        return len(l.q) + (1 if l.executing else 0)
    def pick_lane(self, req: Request) -> Optional[int]:
        if 0 <= req.preferred_lane < self.num_lanes and self.lanes[req.preferred_lane].can_accept():
            return req.preferred_lane
        if self.arb_policy == ArbPolicy.ROUND_ROBIN:
            ready = [ln.can_accept() for ln in self.lanes]
            return self.rr.pick(ready)
        else:
            candidates = [(self.lane_load(i), i) for i in range(self.num_lanes) if self.lanes[i].can_accept()]
            return min(candidates)[1] if candidates else None
    def tick(self):
        if self.incoming:
            req = self.incoming[0]
            lane_id = self.pick_lane(req)
            if lane_id is not None:
                self.incoming.popleft()
                ok = self.lanes[lane_id].enqueue(req)
                if self.trace_enabled:
                    if ok:
                        print(f"[t={self.time}] DIS -> lane{lane_id} req#{req.id}")
                    else:
                        print(f"[t={self.time}] DROP lane{lane_id} full req#{req.id}")
                        self.enq_drops += 1
        finished = []
        for ln in self.lanes:
            res = ln.tick()
            if res:
                finished.append(res)
                self.completed.append(res)
                if self.trace_enabled:
                    print(f"[t={self.time}] OUT <- lane{res.lane} req#{res.id} done")
        self.time += 1
        return finished
    def snapshot(self):
        return {
            "time": self.time,
            "incoming": [r.id for r in list(self.incoming)],
            "lanes": [
                {
                    "id": ln.id,
                    "q": [r.id for r in list(ln.q)],
                    "executing": (ln.executing[0].id, ln.executing[1]) if ln.executing else None,
                    "credits": ln.credits,
                }
                for ln in self.lanes
            ],
            "completed": [r.id for r in self.completed],
            "drops": self.enq_drops,
        }

def demo_run():
    disp = Dispatcher(num_lanes=4, queue_depth=2, init_credits=1, arb_policy=ArbPolicy.ROUND_ROBIN, trace=True)
    rid = 0
    for cyc in range(20):
        if cyc < 8:
            disp.push_request(Request(id=rid, exec_cycles=1 + (rid % 3), preferred_lane=-1))
            rid += 1
        disp.tick()
    return disp.snapshot()

if __name__ == "__main__":
    snap = demo_run()
    print(snap)
