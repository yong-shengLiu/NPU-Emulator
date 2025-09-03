
"""ara_sequencer emulator (functional abstraction)

This emulator models a simple sequencer that:
- fetches instructions from an instruction queue
- proceeds through pipeline stages: FETCH -> DECODE -> ISSUE -> EXECUTE -> WRITEBACK
- interacts with a Dispatcher-like interface via issue_request(request)
- supports backpressure: if dispatcher cannot accept, sequencer will stall in ISSUE
- supports vector "micro-ops" with exec_cycles
- provides tick()-based cycle simulation and tracing

Files produced by this module:
- sequencer_emulator.py : contains Sequencer, Instruction, minimal FU stub
- run_demo_sequencer.py : demo showing interaction with Dispatcher (optional integration)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Deque, Dict, Any, Tuple
from collections import deque
import enum

# Reuse simplified Request/Response concept used by dispatcher_emulator
@dataclass
class Instruction:
    id: int
    opname: str = "VOP"
    payload: Dict[str, Any] = field(default_factory=dict)
    exec_cycles: int = 1   # cycles needed in EX stage
    preferred_lane: int = -1

@dataclass
class IssuedUop:
    instr: Instruction
    remaining: int

class SeqState(enum.Enum):
    FETCH = 0
    DECODE = 1
    ISSUE = 2
    EXECUTE = 3
    WRITEBACK = 4
    IDLE = 5

class Sequencer:
    def __init__(self, issue_width:int=1, fifo_depth:int=8, trace:bool=True):
        self.issue_width = issue_width
        self.fifo_depth = fifo_depth
        self.instr_queue: Deque[Instruction] = deque(maxlen=fifo_depth)
        self.pipeline = {
            'fetch': None,
            'decode': None,
            'issue': [],       # list of Instructions being issued this cycle (up to issue_width)
            'execute': [],     # list of IssuedUop
            'writeback': [],   # completed entries waiting to retire
        }
        self.time = 0
        self.trace_enabled = trace
        # Simple stats
        self.completed = []
        self.stalls = 0
        self.idle_cycles = 0

    # ---------- public API ----------
    def push_instruction(self, instr: Instruction) -> bool:
        if len(self.instr_queue) >= self.fifo_depth:
            if self.trace_enabled:
                print(f"[t={self.time}] SEQ: instr_queue full, drop instr#{instr.id}")
            return False
        self.instr_queue.append(instr)
        if self.trace_enabled:
            print(f"[t={self.time}] SEQ: enq instr#{instr.id} op={instr.opname} cyc={instr.exec_cycles}")
        return True

    def can_accept(self) -> bool:
        return len(self.instr_queue) < self.fifo_depth

    def peek_next(self) -> Optional[Instruction]:
        return self.instr_queue[0] if self.instr_queue else None

    # This method represents trying to issue to a dispatcher.
    # The dispatcher interface expected: issue_request(req: Instruction) -> bool (accepted)
    def issue_request(self, dispatcher, instr: Instruction) -> bool:
        # dispatcher expected to have push_request() or enqueue; we try push_request(Request-like)
        try:
            # create a dispatcher-compatible Request object if present
            if hasattr(dispatcher, 'push_request'):
                # Create minimal request object expected by dispatcher_emulator
                req = type('Req', (), {})()
                req.id = instr.id
                req.exec_cycles = instr.exec_cycles
                req.payload = instr.payload
                req.preferred_lane = instr.preferred_lane
                dispatcher.push_request(req)
                return True
            elif callable(getattr(dispatcher, 'issue', None)):
                return dispatcher.issue(instr)
            else:
                # dispatcher doesn't match expected API
                return False
        except Exception as e:
            # If push_request raises (e.g., doesn't accept), return False to indicate backpressure
            if self.trace_enabled:
                print(f"[t={self.time}] SEQ: dispatcher rejected instr#{instr.id} ({e})")
            return False

    # ---------- internal pipeline control ----------
    def _fetch(self):
        if self.pipeline['fetch'] is None and self.instr_queue:
            self.pipeline['fetch'] = self.instr_queue.popleft()
            if self.trace_enabled:
                print(f"[t={self.time}] SEQ: FETCH instr#{self.pipeline['fetch'].id}")

    def _decode(self):
        if self.pipeline['fetch'] and self.pipeline['decode'] is None:
            self.pipeline['decode'] = self.pipeline['fetch']
            self.pipeline['fetch'] = None
            if self.trace_enabled:
                print(f"[t={self.time}] SEQ: DECODE instr#{self.pipeline['decode'].id}")

    def _issue(self, dispatcher):
        # Move decode -> issue, then try to push up to issue_width to dispatcher
        if self.pipeline['decode'] and len(self.pipeline['issue']) < self.issue_width:
            self.pipeline['issue'].append(self.pipeline['decode'])
            if self.trace_enabled:
                print(f"[t={self.time}] SEQ: READY TO ISSUE instr#{self.pipeline['decode'].id}")
            self.pipeline['decode'] = None

        # Try to issue each pending issue slot to dispatcher
        issued_now = []
        for instr in list(self.pipeline['issue']):
            # if dispatcher refuses (e.g., full), stall here
            ok = self.issue_request(dispatcher, instr)
            if ok:
                self.pipeline['issue'].remove(instr)
                # create an executing entry in execute stage
                self.pipeline['execute'].append(IssuedUop(instr=instr, remaining=instr.exec_cycles))
                if self.trace_enabled:
                    print(f"[t={self.time}] SEQ: ISSUED instr#{instr.id} to dispatcher -> EXECUTE")
                issued_now.append(instr.id)
            else:
                # can't issue due to backpressure
                if self.trace_enabled:
                    print(f"[t={self.time}] SEQ: STALL issuing instr#{instr.id} (dispatcher busy)")
                self.stalls += 1
                # do not remove from issue list; stall persists
        return issued_now

    def _execute(self):
        finished = []
        for uop in list(self.pipeline['execute']):
            uop.remaining -= 1
            if uop.remaining <= 0:
                self.pipeline['execute'].remove(uop)
                self.pipeline['writeback'].append(uop.instr)
                if self.trace_enabled:
                    print(f"[t={self.time}] SEQ: EXEC complete instr#{uop.instr.id}")
                finished.append(uop.instr.id)
        return finished

    def _writeback(self):
        # retire all writeback entries
        while self.pipeline['writeback']:
            instr = self.pipeline['writeback'].pop(0)
            self.completed.append(instr.id)
            if self.trace_enabled:
                print(f"[t={self.time}] SEQ: WRITEBACK instr#{instr.id} retired")

    def tick(self, dispatcher):
        """Advance sequencer by one cycle. `dispatcher` is used during issue stage to accept requests."""
        # Execute stages in reverse order (to reflect pipeline behavior)
        self._writeback()
        finished_exec = self._execute()
        issued = self._issue(dispatcher)
        self._decode()
        self._fetch()
        if not any([self.pipeline['fetch'], self.pipeline['decode'], self.pipeline['issue'], self.pipeline['execute'], self.pipeline['writeback']]) and not self.instr_queue:
            self.idle_cycles += 1
        self.time += 1
        return {
            "time": self.time,
            "issued": issued,
            "finished_exec": finished_exec,
            "stats": {"completed": list(self.completed), "stalls": self.stalls, "idle": self.idle_cycles}
        }

# Simple FU stub (not used if dispatcher drives execution)
class FunctionalUnit:
    def __init__(self, latency=1):
        self.latency = latency
        self.busy_cycles = 0

    def accept(self, instr: Instruction):
        if self.busy_cycles == 0:
            self.busy_cycles = self.latency
            return True
        return False

# ---------------- Demo integration with dispatcher_emulator if available ----------------
if __name__ == "__main__":
    # Provide a small demo that tries to import dispatcher_emulator and runs together
    try:
        from dispatcher_emulator import Dispatcher, Request, ArbPolicy
        use_dispatcher = True
    except Exception as e:
        use_dispatcher = False

    seq = Sequencer(issue_width=1, fifo_depth=8, trace=True)
    if use_dispatcher:
        disp = Dispatcher(num_lanes=2, queue_depth=2, init_credits=1, arb_policy=ArbPolicy.ROUND_ROBIN, trace=True)
    else:
        disp = None

    # Push instructions into sequencer
    for i in range(6):
        seq.push_instruction(Instruction(id=i, opname='VOP', exec_cycles=1 + (i % 3)))

    # Run for a bunch of cycles, trying to issue to dispatcher (if present)
    for cyc in range(20):
        print('\n---- CYCLE', cyc, '----')
        snap = seq.tick(disp)
        # optionally tick dispatcher too so it processes requests
        if disp:
            disp.tick()
        print('SEQ SNAP:', snap)
        if disp:
            print('DISP SNAP:', disp.snapshot())
