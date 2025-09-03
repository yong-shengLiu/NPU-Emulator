
from typing import List
from .types import LaneConfig, MicroOp
from .vector_regfile import VectorRegFile
from .operand_queues_stage import OperandQueuesStage
from .operand_requester import OperandRequester
from .vector_fus_stage import VectorFUSStage
from .lane_sequencer import LaneSequencer
from .power_gating import PowerGate

class Lane:
    """Top-level Lane emulator (behavioral, non-cycle-accurate).
    Pipeline (simplified):
      VRF -> OperandRequester -> OperandQueues -> FUS (SIMD_ALU) -> VRF
    Sequenced by LaneSequencer. Power gating can disable execution.
    """
    def __init__(self, cfg: LaneConfig = LaneConfig()):
        self.cfg = cfg
        self.vrf = VectorRegFile(cfg)
        self.queues = OperandQueuesStage(depth=8)
        self.req = OperandRequester(self.vrf, self.queues)
        self.fus = VectorFUSStage()
        self.pg = PowerGate()
        self.seq = None

    def load_program(self, program: List[MicroOp]):
        self.seq = LaneSequencer(program)

    def step(self) -> bool:
        if self.seq is None or not self.seq.has_next():
            return False
        uop = self.seq.next()
        # Request operands into queues (rs2 optional)
        v1 = self.vrf.read(uop.rs1)
        v2 = self.vrf.read(uop.rs2) if uop.rs2 is not None else None
        self.queues.q1.push(v1)
        self.queues.q2.push(v2 if v2 is not None else v1)  # if unary, mirror rs1
        # Issue if power is on
        if not self.pg.is_on():
            # Skip execution, keep destination unchanged
            return True
        a, b = self.queues.issue()
        out = self.fus.execute(uop, a, b)
        self.vrf.write(uop.rd, out, uop.dtype)
        return True

    def run(self):
        while self.step():
            pass
