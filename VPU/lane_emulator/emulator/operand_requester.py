
from .vector_regfile import VectorRegFile

class OperandRequester:
    """Reads VRF and pushes operands to the operand queues."""
    def __init__(self, vrf: VectorRegFile, queues):
        self.vrf = vrf
        self.queues = queues

    def request(self, rs1: int, rs2: int):
        v1 = self.vrf.read(rs1)
        v2 = self.vrf.read(rs2)
        self.queues.q1.push(v1)
        self.queues.q2.push(v2)
