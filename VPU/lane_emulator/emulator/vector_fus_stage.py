
from .simd_alu import SIMD_ALU
from .types import MicroOp

class VectorFUSStage:
    """Functional Unit Selection (only SIMD_ALU in this simplified model)."""
    def __init__(self):
        self.alu = SIMD_ALU()

    def execute(self, uop: MicroOp, v1, v2):
        return self.alu.exec(uop, v1, v2)
