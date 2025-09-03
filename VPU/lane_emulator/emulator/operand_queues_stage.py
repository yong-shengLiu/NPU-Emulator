
from .operand_queue import OperandQueue
from typing import List, Optional

class OperandQueuesStage:
    """Holds two operand FIFOs (rs1, rs2)."""
    def __init__(self, depth: int = 4):
        self.q1 = OperandQueue(depth)
        self.q2 = OperandQueue(depth)

    def can_issue(self) -> bool:
        return len(self.q1) > 0 and len(self.q2) > 0

    def issue(self):
        return self.q1.pop(), self.q2.pop()
