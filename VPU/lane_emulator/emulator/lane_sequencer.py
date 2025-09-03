
from typing import List
from .types import MicroOp

class LaneSequencer:
    """Takes a list of MicroOps and feeds them to the pipeline stages."""
    def __init__(self, program: List[MicroOp]):
        self.program = list(program)
        self.pc = 0

    def has_next(self) -> bool:
        return self.pc < len(self.program)

    def next(self) -> MicroOp:
        u = self.program[self.pc]
        self.pc += 1
        return u
