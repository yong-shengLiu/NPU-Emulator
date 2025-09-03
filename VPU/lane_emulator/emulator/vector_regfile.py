
from typing import List, Dict
from .types import LaneConfig, DType, dtype_bits, clip

class VectorRegFile:
    """Simple vector register file: nregs x vlen array of integers."""
    def __init__(self, cfg: LaneConfig):
        self.cfg = cfg
        self.regs: Dict[int, List[int]] = {i: [0]*cfg.vlen for i in range(cfg.nregs)}

    def write(self, idx: int, data: List[int], dtype: DType):
        if len(data) != self.cfg.vlen:
            raise ValueError("vector length mismatch")
        self.regs[idx] = [clip(v, dtype) for v in data]

    def read(self, idx: int) -> List[int]:
        return list(self.regs[idx])

    def load_literal(self, idx: int, scalar: int, dtype: DType):
        self.regs[idx] = [scalar]*self.cfg.vlen
