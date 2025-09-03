
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from .memory import Memory
from .vldu import LoadUnit
from .vstu import StoreUnit
from .types import PeReq, PeResp, LSUException

@dataclass
class VLSU:
    """High-level emulator of Ara's Vector Load/Store Unit (VLSU).
    It composes a Memory, a VRF (vector register file), and separate Load/Store units.
    This model is NOT cycle-accurate. It performs one request at a time and
    returns a PeResp indicating success or exception.
    """
    mem: Memory = field(default_factory=Memory)
    NrLanes: int = 4
    VLEN: int = 256  # elements per vector (for allocation convenience)

    def __post_init__(self):
        self.vrf: Dict[str, List[int]] = {}
        self.load_unit = LoadUnit(self.mem, self.vrf)
        self.store_unit = StoreUnit(self.mem, self.vrf)

    def write_v(self, reg: str, data: List[int]):
        self.vrf[reg] = list(data)

    def read_v(self, reg: str) -> List[int]:
        return list(self.vrf.get(reg, []))

    def execute(self, req: PeReq, runtime_index: Optional[List[int]] = None) -> PeResp:
        if req.op == "load":
            return self.load_unit.execute(req, runtime_index)
        elif req.op == "store":
            return self.store_unit.execute(req, runtime_index)
        else:
            return PeResp(ok=False, info=f"unknown op {req.op}")
