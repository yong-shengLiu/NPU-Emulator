
from typing import Dict, List
from .types import SlduReq, SlduResp, LSUException
from .memory import Memory
from .p2_stride_gen import P2StrideGen
from .sldu_op_dp import OpDP

class SLDU:
    """Top-level Segment/2D Strided Load Unit (behavioral emulator).
    - Generates 2D-strided addresses (row-major) via P2StrideGen
    - Loads elements via OpDP
    - Writes results to a simple vector register file (VRF) under name 'vd'
    """
    def __init__(self, mem_size: int = 1 << 20):
        self.mem = Memory(mem_size)
        self.vrf: Dict[str, List[int]] = {}

    def read_vreg(self, name: str) -> List[int]:
        return list(self.vrf.get(name, []))

    def execute(self, req: SlduReq) -> SlduResp:
        try:
            addrs = P2StrideGen.addresses(req)
            data = OpDP.load(self.mem, addrs, req.sew, req.little_endian, req.mask)
            self.vrf[req.vd] = data
            return SlduResp(True, info=f"Loaded {len(data)} elements to {req.vd}")
        except IndexError as e:
            return SlduResp(False, info=str(e), exception=LSUException('LoadAccessFault', str(e)))
        except Exception as e:
            return SlduResp(False, info=str(e), exception=LSUException('SLDU', str(e)))
