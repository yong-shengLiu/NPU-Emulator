
from typing import Dict, List
from .types import OpReq, OpResp, LSUException, is_float
from .valu import VALU
from .simd_mul import SIMD_MUL
from .simd_div import SIMD_DIV

class VMFPU:
    """Top-level Vector Math / Floating-Point Unit (behavioral emulator).
    Routes integer ops to VALU, and mul/div (int/float) to SIMD submodules.
    Maintains a tiny VRF (dict of named vectors).
    """
    def __init__(self):
        self.vrf: Dict[str, List] = {}

    def write_vreg(self, name: str, data: List):
        self.vrf[name] = list(data)

    def read_vreg(self, name: str) -> List:
        return list(self.vrf.get(name, []))

    def execute(self, req: OpReq) -> OpResp:
        try:
            if req.op in ('iadd','isub','iand','ior','ixor','shl','shr','sar','umin','umax','imin','imax'):
                resp = VALU.execute(req)
            elif req.op in ('imul','fmul'):
                resp = SIMD_MUL.execute(req)
            elif req.op in ('idiv','fdiv'):
                resp = SIMD_DIV.execute(req)
            elif req.op in ('fadd','fsub'):
                # emulate via Python float add/sub directly here
                n = len(req.src0)
                a = req.src0
                b = (req.src1 or [0]) * (n if len(req.src1 or [0])==1 else 1)
                if len(b) != n:
                    b = req.src1
                m = req.mask or [True]*n
                out = [0]*n
                for i in range(n):
                    if not m[i]:
                        out[i] = a[i]
                        continue
                    if req.op == 'fadd':
                        out[i] = float(a[i]) + float(b[i])
                    else:
                        out[i] = float(a[i]) - float(b[i])
                resp = OpResp(True, info=f'VMFPU {req.op}', data=out)
            else:
                return OpResp(False, info=f'Unknown op {req.op}')
            if resp.ok and resp.data is not None and req.dst:
                self.vrf[req.dst] = list(resp.data)
            return resp
        except Exception as e:
            return OpResp(False, info=str(e), exception=LSUException('VMFPU', str(e)))
