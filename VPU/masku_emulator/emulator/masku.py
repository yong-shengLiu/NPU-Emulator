
from typing import Dict, List
from .types import MaskReq, MaskResp, LSUException
from .masku_operands import (
    cmp_vec, logic2, invert_mask, select_by_mask, apply_mask_zero, popcount
)

class MaskU:
    def __init__(self):
        self.mrf: Dict[str, List[bool]] = {}

    def read_mask(self, name: str) -> List[bool]:
        return list(self.mrf.get(name, []))

    def write_mask(self, name: str, m: List[bool]):
        self.mrf[name] = list(m)

    def execute(self, req: MaskReq) -> MaskResp:
        try:
            if req.mode == "cmp":
                m = cmp_vec(req.a, req.b, req.cmp_op, signed=req.signed)
                if req.md:
                    self.write_mask(req.md, m)
                return MaskResp(True, f"cmp->{req.md or '<temp>'}", mask=m, popcnt=popcount(m))

            elif req.mode == "logic":
                m = logic2(req.m0, req.m1, req.logic_op)
                if req.md:
                    self.write_mask(req.md, m)
                return MaskResp(True, f"logic->{req.md or '<temp>'}", mask=m, popcnt=popcount(m))

            elif req.mode == "invert":
                m = invert_mask(req.m0)
                if req.md:
                    self.write_mask(req.md, m)
                return MaskResp(True, f"invert->{req.md or '<temp>'}", mask=m, popcnt=popcount(m))

            elif req.mode == "select":
                data = select_by_mask(req.mask, req.x, req.y)
                return MaskResp(True, "select->data", data=data)

            elif req.mode == "apply":
                data = apply_mask_zero(req.mask, req.data)
                return MaskResp(True, "apply->data", data=data)

            else:
                return MaskResp(False, f"Unknown mode {req.mode}")

        except Exception as e:
            return MaskResp(False, str(e), exception=LSUException("MaskU", str(e)))
