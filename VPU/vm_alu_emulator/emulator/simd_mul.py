
from typing import List
from .types import OpReq, OpResp, DType, is_float, dtype_bits, clip_int, LSUException

class SIMD_MUL:
    @staticmethod
    def _broadcast(v: List[int], n: int) -> List[int]:
        if v is None:
            return [0]*n
        if len(v) == n:
            return v
        if len(v) == 1:
            return v * n
        raise ValueError("Operand length mismatch")

    @classmethod
    def execute(cls, req: OpReq) -> OpResp:
        n = len(req.src0)
        a = req.src0
        b = cls._broadcast(req.src1 if req.src1 is not None else [0], n)
        m = req.mask or [True]*n
        out: List[int] = [0]*n
        try:
            if is_float(req.dtype):
                # treat inputs as Python floats
                for i in range(n):
                    out[i] = a[i] if not m[i] else float(a[i]) * float(b[i])
                return OpResp(True, info=f'SIMD fmul {req.dtype}', data=out)
            else:
                bits = dtype_bits(req.dtype)
                for i in range(n):
                    val = a[i] if not m[i] else int(a[i]) * int(b[i])
                    out[i] = clip_int(val, req.dtype, saturating=req.saturating)
                return OpResp(True, info=f'SIMD imul {req.dtype}', data=out)
        except Exception as e:
            return OpResp(False, info=str(e), exception=LSUException('SIMD_MUL', str(e)))
