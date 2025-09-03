
from typing import List
from .types import OpReq, OpResp, DType, is_float, dtype_bits, clip_int, LSUException

class SIMD_DIV:
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
                for i in range(n):
                    if not m[i]:
                        out[i] = a[i]
                        continue
                    if float(b[i]) == 0.0:
                        out[i] = float('inf') if float(a[i]) >= 0.0 else float('-inf')
                    else:
                        out[i] = float(a[i]) / float(b[i])
                return OpResp(True, info=f'SIMD fdiv {req.dtype}', data=out)
            else:
                bits = dtype_bits(req.dtype)
                for i in range(n):
                    if not m[i]:
                        out[i] = a[i]
                        continue
                    if int(b[i]) == 0:
                        # emulate div-by-zero trap by clamping to max value
                        out[i] = clip_int((1<<bits)-1 if req.dtype[0]=='u' else (1<<(bits-1))-1, req.dtype, saturating=True)
                    else:
                        out[i] = clip_int(int(a[i]) // int(b[i]), req.dtype, saturating=req.saturating)
                return OpResp(True, info=f'SIMD idiv {req.dtype}', data=out)
        except Exception as e:
            return OpResp(False, info=str(e), exception=LSUException('SIMD_DIV', str(e)))
