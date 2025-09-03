
from typing import List, Optional
from .types import OpReq, OpResp, DType, dtype_bits, is_float, is_signed, clip_int, LSUException

class VALU:
    """Vector integer/logic ALU (behavioral).
    Supports: iadd, isub, iand, ior, ixor, shl, shr (logical), sar (arith), umin, umax, imin, imax
    """
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
        if is_float(req.dtype):
            return OpResp(False, info='VALU handles integer ops only')
        n = len(req.src0)
        a = req.src0
        b = cls._broadcast(req.src1 if req.src1 is not None else [0], n)
        m = req.mask or [True]*n
        bits = dtype_bits(req.dtype)
        signed = req.dtype[0] == 'i'
        out: List[int] = [0]*n
        try:
            for i in range(n):
                if not m[i]:
                    out[i] = a[i]  # pass-through
                    continue
                op = req.op
                if op == 'iadd':
                    val = a[i] + b[i]
                elif op == 'isub':
                    val = a[i] - b[i]
                elif op == 'iand':
                    val = a[i] & b[i]
                elif op == 'ior':
                    val = a[i] | b[i]
                elif op == 'ixor':
                    val = a[i] ^ b[i]
                elif op == 'shl':
                    val = a[i] << (b[i] & 0x3F)
                elif op == 'shr':  # logical right
                    val = (a[i] & ((1<<bits)-1)) >> (b[i] & 0x3F)
                elif op == 'sar':  # arithmetic right
                    if signed:
                        # convert to signed then shift
                        ai = a[i] & ((1<<bits)-1)
                        if ai & (1<<(bits-1)):
                            ai = ai - (1<<bits)
                        val = ai >> (b[i] & 0x3F)
                    else:
                        val = (a[i] & ((1<<bits)-1)) >> (b[i] & 0x3F)
                elif op == 'umin':
                    val = a[i] if a[i] < b[i] else b[i]
                elif op == 'umax':
                    val = a[i] if a[i] > b[i] else b[i]
                elif op == 'imin':
                    # interpret as signed
                    def to_signed(x: int) -> int:
                        x &= (1<<bits)-1
                        return x - (1<<bits) if x & (1<<(bits-1)) else x
                    ai, bi = to_signed(a[i]), to_signed(b[i])
                    val = a[i] if ai <= bi else b[i]
                elif op == 'imax':
                    def to_signed(x: int) -> int:
                        x &= (1<<bits)-1
                        return x - (1<<bits) if x & (1<<(bits-1)) else x
                    ai, bi = to_signed(a[i]), to_signed(b[i])
                    val = a[i] if ai >= bi else b[i]
                else:
                    return OpResp(False, info=f'Unsupported ALU op {op}')
                out[i] = clip_int(val, req.dtype, saturating=req.saturating)
            return OpResp(True, info=f'VALU {req.op} {req.dtype}', data=out)
        except Exception as e:
            return OpResp(False, info=str(e), exception=LSUException('VALU', str(e)))
