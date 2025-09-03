
from typing import List, Optional
from .types import MicroOp, DType, dtype_bits, is_signed, to_signed, clip
from .fixed_p_rounding import FixedPRounding

class SIMD_ALU:
    @staticmethod
    def exec(op: MicroOp, a: List[int], b: Optional[List[int]]) -> List[int]:
        n = len(a)
        m = op.mask or [True]*n
        out: List[int] = [0]*n
        bits = dtype_bits(op.dtype)
        signed = is_signed(op.dtype)
        for i in range(n):
            ai = a[i]
            bi = (b[i] if b is not None else 0)
            if op.imm is not None and b is None:
                bi = op.imm
            if not m[i]:
                out[i] = a[i]  # pass-through
                continue
            if op.op == 'add':
                val = ai + bi
                if op.saturating:
                    if signed:
                        lo, hi = -(1<<(bits-1)), (1<<(bits-1))-1
                        val = max(lo, min(hi, val))
                    else:
                        lo, hi = 0, (1<<bits)-1
                        val = max(lo, min(hi, val))
                out[i] = clip(val, op.dtype)
            elif op.op == 'sub':
                val = ai - bi
                if op.saturating:
                    if signed:
                        lo, hi = -(1<<(bits-1)), (1<<(bits-1))-1
                        val = max(lo, min(hi, val))
                    else:
                        lo, hi = 0, (1<<bits)-1
                        val = max(lo, min(hi, val))
                out[i] = clip(val, op.dtype)
            elif op.op == 'and':
                out[i] = clip(ai & bi, op.dtype)
            elif op.op == 'or':
                out[i] = clip(ai | bi, op.dtype)
            elif op.op == 'xor':
                out[i] = clip(ai ^ bi, op.dtype)
            elif op.op == 'shl':
                out[i] = clip(ai << (bi & 0x3F), op.dtype)
            elif op.op == 'shr':
                out[i] = clip((ai & ((1<<bits)-1)) >> (bi & 0x3F), op.dtype)
            elif op.op == 'sar':
                # arithmetic right
                x = ai & ((1<<bits)-1)
                if signed and (x & (1<<(bits-1))):
                    x = x - (1<<bits)
                if op.round_mode != 'trunc' and bi > 0:
                    val = FixedPRounding.round_right_shift(x, bi, op.round_mode, signed, bits)
                else:
                    val = x >> (bi & 0x3F)
                out[i] = clip(val, op.dtype)
            else:
                raise ValueError(f"unsupported op {op.op}")
        return out
