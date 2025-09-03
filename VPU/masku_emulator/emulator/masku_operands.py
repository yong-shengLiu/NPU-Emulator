
from typing import List, Union
from .types import CmpOp, LogicOp

def broadcast_like(b: Union[int, List[int]], n: int) -> List[int]:
    if isinstance(b, list):
        if len(b) == n:
            return b
        if len(b) == 1:
            return b * n
        raise ValueError("Length of b does not match")
    return [int(b)] * n

def _to_signed(x: int, bits: int = 64) -> int:
    mask = (1 << bits) - 1
    x &= mask
    if x & (1 << (bits - 1)):
        return x - (1 << bits)
    return x

def cmp_vec(a: List[int], b: Union[int, List[int]], op: CmpOp, signed: bool = False) -> List[bool]:
    n = len(a)
    bb = broadcast_like(b, n)
    out: List[bool] = [False]*n
    for i in range(n):
        ai = _to_signed(a[i]) if signed else a[i]
        bi = _to_signed(bb[i]) if signed else bb[i]
        if op == "lt":
            out[i] = ai < bi
        elif op == "le":
            out[i] = ai <= bi
        elif op == "eq":
            out[i] = ai == bi
        elif op == "ne":
            out[i] = ai != bi
        elif op == "gt":
            out[i] = ai > bi
        elif op == "ge":
            out[i] = ai >= bi
        else:
            raise ValueError(f"Unknown cmp op {op}")
    return out

def logic2(m0: List[bool], m1: List[bool], op: LogicOp) -> List[bool]:
    if len(m0) != len(m1):
        raise ValueError("Mask length mismatch")
    n = len(m0)
    out: List[bool] = [False]*n
    for i in range(n):
        a, b = m0[i], m1[i]
        if op == "and":
            out[i] = a and b
        elif op == "or":
            out[i] = a or b
        elif op == "xor":
            out[i] = (a != b)
        elif op == "andnot":
            out[i] = a and (not b)
        elif op == "ornot":
            out[i] = a or (not b)
        else:
            raise ValueError(f"Unknown logic op {op}")
    return out

def invert_mask(m: List[bool]) -> List[bool]:
    return [not x for x in m]

def select_by_mask(mask: List[bool], x: List[int], y: List[int]) -> List[int]:
    if len(mask) != len(x) or len(x) != len(y):
        raise ValueError("Length mismatch in select_by_mask")
    return [x[i] if mask[i] else y[i] for i in range(len(mask))]

def apply_mask_zero(mask: List[bool], data: List[int]) -> List[int]:
    if len(mask) != len(data):
        raise ValueError("Length mismatch in apply_mask_zero")
    return [data[i] if mask[i] else 0 for i in range(len(mask))]

def popcount(mask: List[bool]) -> int:
    return sum(1 for v in mask if v)
