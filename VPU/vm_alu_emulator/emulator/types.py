
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

DType = Literal['i8','i16','i32','i64','u8','u16','u32','u64','f16','f32','f64']

def dtype_bits(dtype: DType) -> int:
    return int(dtype[1:]) if dtype[0] in ('i','u','f') else 32

def is_float(dtype: DType) -> bool:
    return dtype[0] == 'f'

def is_signed(dtype: DType) -> bool:
    return dtype[0] == 'i'

def mask_to_bits(x: int, bits: int) -> int:
    return x & ((1<<bits)-1)

def to_signed(x: int, bits: int) -> int:
    x &= (1<<bits)-1
    if x & (1<<(bits-1)):
        return x - (1<<bits)
    return x

def clip_int(x: int, dtype: DType, saturating: bool=False) -> int:
    bits = dtype_bits(dtype)
    if saturating:
        if dtype[0] == 'u':
            lo, hi = 0, (1<<bits)-1
        else:
            lo, hi = -(1<<(bits-1)), (1<<(bits-1))-1
        return max(lo, min(hi, int(x)))
    # wrap-around
    if dtype[0] == 'u':
        return mask_to_bits(int(x), bits)
    else:
        return to_signed(int(x), bits)

def cast_from_float(x: float, dtype: DType, saturating: bool=False) -> int:
    return clip_int(int(x), dtype, saturating=saturating)

def cast_to_float(x: int, src_dtype: DType) -> float:
    # interpret integer bits as signed/unsigned value already in x
    return float(x)

@dataclass
class LSUException(Exception):
    kind: str
    detail: str = ""
    index: Optional[int] = None
    def __str__(self) -> str:
        s = f"{self.kind}: {self.detail}"
        if self.index is not None:
            s += f" (at lane {self.index})"
        return s

@dataclass
class OpReq:
    """Vector operation request.
    op: operation string, e.g. 'iadd','isub','iand','ior','ixor','shl','shr','sar','umin','umax','imin','imax','fmul','fdiv','fadd','fsub'
    dtype: element type (i8/i16/i32/i64/u8/u16/u32/u64/f16/f32/f64)
    src0: first operand vector
    src1: optional second operand vector (or scalar if len==1)
    mask: optional per-lane activity mask (True=active). If not set, all lanes active.
    dst: name of destination vector register (informational; VMFPU keeps a small VRF)
    saturating: for integer ops, enable saturation instead of wrap
    """
    op: str
    dtype: DType
    src0: List[int]  # for floats, pass int bit-pattern or just floats; emulator treats python floats for f* ops
    src1: Optional[List[int]] = None
    mask: Optional[List[bool]] = None
    dst: str = 'v0'
    saturating: bool = False

@dataclass
class OpResp:
    ok: bool
    info: str = ""
    data: Optional[List[int]] = None
    exception: Optional[LSUException] = None
