
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict

DType = Literal['i8','i16','i32','i64','u8','u16','u32','u64']

RoundMode = Literal['trunc','floor','ceil','nearest_even']

@dataclass
class LaneConfig:
    vlen: int = 8             # lanes per vector
    nregs: int = 32           # vector registers
    dtype: DType = 'i16'      # default element dtype

def dtype_bits(dtype: DType) -> int:
    return int(dtype[1:])

def is_signed(dtype: DType) -> bool:
    return dtype[0] == 'i'

def to_signed(x: int, bits: int) -> int:
    x &= (1<<bits)-1
    return x - (1<<bits) if x & (1<<(bits-1)) else x

def clip(x: int, dtype: DType) -> int:
    bits = dtype_bits(dtype)
    if is_signed(dtype):
        lo, hi = -(1<<(bits-1)), (1<<(bits-1))-1
        return max(lo, min(hi, int(x)))
    else:
        return int(x) & ((1<<bits)-1)

@dataclass
class LSUException(Exception):
    kind: str
    detail: str = ""
    index: Optional[int] = None
    def __str__(self) -> str:
        s = f"{self.kind}: {self.detail}"
        if self.index is not None:
            s += f" (lane {self.index})"
        return s

@dataclass
class MicroOp:
    """A minimal micro-op describing a lane ALU operation.
    op: 'add','sub','and','or','xor','shl','shr','sar' (arith right)
    rd: destination vreg index
    rs1, rs2: source vreg indices (rs2 may be None, then imm used or pass-through)
    imm: optional scalar immediate (e.g., shift amount)
    mask: optional per-lane mask (True=active). If None → all active.
    dtype: element type (affects width/sign and clipping)
    round_mode: fixed-point rounding mode (used for shift-right ops)
    saturating: if True, saturate on overflow for add/sub
    info: free-form text
    """
    op: str
    rd: int
    rs1: int
    rs2: Optional[int] = None
    imm: Optional[int] = None
    mask: Optional[List[bool]] = None
    dtype: DType = 'i16'
    round_mode: RoundMode = 'trunc'
    saturating: bool = False
    info: str = ""
