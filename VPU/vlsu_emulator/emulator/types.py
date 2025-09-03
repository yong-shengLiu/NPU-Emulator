
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict

AddrMode = Literal["unit", "strided", "indexed"]

@dataclass
class PeReq:
    """Minimal instruction request structure for the emulator.
    - op: 'load' or 'store'
    - addr_mode: 'unit' | 'strided' | 'indexed'
    - base: base byte address (int)
    - vl: vector length (number of elements)
    - sew: element bit width (8, 16, 32, 64)
    - stride: byte stride (only for 'strided')
    - index: per-element signed/unsigned byte offsets (only for 'indexed'); multiplied by (sew//8) unless index_is_bytes=True
    - index_is_bytes: if True, 'index' values are used directly as byte offsets
    - mask: optional per-element boolean mask (True = active, False = masked off)
    - vd: destination vector register name for loads (e.g., 'v0')
    - vs: source vector register name for stores (e.g., 'v0')
    - index_v: name of vector register holding indices when addr_mode='indexed' and index is not given inline
    - little_endian: memory endianness for element packing/unpacking
    """
    op: Literal["load", "store"]
    addr_mode: AddrMode
    base: int
    vl: int
    sew: int
    stride: int = 0
    index: Optional[List[int]] = None
    index_is_bytes: bool = False
    mask: Optional[List[bool]] = None
    vd: Optional[str] = None
    vs: Optional[str] = None
    index_v: Optional[str] = None
    little_endian: bool = True

@dataclass
class PeResp:
    ok: bool
    info: str = ""
    exception: Optional["LSUException"] = None

@dataclass
class LSUException(Exception):
    kind: str
    detail: str = ""
    vstart: Optional[int] = None

    def __str__(self) -> str:
        s = f"{self.kind}: {self.detail}"
        if self.vstart is not None:
            s += f" (at element {self.vstart})"
        return s
