
from dataclasses import dataclass
from typing import List, Optional, Literal

@dataclass
class LSUException(Exception):
    kind: str
    detail: str = ""
    index: Optional[int] = None
    def __str__(self) -> str:
        s = f"{self.kind}: {self.detail}"
        if self.index is not None:
            s += f" (at element {self.index})"
        return s

@dataclass
class SlduReq:
    """Request to SLDU (Segment/2D Strided Load Unit).
    This emulates a 2D nested-loop address generation:
        for r in range(outer_count):
            for c in range(inner_count):
                addr = base + r*outer_stride*scale + c*inner_stride*scale
    where scale = elem_bytes if stride_unit == 'elems' else 1.
    Results are flattened in row-major into a single destination vector register.
    """
    base: int                      # base byte address
    sew: int                       # element bit-width: 8/16/32/64
    inner_count: int               # number of elements per row
    outer_count: int               # number of rows
    inner_stride: int              # stride between elements within a row
    outer_stride: int              # stride between rows
    stride_unit: Literal['bytes','elems'] = 'bytes'
    little_endian: bool = True
    vd: str = 'v0'                 # destination vector register name
    mask: Optional[List[bool]] = None   # optional mask of length outer_count*inner_count

@dataclass
class SlduResp:
    ok: bool
    info: str = ""
    exception: Optional[LSUException] = None
