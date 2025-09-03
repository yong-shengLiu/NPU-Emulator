
from dataclasses import dataclass
from typing import List, Optional, Literal, Union

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

CmpOp = Literal["lt", "le", "eq", "ne", "gt", "ge"]
LogicOp = Literal["and", "or", "xor", "andnot", "ornot"]

@dataclass
class MaskReq:
    mode: Literal["cmp", "logic", "invert", "select", "apply"]
    md: Optional[str] = None
    a: Optional[List[int]] = None
    b: Optional[Union[List[int], int]] = None
    cmp_op: Optional[CmpOp] = None
    signed: bool = False
    m0: Optional[List[bool]] = None
    m1: Optional[List[bool]] = None
    logic_op: Optional[LogicOp] = None
    mask: Optional[List[bool]] = None
    x: Optional[List[int]] = None
    y: Optional[List[int]] = None
    data: Optional[List[int]] = None

@dataclass
class MaskResp:
    ok: bool
    info: str = ""
    mask: Optional[List[bool]] = None
    data: Optional[List[int]] = None
    popcnt: Optional[int] = None
    exception: Optional[LSUException] = None
