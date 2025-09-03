
from typing import List, Optional
from .types import PeReq, LSUException

class AddrGen:
    """High-level address generator for vector loads/stores.
    This is a behavioral model of Ara's addrgen block, supporting three addressing modes:
    - unit-stride   : base + i * elem_bytes
    - strided       : base + i * stride
    - indexed       : base + index[i] * (elem_bytes or 1 if index_is_bytes)
    The real RTL is more feature-rich (segment loads, faults-first, exceptions, etc.).
    """

    @staticmethod
    def element_addresses(req: PeReq, runtime_index: Optional[List[int]] = None) -> List[int]:
        vl = req.vl
        elem_bytes = req.sew // 8
        base = req.base
        if req.addr_mode == "unit":
            return [base + i * elem_bytes for i in range(vl)]
        elif req.addr_mode == "strided":
            return [base + i * req.stride for i in range(vl)]
        elif req.addr_mode == "indexed":
            idx = req.index if req.index is not None else (runtime_index or [])
            if len(idx) < vl:
                raise LSUException(kind="AddrGen", detail="insufficient index length", vstart=len(idx))
            scale = 1 if req.index_is_bytes else elem_bytes
            return [base + int(idx[i]) * scale for i in range(vl)]
        else:
            raise LSUException(kind="AddrGen", detail=f"unknown addr_mode {req.addr_mode}")
