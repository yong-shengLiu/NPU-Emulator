
from typing import List
from .types import SlduReq

class P2StrideGen:
    """Phase-2 (2D) stride address generator.
    Produces a flat list of addresses for a (outer_count x inner_count) tile.
    """
    @staticmethod
    def addresses(req: SlduReq) -> List[int]:
        elem_bytes = req.sew // 8
        scale = 1 if req.stride_unit == 'bytes' else elem_bytes
        addrs: List[int] = []
        for r in range(req.outer_count):
            row_base = req.base + r * req.outer_stride * scale
            for c in range(req.inner_count):
                addrs.append(row_base + c * req.inner_stride * scale)
        return addrs
