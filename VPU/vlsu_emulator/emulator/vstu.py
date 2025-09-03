
from typing import List, Dict, Optional
from .types import PeReq, PeResp, LSUException
from .addrgen import AddrGen

class StoreUnit:
    """Behavioral vector store unit.
    - Reads elements from the VRF (req.vs) and writes to memory at addresses from AddrGen.
    - Respects optional per-element mask (masked-off elements do not write).
    """
    def __init__(self, mem, vrf: Dict[str, List[int]]):
        self.mem = mem
        self.vrf = vrf

    @staticmethod
    def _pack(value: int, sew: int, little_endian: bool) -> bytes:
        if sew == 8:
            return bytes([value & 0xFF])
        n = sew // 8
        return int(value & ((1 << (8*n)) - 1)).to_bytes(n, byteorder='little' if little_endian else 'big', signed=False)

    def execute(self, req: PeReq, runtime_index: Optional[List[int]] = None) -> PeResp:
        if req.op != "store" or not req.vs:
            return PeResp(ok=False, info="invalid store request: missing vs or wrong op")
        try:
            if req.vs not in self.vrf or len(self.vrf[req.vs]) < req.vl:
                raise LSUException(kind="VRF", detail=f"source {req.vs} too short or missing")
            addrs = AddrGen.element_addresses(req, runtime_index)
            elem_bytes = req.sew // 8
            mask = req.mask or [True] * req.vl
            for i in range(req.vl):
                if not mask[i]:
                    continue
                data = self._pack(self.vrf[req.vs][i], req.sew, req.little_endian)
                self.mem.write_bytes(addrs[i], data)
            return PeResp(ok=True, info=f"stored {req.vl} elements from {req.vs}")
        except LSUException as e:
            return PeResp(ok=False, info=str(e), exception=e)
        except IndexError as e:
            ex = LSUException(kind="Memory", detail=str(e))
            return PeResp(ok=False, info=str(ex), exception=ex)
