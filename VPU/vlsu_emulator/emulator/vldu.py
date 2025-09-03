
from typing import List, Dict, Optional
from .types import PeReq, PeResp, LSUException
from .addrgen import AddrGen

class LoadUnit:
    """Behavioral vector load unit.
    - Reads elements from memory at addresses computed by AddrGen.
    - Writes results into the VLSU's vector register file (VRF).
    - Respects optional per-element mask (masked-off elements are left unchanged).
    """
    def __init__(self, mem, vrf: Dict[str, List[int]]):
        self.mem = mem
        self.vrf = vrf

    @staticmethod
    def _unpack(data: bytes, sew: int, little_endian: bool) -> int:
        if sew == 8:
            return data[0]
        n = sew // 8
        return int.from_bytes(data[:n], byteorder='little' if little_endian else 'big', signed=False)

    def execute(self, req: PeReq, runtime_index: Optional[List[int]] = None) -> PeResp:
        if req.op != "load" or not req.vd:
            return PeResp(ok=False, info="invalid load request: missing vd or wrong op")
        try:
            addrs = AddrGen.element_addresses(req, runtime_index)
            elem_bytes = req.sew // 8
            mask = req.mask or [True] * req.vl
            # Ensure VRF destination exists
            if req.vd not in self.vrf:
                self.vrf[req.vd] = [0] * req.vl
            for i in range(req.vl):
                if not mask[i]:
                    continue
                val = self._unpack(self.mem.read_bytes(addrs[i], elem_bytes), req.sew, req.little_endian)
                self.vrf[req.vd][i] = val
            return PeResp(ok=True, info=f"loaded {req.vl} elements into {req.vd}")
        except LSUException as e:
            return PeResp(ok=False, info=str(e), exception=e)
        except IndexError as e:
            ex = LSUException(kind="Memory", detail=str(e))
            return PeResp(ok=False, info=str(ex), exception=ex)
