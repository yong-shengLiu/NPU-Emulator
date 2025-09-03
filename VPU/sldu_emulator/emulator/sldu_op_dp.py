
from typing import List, Optional
from .memory import Memory

def bytes_to_int(b: bytes, little: bool) -> int:
    return int.from_bytes(b, byteorder='little' if little else 'big', signed=False)

class OpDP:
    """Datapath for SLDU: performs memory reads and element unpacking.
    """
    @staticmethod
    def load(mem: Memory, addresses: List[int], sew: int, little_endian: bool,
             mask: Optional[List[bool]] = None) -> List[int]:
        elem_bytes = sew // 8
        out: List[int] = [0] * len(addresses)
        active = mask if mask is not None else [True] * len(addresses)
        if len(active) < len(addresses):
            raise ValueError("mask shorter than number of addresses")
        for i, addr in enumerate(addresses):
            if not active[i]:
                continue
            raw = mem.read_bytes(addr, elem_bytes)
            out[i] = bytes_to_int(raw, little_endian)
        return out
