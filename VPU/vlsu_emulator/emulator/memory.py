
from typing import List
from dataclasses import dataclass, field

@dataclass
class Memory:
    """Simple byte-addressable memory model using a Python bytearray.
    The default size is 1 MiB, but you can pass a different size at construction.
    """
    size: int = 1 << 20  # 1 MiB
    mem: bytearray = field(init=False)

    def __post_init__(self):
        self.mem = bytearray(self.size)

    def check_range(self, addr: int, nbytes: int):
        if addr < 0 or addr + nbytes > self.size:
            raise IndexError(f"Memory access out of range: addr=0x{addr:x}, size={nbytes}")

    def read_bytes(self, addr: int, nbytes: int) -> bytes:
        self.check_range(addr, nbytes)
        return bytes(self.mem[addr:addr+nbytes])

    def write_bytes(self, addr: int, data: bytes):
        self.check_range(addr, len(data))
        self.mem[addr:addr+len(data)] = data

    def fill_pattern(self, start: int, nbytes: int, pattern: int = 0xA5):
        self.check_range(start, nbytes)
        for i in range(nbytes):
            self.mem[start + i] = (pattern + i) & 0xFF
