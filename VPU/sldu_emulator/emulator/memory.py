
from dataclasses import dataclass, field

@dataclass
class Memory:
    size: int = 1 << 20  # 1 MiB
    mem: bytearray = field(init=False)

    def __post_init__(self):
        self.mem = bytearray(self.size)

    def check(self, addr: int, nbytes: int):
        if addr < 0 or addr + nbytes > self.size:
            raise IndexError(f"Memory access OOB: addr=0x{addr:x}, size={nbytes}")

    def read_bytes(self, addr: int, nbytes: int) -> bytes:
        self.check(addr, nbytes)
        return bytes(self.mem[addr:addr+nbytes])

    def write_bytes(self, addr: int, data: bytes):
        self.check(addr, len(data))
        self.mem[addr:addr+len(data)] = data

    def fill_u32_inc(self, base: int, count: int):
        """Write 'count' 32-bit little-endian ascending ints starting at base."""
        for i in range(count):
            self.write_bytes(base + 4*i, int(i).to_bytes(4, 'little'))
