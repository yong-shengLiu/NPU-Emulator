# VLSU Emulator (Python)

This is a **behavioral** Python emulator for a Vector Load/Store Unit (VLSU), inspired by your RTL modules: `vlsu.sv` (top), `vldu.sv`, `vstu.sv`, and `addrgen.sv`.
It focuses on clarity and correctness of addressing modes rather than cycle-accurate micro-architecture.

## Features
- Unit-stride, Strided, and Indexed addressing
- Per-element mask
- Configurable element widths (SEW = 8/16/32/64)
- Simple byte-addressable memory model
- Minimal VRF (vector register file) with named vector registers (e.g., `v0`, `v1`)

## Layout
```
vlsu_emulator/
  emulator/
    __init__.py
    types.py
    memory.py
    addrgen.py
    vldu.py
    vstu.py
    vlsu.py
  run_demo.py
  README.md
```

## Quick Start
```bash
python /mnt/data/vlsu_emulator/run_demo.py
```

Expected output shows:
1. A unit-stride load into `v0`
2. A strided store from `v1`
3. An indexed masked load into `v2`

You can import the library and drive `VLSU.execute(PeReq(...))` with your own scenarios.

## Notes
- This is not cycle-accurate and does not model exceptions beyond simple range checks.
- Endianness defaults to little-endian to match common RISC-V vector systems; set `little_endian=False` in `PeReq` to switch.
- Extend `PeReq`/`PeResp` and the units as needed (segment loads, faults-first, misalignment, sign-extension, etc.).
