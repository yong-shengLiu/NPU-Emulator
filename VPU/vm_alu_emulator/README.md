
# VMFPU / VALU / SIMD Emulator (Python)

A **behavioral** Python emulator inspired by your RTL:
- `vmfpu.sv` (top)    → `emulator/vmfpu.py` (VMFPU top)
- `valu.sv`           → `emulator/valu.py` (integer/logic ALU)
- `simd_mul.sv`       → `emulator/simd_mul.py` (elementwise mul)
- `simd_div.sv`       → `emulator/simd_div.py` (elementwise div)

## Features
- Integer ops: add/sub/and/or/xor/shl/shr/sar/umin/umax/imin/imax (wrap or saturating)
- Float ops: fadd/fsub/fmul/fdiv (using Python float math)
- Per-lane mask (inactive lanes pass-through)
- Tiny VRF dictionary with named vector registers (e.g., `v0`, `vf0`)

## Layout
```
vm_alu_emulator/
  emulator/
    __init__.py
    types.py
    valu.py
    simd_mul.py
    simd_div.py
    vmfpu.py
  run_demo.py
  README.md
```

## Quick Start
```bash
python /mnt/data/vm_alu_emulator/run_demo.py
```

## Notes
- This is not cycle-accurate; it models functional behavior to help with ISA/algorithm checks.
- If your RTL has more specific semantics (rounding modes, exceptions, NaNs, denormals, etc.), we can extend the emulator accordingly.
